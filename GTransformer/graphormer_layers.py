import math
import torch
import torch.nn as nn
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        # 1 for graph token
        self.ae = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.de = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.doe = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.gt = nn.Embedding(1, hidden_dim) #graph token embedding
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, nf, di, dout):
        ng, nn = nf.size()[:2] #graph and node number
        # node feauture + graph token
        nf = self.ae(nf).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        nf = (nf + self.de(di)+ self.doe(dout))
        gtf = self.gt.weight.unsqueeze(0).repeat(ng, 1, 1)
        gtf = gtf.sum(-2)
        gnf = torch.cat([gtf, ng], dim=1)
        return gnf
class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(self,num_heads,num_atoms,num_edges,num_spatial,num_edge_dis,hidden_dim,edge_type,multi_hop_max_dist,n_layers):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, ab, sp, nf, ei, atnet):
        ng, nn = nf.size()[:2] #number of graph and number of nodes
        gatn = ab.clone()
        # [n_graph, n_head, n_node+1, n_node+1]
        gatn = gatn.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  
        # spatial pos
        sp = torch.from_numpy(sp).long().unsqueeze(1).repeat(1, 1, 1)
        # spatial position bias
        # [n_graph, n_head, n_node, n_node]
        spb = self.spatial_pos_encoder(sp).permute(1, 0, 3, 2) 
        gatn[:, :, :, :] = gatn[:, :, :, :] + spb 
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        gatn[:, :, 1:, 0] = gatn[:, :, 1:, 0] + t
        gatn[:, :, 0, :] = gatn[:, :, 0, :] + t
        # edge feature ==> #edge encoding [ee]
        ee = self.edge_encoder(atnet).permute(0, 1, 2, 3) 
        ee = ee.sum(-2)
        ee = ee.repeat(1,1,1,1)
        ee = ee.permute(0, 1, 3, 2)
        gatn += ee
        gatn = gatn + atnet.unsqueeze(1).permute(0,1,3,2).sum(-2).unsqueeze(3).permute(1,0,3,2)  # reset
        gatn = gatn.permute(0,2,1,3) # [n_graph, n_head, n_node, n_node]
        return gatn
