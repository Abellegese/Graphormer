from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
import torch
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
import numpy as np
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group


# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
out_dir = 'out'
import os
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8 # simulate 8 gpus

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
class SmileDataset(object):
    """SMILE Dataset: For Graph Level Prediction"""
    def __init__(self, dataset_name='ogbg-molhiv',bls=8, bs=32):
        self.dataset_name = dataset_name # small dataset as default ~ 41,000
        self.atom_dims = get_atom_feature_dims()
        self.edge_dims = get_bond_feature_dims()
        self.bls = bls
        self.bs = bs
    def get_data(self):
        data = np.genfromtxt(self.path, delimiter=",", dtype=str)
        return data
    def construct(self, split):
        #the dataset will be constructed based on DGL graph object
        ds = DglGraphPropPredDataset(name = 'ogbg-molhiv')
        sidx = ds.get_idx_split()
        trl = DataLoader(ds[sidx[split]], batch_size=32, shuffle=True, collate_fn=collate_dgl) #train loader
        vdl = DataLoader(ds[sidx[split]], batch_size=32, shuffle=False, collate_fn=collate_dgl) #valid loader
        tsl = DataLoader(ds[sidx[split]], batch_size=32, shuffle=False, collate_fn=collate_dgl) #test loader
        if split == "train": return trl
        elif split == "valid": return trl
        else: return tsl
    def get_inputs(self, graph, label):
        #graph and label are batch data
        ei = graph.edges()
        nf = self.convert_to_single_emb(graph.ndata['feat']) #node feature
        N = nf.shape[0] # Number of nodes
        ef = self.convert_to_single_emb(graph.edata['feat']) #edge feature
        aet = torch.zeros([N, N, ef.shape[1]], dtype=torch.long) #attention edge type
        aet[ei[0].long(), ei[1].long()] = self.convert_to_single_emb(ef)
        di = graph.in_degrees() #degree in
        dout = di # for undirected graph -> degree in == degree out
        #computing adjacency matrix for distance computation
        adj = graph.adj().to_dense().type(torch.int) 
        adj = csr_matrix(adj)
        sp, pred = floyd_warshall(csgraph=adj, directed=False, return_predecessors=True) #shortest path
        #setting unreachable paths from infinity -> 510
        for i in range(sp.shape[0]):
            for j in range(sp.shape[1]):
                if sp[i][j] == np.inf:
                    sp[i][j] = 510
        maxd = np.amax(sp) # maximum distance
        ab = torch.zeros([N, N], dtype=torch.float) #attention bias
        return nf, ef, aet, sp, di, dout, maxd, ab
    def get_batch(self, split):
        #batch version of the data for training
        trl = self.construct(split)
        graph, label = next(iter(trl))
        nf, ef, aet, sp, di, dout, maxd, ab  = self.get_inputs(graph, label)
        return nf, ef, aet, sp, di, dout, maxd, ab
    def convert_to_single_emb(self, x, offset: int = 512):
        feature_num = x.size(1) if len(x.size()) > 1 else 1
        feature_offset = 1 + torch.arange(0, feature_num * offset, offset)
        x = x + feature_offset
        return x
    def stack(self, x):
        x = torch.from_numpy(np.concatenate(x)).flatten()
        ix = torch.randint(len(x) - self.bs, (self.bs,))
        x = torch.stack([(x[i:i+self.bs]) for i in ix])
        return x