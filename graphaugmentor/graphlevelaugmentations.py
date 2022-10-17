# Adapted and modified from https://github.com/Shen-Lab/GraphCL/blob/master/semisupervised_TU/pre-training/tu_dataset.py
import torch
import numpy as np
from copy import deepcopy
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import inspect
import multiprocessing as mp


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    data.x = data.x.type('torch.FloatTensor')
    token = data.x.mean(dim=0)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data

def mask_edges(data, aug_ratio):
    _, edge_num = data.edge_index.size()
    mask_num = int(edge_num * aug_ratio)

    data.edge_attr = data.edge_attr.type('torch.FloatTensor')
    # temp_tensor = data.x.type('torch.FloatTensor')
    # token = temp_tensor.mean(dim=0)
    token = data.edge_attr.mean(dim=0)

    idx_mask = np.random.choice(edge_num, mask_num, replace=False)
    data.edge_attr[idx_mask] = torch.tensor(token, dtype=torch.float32)
    # new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, num_nodes=data.x.size()[0])
    # data.num_nodes = data.x.size()
    return data

def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()

    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    print(idx_nondrop)

    # idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

    edge_index = data.edge_index.numpy()

    # When dealing with batch, adjust the batch (graph-level) information:

    if data.batch is not None:
        data.batch = data.batch[idx_nondrop]

    if hasattr(data, 'ptr'):
        if data.ptr is not None:
            curr_b_idx = -1
            new_ptr = []
            for t_idx, b_idx in enumerate(data.batch.cpu().detach().numpy()):
                if b_idx == curr_b_idx+1:
                    new_ptr.append(t_idx)
                    curr_b_idx+=1
            new_ptr.append(data.batch.size(dim=0)) #append last index also
        data.ptr = torch.tensor(new_ptr)

    if data.edge_attr is not None:
        mask = np.isin(edge_index,idx_nondrop)
        edge_idx_nondrop = [idx for idx in range(len(mask[0])) if mask[0][idx]==False or mask[1][idx]==False]
        data.edge_attr = data.edge_attr[edge_idx_nondrop, :]

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_nondrop, :][:, idx_nondrop]
    edge_index = adj.nonzero().t()

    # new_data = Data(x=data.x[idx_nondrop], edge_index=edge_index, y=data.y, num_nodes=data.x[idx_nondrop].size()[0])

    data.edge_index = edge_index
    data.x = data.x[idx_nondrop]


    '''
    try:
        # data = Data(x=data.x[idx_nondrop], edge_index=edge_index)
        data.edge_index = edge_index
        data.x = data.x[idx_nondrop]

    except:
        print('except')
        data = data

    # data.num_nodes = data.x.size()
    '''
    return data

def drop_edges_undirected(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_nondrop = np.sort(np.random.choice(edge_num, (edge_num - permute_num), replace=False))
    print(idx_nondrop)
    edge_index = edge_index[:, idx_nondrop]
    data.edge_index = torch.tensor(edge_index)

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[idx_nondrop, :]
    # new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, num_nodes=data.x.size()[0])
    # data.num_nodes = data.x.size()
    return data

def drop_edges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_nondrop = np.sort(np.random.choice(edge_num, (edge_num - permute_num), replace=False))
    print(idx_nondrop)
    edge_index = edge_index[:, idx_nondrop]
    data.edge_index = torch.tensor(edge_index)

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[idx_nondrop, :]
    # new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, num_nodes=data.x.size()[0])
    # data.num_nodes = data.x.size()
    return data



