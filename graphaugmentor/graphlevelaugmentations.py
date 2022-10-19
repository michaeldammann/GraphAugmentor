# Adapted and modified from https://github.com/Shen-Lab/GraphCL/blob/master/semisupervised_TU/pre-training/tu_dataset.py
import torch
import numpy as np
from torch_geometric.data import Data

def new_ptr(batch):
    curr_b_idx = -1
    new_ptr = []
    for t_idx, b_idx in enumerate(batch):
        if b_idx == curr_b_idx + 1:
            new_ptr.append(t_idx)
            curr_b_idx += 1
    new_ptr.append(len(batch))  # append last index also
    return new_ptr

def generate_subgraph(data, aug_ratio):
    node_num, _ = data.x.size()

    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])
    new_edges = []

    while np.unique(idx_sub).size < sub_num:
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if [idx_sub[-1], sample_node] not in new_edges:
            new_edges.append([idx_sub[-1], sample_node])
        idx_sub.append(sample_node)
        idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]])
    idx_sub = np.sort(np.unique(idx_sub))
    new_edges_idx = []
    for new_edge in new_edges:
        for o_idx, old_edge in enumerate(edge_index.T):
            if np.array_equal(new_edge, old_edge):
                new_edges_idx.append(o_idx)

    node_offsets = []
    last_offset = 0
    for i in range(node_num):
        if i in idx_sub:
            node_offsets.append(last_offset)
        else:
            last_offset += 1
            node_offsets.append(last_offset)

    for idx_1 in range(len(new_edges)):
        for idx_2 in range(len(new_edges[0])):
            new_edges[idx_1][idx_2] = new_edges[idx_1][idx_2] - node_offsets[new_edges[idx_1][idx_2]]

    data.edge_index = torch.tensor(np.array(new_edges).T)

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[new_edges_idx]

    idx_nondrop = idx_sub
    data.x = data.x[idx_nondrop]

    return data

def subgraph_directed(data, aug_ratio):
    if data.batch is None:
        return generate_subgraph(data, aug_ratio)
    else:
        edge_index = data.edge_index.numpy()
        edge_index_batch = [data.batch.numpy()[edge_index[0][i]] for i in range(edge_index[0].size)]
        print(edge_index_batch)
        edge_ptr = new_ptr(edge_index_batch)
        batch_size = np.amax(edge_index_batch)
        for i in range(batch_size):
            temp_x = data.x[edge_ptr[i]:edge_ptr[i+1], :]
            temp_edge_index = data.edge_index[:, edge_ptr[i]:edge_ptr[i+1]]
            if data.edge_attr is not None:
                temp_edge_attr = data.edge_attr[edge_ptr[i]:edge_ptr[i+1],:]
                tempdata = Data(x=temp_x, edge_index=temp_edge_index, edge_attr=temp_edge_attr)
                generate_subgraph(tempdata, aug_ratio)
            else:
                tempdata = Data(x=temp_x, edge_index=temp_edge_index, edge_attr=None)


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
    token = data.edge_attr.mean(dim=0)

    idx_mask = np.random.choice(edge_num, mask_num, replace=False)
    data.edge_attr[idx_mask] = torch.tensor(token, dtype=torch.float32)

    return data

def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()

    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()

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

    data.edge_index = edge_index
    data.x = data.x[idx_nondrop]

    return data

def drop_edges_undirected(data, aug_ratio):
    edge_index = data.edge_index.numpy()
    unique_edges = []
    unique_idcs = []
    for e_idx, edge in enumerate(edge_index.T):
        if (edge[1], edge[0]) not in unique_edges:
            unique_edges.append(tuple(edge))
            unique_idcs.append(e_idx)
    edge_index = np.array(unique_edges).T
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[unique_idcs, :]
    _, edge_num = edge_index.shape
    permute_num = int(edge_num * aug_ratio)

    idx_nondrop = np.sort(np.random.choice(edge_num, (edge_num - permute_num), replace=False))
    edge_index = edge_index[:, idx_nondrop]
    data.edge_index = torch.tensor(edge_index)

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[idx_nondrop, :]

    edge_index_bothdirs = edge_index.T
    for edge in edge_index.T:
        edge_index_bothdirs = np.append(edge_index_bothdirs, [[edge[1], edge[0]]], axis=0)
    edge_index = edge_index_bothdirs.T

    data.edge_index = torch.tensor(edge_index)

    if data.edge_attr is not None:
        edge_attr_bothdirs = data.edge_attr.numpy()
        for edge_a in data.edge_attr.numpy():
            edge_attr_bothdirs = np.append(edge_attr_bothdirs, [edge_a], axis=0)
        data.edge_attr = torch.tensor(edge_attr_bothdirs)

    return data

def drop_edges_directed(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index.numpy()

    idx_nondrop = np.sort(np.random.choice(edge_num, (edge_num - permute_num), replace=False))
    edge_index = edge_index[:, idx_nondrop]
    data.edge_index = torch.tensor(edge_index)

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[idx_nondrop, :]
    # new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, num_nodes=data.x.size()[0])
    # data.num_nodes = data.x.size()
    return data





