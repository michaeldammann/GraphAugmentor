# Adapted and modified from https://github.com/Shen-Lab/GraphCL/blob/master/semisupervised_TU/pre-training/tu_dataset.py
import torch
import numpy as np


def subgraph_directed(data, aug_ratio):
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


def subgraph_undirected(data, aug_ratio):
    data = subgraph_directed(data, aug_ratio)
    edge_index_T = data.edge_index.numpy().T.tolist()
    has_edge_attr = False
    if data.edge_attr is not None:
        edge_attr = data.edge_attr.numpy().tolist()
        has_edge_attr = True

    for e_idx, edge in enumerate(edge_index_T):
        if [edge[1], edge[0]] not in edge_index_T:
            edge_index_T.append([edge[1], edge[0]])
            if has_edge_attr:
                edge_attr.append(edge_attr[e_idx])
    data.edge_index = torch.tensor(np.array(edge_index_T).T)
    if has_edge_attr:
        data.edge_attr = torch.tensor(np.array(edge_attr))

    return data


def identity(data, aug_ratio):
    return data


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    data.x = data.x.type('torch.FloatTensor')
    token = data.x.mean(dim=0)
    token = token.type('torch.FloatTensor')

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token
    return data


def mask_edges_directed(data, aug_ratio):
    _, edge_num = data.edge_index.size()
    mask_num = int(edge_num * aug_ratio)

    data.edge_attr = data.edge_attr.type('torch.FloatTensor')
    token = data.edge_attr.mean(dim=0)
    token = token.type('torch.FloatTensor')

    idx_mask = np.random.choice(edge_num, mask_num, replace=False)
    data.edge_attr[idx_mask] = token

    return data

def mask_edges_undirected(data, aug_ratio):
    edge_index = data.edge_index
    unique_edges = []
    unique_idcs = []
    edge_index_T = edge_index.numpy().T
    for e_idx, edge in enumerate(edge_index_T):
        if (edge[1], edge[0]) not in unique_edges:
            unique_edges.append(tuple(edge))
            unique_idcs.append(e_idx)

    _, edge_num = edge_index.size()
    mask_num = int(0.5 * edge_num * aug_ratio)

    edge_pairs_dict = {}
    for idx0, edge0 in enumerate(edge_index_T):
        for idx1, edge1 in enumerate(edge_index_T):
            if np.array_equal(edge1, [edge0[1], edge0[0]]):
                edge_pairs_dict[idx0] = idx1

    data.edge_attr = data.edge_attr.type('torch.FloatTensor')
    token = data.edge_attr.mean(dim=0)
    token = token.type('torch.FloatTensor')

    idx_mask = np.random.choice(unique_idcs, mask_num, replace=False)
    for elem in idx_mask:
        idx_mask = np.append(idx_mask,[edge_pairs_dict[elem]], axis=0)
    data.edge_attr[idx_mask] = token

    return data


def drop_nodes(data, aug_ratio):
    node_num, _ = data.x.size()

    has_edge_attr = False
    if data.edge_attr is not None:
        edge_attr = data.edge_attr.numpy().tolist()
        has_edge_attr = True

    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()

    node_offsets = []
    last_offset = 0
    for i in range(node_num):
        if i in idx_nondrop:
            node_offsets.append(last_offset)
        else:
            last_offset += 1
            node_offsets.append(last_offset)

    edge_index_T = data.edge_index.numpy().T
    new_edges_T = []
    new_edge_attr = []
    for e_idx, edge in enumerate(edge_index_T):
        if edge[0] in idx_nondrop and edge[1] in idx_nondrop:
            new_edge = [edge[0]-node_offsets[edge[0]], edge[1]-node_offsets[edge[1]]]
            new_edges_T.append(new_edge)
            if has_edge_attr:
                new_edge_attr.append(edge_attr[e_idx])

    data.edge_index = torch.tensor(np.array(new_edges_T).T)
    data.x = data.x[idx_nondrop]
    if has_edge_attr:
        data.edge_attr = torch.tensor(np.array(new_edge_attr))

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
