import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from singlegraphaugmentations import drop_edges_directed, drop_nodes, mask_nodes, mask_edges, drop_edges_undirected, subgraph_directed, subgraph_undirected, identity
from batchaugmentor import BatchAugmentor

'''
augfuncratiodict = {drop_edges_directed:0.2,
                    drop_nodes:0.2,
                    mask_nodes:0.2,
                    mask_edges:0.2,
                    subgraph_directed:0.8,
                    identity:0.0}
baug = BatchAugmentor([drop_edges_directed, drop_nodes, mask_nodes, mask_edges, subgraph_directed, identity], augfuncratiodict)
'''

augfuncratiodict = {subgraph_directed:0.8}
baug = BatchAugmentor([subgraph_directed], augfuncratiodict)

'''
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 3, 0, 2],
                           [1, 0, 2, 1, 3, 2, 3, 0, 2, 0]], dtype=torch.long)
edge_attr = torch.tensor([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11]])
'''
x = torch.tensor([[0,0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]], dtype=torch.float)

edge_index = torch.tensor([[0, 1, 2, 0, 0, 0, 2, 5, 5, 5, 4, 1, 6, 3, 7, 7, 7, 8, 8, 8, 9, 9, 1, 3, 6],
                           [1, 2, 3, 3, 2, 4, 4, 3, 1, 4, 6, 6, 2, 5, 4, 2, 1, 6, 4, 2, 0, 1, 7, 8, 9]], dtype=torch.long)
edge_attr = torch.tensor([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6], [5,6,7],
                          [6,1,2],[7,2,3],[8,3,4],[9,4,5],[10,5,6], [11,6,7],
                          [12,1,2],[13,2,3],[14,3,4],[15,4,5],[16,5,6], [17,6,7],
                          [18,1,2],[19,2,3],[20,3,4],[21,4,5],[22,5,6], [23,6,7],
                          [24,1,2]])

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
data_1 = data.clone()

loader = DataLoader([data,data_1], batch_size=2, shuffle=False)


for batch in loader:
    '''
    print(batch)
    print(batch.x)
    print(batch.edge_index)
    print(batch.edge_attr)
    print(batch.batch)
    print(batch.ptr)
    print("-------------------------------------")
    '''
    list_graphs_orig = Batch.to_data_list(batch)
    print(list_graphs_orig)
    print(list_graphs_orig[1].x)
    print(list_graphs_orig[1].edge_index)
    print(list_graphs_orig[1].edge_attr)
    augs=baug.augment_batch(batch)
    list_graphs = Batch.to_data_list(augs)
    print(list_graphs)
    print(list_graphs[1].x)
    print(list_graphs[1].edge_index)
    print(list_graphs[1].edge_attr)
    '''
    #new_batch = subgraph_directed(batch, aug_ratio=0.25)
    print(batch)
    print(new_batch.x)
    print(new_batch.edge_index)
    print(new_batch.edge_attr)
    print(new_batch.batch)
    print(new_batch.ptr)
    '''

'''
print(data.x)
print(data.edge_index)
print(data.edge_attr)

new_data = subgraph_undirected(data, aug_ratio=0.8)
print('--------------------------------')

print(new_data.x)
print(new_data.edge_index)
print(new_data.edge_attr)

'''
