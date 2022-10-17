import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric
import networkx as nx

from GraphAugmentor import GraphTransform_All
from graphlevelaugmentations import drop_edges, drop_nodes, mask_nodes, mask_edges

gaugm = GraphTransform_All(0.2, 0.8, 0.2, 0.2, 1)

edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 0, 3, 0, 2],
                           [1, 0, 2, 1, 3, 2, 3, 0, 2, 0]], dtype=torch.long)
x = torch.tensor([[-1, -1], [0,0], [1, 1], [2, 2]], dtype=torch.float)
edge_attr = torch.tensor([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11]])

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
data_1 = data.clone()

loader = DataLoader([data,data_1], batch_size=2, shuffle=False)
'''
for batch in loader:
    print(batch)
    print(batch.x)
    print(batch.edge_index)
    print(batch.edge_attr)
    print(batch.batch)
    print(batch.ptr)
    print("-------------------------------------")
    new_batch = mask_edges(batch, aug_ratio=0.25)
    print(batch)
    print(new_batch.x)
    print(new_batch.edge_index)
    print(new_batch.edge_attr)
    print(new_batch.batch)
    print(new_batch.ptr)

'''
print(data.x)
print(data.edge_index)
print(data.edge_attr)

new_data = mask_edges(data, aug_ratio=0.25)
print('--------------------------------')

print(new_data.x)
print(new_data.edge_index)
print(new_data.edge_attr)


#todo: drop_edges_undirected