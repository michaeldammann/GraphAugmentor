from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import torch_geometric


dataset = PygGraphPropPredDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='ogbg-molhiv')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

it = iter(loader)
print("starting iteration")
for data_b in it:
    print(data_b)
    #data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
    break
