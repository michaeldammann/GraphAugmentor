import torch
from torch_geometric.data import Data, Batch

from singlegraphaugmentations import drop_edges_directed, drop_edges_undirected, drop_nodes, mask_nodes, mask_edges_undirected, mask_edges_directed, subgraph_directed, subgraph_undirected, identity
from batchaugmentor import BatchAugmentor

'''
Determining whether the graphs are directed or undirected is task of the developer and appropiate augmentations
(i.e., the "_directed" or "_undirected" versions) have to be chosen manually, see the following examples
Both graphs with and without edge attributes (edge_attr) are accepted, choose "mask_edges_(un)directed" accordingly. 
'''

############### Example for batch of DIRECTED graphs ################

# Choose set of augmentations and their respective aug_ratios (share of nodes/edges to be augmented/dropped/considered etc.)
augfuncratiodict_directed = {drop_edges_directed:0.2,
                    drop_nodes:0.2,
                    mask_nodes:0.2,
                    mask_edges_directed:0.2,
                    subgraph_directed:0.8,
                    identity:0.0} #identity function is independent of aug_ratio and aug_ratio is only listed for consistency reasons
baug_directed = BatchAugmentor([drop_edges_directed, drop_nodes, mask_nodes, mask_edges_directed, subgraph_directed, identity],
                               augmentfuncratiodict=augfuncratiodict_directed,
                               probs=[1./6, 1./6, 1./6, 1./6, 1./6, 1./6]) #probs can also be left blank for automatic uniform distribution

#Construct simple directed graph
x_directed = torch.tensor([[0,0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]], dtype=torch.float)

edge_index_directed = torch.tensor([[0, 1, 2, 0, 0, 0, 2, 5, 5, 5, 4, 1, 6, 3, 7, 7, 7, 8, 8, 8, 9, 9, 1, 3, 6],
                           [1, 2, 3, 3, 2, 4, 4, 3, 1, 4, 6, 6, 2, 5, 4, 2, 1, 6, 4, 2, 0, 1, 7, 8, 9]], dtype=torch.long)
# !!! Edge attributes are optional !!!

edge_attr_directed = torch.tensor([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6], [5,6,7],
                          [6,1,2],[7,2,3],[8,3,4],[9,4,5],[10,5,6], [11,6,7],
                          [12,1,2],[13,2,3],[14,3,4],[15,4,5],[16,5,6], [17,6,7],
                          [18,1,2],[19,2,3],[20,3,4],[21,4,5],[22,5,6], [23,6,7],
                          [24,1,2]])

data_directed_0 = Data(x=x_directed, edge_index=edge_index_directed, edge_attr=edge_attr_directed)
data_directed_1 = data_directed_0.clone()

batch_directed = Batch.from_data_list([data_directed_0, data_directed_1])

augmented_batch_directed = baug_directed.augment_batch(batch_directed, num_augs=1) #apply num_augs augmentations to each graph in the batch


############### Example for batch of UNDIRECTED graphs ################

augfuncratiodict_undirected = {drop_edges_undirected:0.2,
                    drop_nodes:0.2,
                    mask_nodes:0.2,
                    mask_edges_undirected:0.2,
                    subgraph_undirected:0.8,
                    identity:0.0} #identity function is independent of aug_ratio and aug_ratio is only listed for consistency reasons

baug_undirected = BatchAugmentor([drop_edges_directed, drop_nodes, mask_nodes, mask_edges_directed, subgraph_directed, identity],
                               augmentfuncratiodict=augfuncratiodict_directed,
                               probs=[1./6, 1./6, 1./6, 1./6, 1./6, 1./6]) #probs can also be left blank for automatic uniform distribution

x_undirected = torch.tensor([[0,0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=torch.float)
edge_index_undirected = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                    [1, 2, 0, 1, 3, 4, 2, 5, 3, 4]], dtype=torch.long)
edge_attr_undirected = torch.tensor([[0,1,2], [1,2,3], [0,1,2], [1,2,3], [2,3,4], [3,4,5],
                                   [2,3,4], [4,5,6], [3,4,5], [4,5,6]])

data_undirected_0 = Data(x=x_directed, edge_index=edge_index_directed, edge_attr=edge_attr_directed)
data_undirected_1 = data_undirected_0.clone()

batch_undirected = Batch.from_data_list([data_undirected_0, data_undirected_1])

augmented_batch_undirected = baug_undirected.augment_batch(batch_undirected, num_augs=1) #apply num_augs augmentations to each graph in the batch
