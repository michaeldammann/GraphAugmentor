<p align="center">
  <img src="graphaugmentor_logo.SVG" width="50%"/>
</p>

# GraphAugmentor
A simple Graph Augmentation package for self-supervised and contrastive graph machine learning based on PyTorch Geometric

## Introduction
GraphAugmentor provides simple to use graph augmentations based on PyTorch Geometric graph representations. Augmentations can be applied to single graphs or batches of graphs, making this package suitable for graph-level tasks. GraphAugmentor also provides augmentations tailored for both undirected and directed graphs.

## Augmentations
This package is suitable for directed and undirected graphs and graphs with and without edge attributes (edge_attr in PyTorch Geometric). See the provided examples (next section) for more details on how to use this package depending on the kind of graphs you are dealing with.

The following augmentations are provided (see [singlegraphaugmentations.py](graphaugmentor/singlegraphaugmentations.py)):

- Subgraph (`subgraph_directed`, `subgraph_undirected`): Starting at random node, walk through `aug_ratio`*100 % of the nodes to create a subgraph. Available for both directed and undirected graphs.
- Mask nodes (`mask_nodes`): Mask `aug_ratio`*100 % of the feature vectors of all available nodes. Same for directed and undirected graphs.
- Mask edges (`mask_edges_directed`, `mask_edges_undirected`): Mask `aug_ratio`*100 % of the edges in a graph. Available for both directed and undirected graphs.
- Drop nodes (`drop_nodes`): Drop `aug_ratio`*100 % of the nodes in the graphs. Same for directed and undirected graphs.
- Drop edges (`drop_edges_directed`, `drop_edges_undirected`): Drop `aug_ratio`*100 % of the edges in the graphs.  Available for both directed and undirected graphs.
- Identity (`identity`): Return the original graph. (`aug_ratio` has no effect)

## Examples
See [examples.py](graphaugmentor/examples.py) for an example for a batch of directed and a batch of undirected graphs.
