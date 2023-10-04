<p align="center">
  <img src="graphaugmentor_logo.SVG" width="50%"/>
</p>

# GraphAugmentor
A simple graph augmentation package for self-supervised and contrastive graph machine learning based on PyTorch Geometric.

## Introduction
GraphAugmentor provides simple to use graph augmentations based on PyTorch Geometric graph representations. Augmentations can be applied to single graphs or batches of graphs, making this package suitable for graph-level tasks. GraphAugmentor also provides augmentations tailored for both undirected and directed graphs.

## Installation
Tested on Ubuntu 20.04 with PyTorch 1.13.0, create a virtual environment and install (Cuda is not used, cu116 is replaceable with cpu):

    pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

## Augmentations
This package is suitable for directed and undirected graphs and graphs with and without edge attributes (edge_attr in PyTorch Geometric). See the provided examples (next section) for more details on how to use this package depending on the kind of graphs you are dealing with.

The following augmentations are provided (see [singlegraphaugmentations.py](graphaugmentor/singlegraphaugmentations.py)):

- Subgraph (`subgraph_directed`, `subgraph_undirected`): Starting at random node, walk through `aug_ratio`*100 % of the nodes to create a subgraph. Available for both directed and undirected graphs.
- Mask nodes (`mask_nodes`): Mask `aug_ratio`*100 % of the feature vectors of all available nodes. Same for directed and undirected graphs.
- Mask edges (`mask_edges_directed`, `mask_edges_undirected`): Mask `aug_ratio`*100 % of the edges in a graph. Available for both directed and undirected graphs.
- Drop nodes (`drop_nodes`): Drop `aug_ratio`*100 % of the nodes in the graphs. Same for directed and undirected graphs.
- Drop edges (`drop_edges_directed`, `drop_edges_undirected`): Drop `aug_ratio`*100 % of the edges in the graphs.  Available for both directed and undirected graphs.
- Identity (`identity`): Return the original graph. (`aug_ratio` has no effect)

## Usage and Examples
You can either use the functions directly from [singlegraphaugmentations.py](graphaugmentor/singlegraphaugmentations.py) for single graphs or use the [BatchAugmentor](graphaugmentor/batchaugmentor.py) to specify what augmentations to use (`augmentfunclist`), together with the `aug_ratio` settings summarized in `augmentfuncratiodict` and a custom sample distribution (`probs`). `BatchAugmentor` then provides `augment_batch` to augment a single batch, e.g., provided by the PyTorch Geometric DataLoader. The number of (random) augmentations of each graph can be customized using `num_augs`. See [examples.py](graphaugmentor/examples.py) for a minimal example for a batch of directed and a batch of undirected graphs.

## Citation

Please consider citing this repository if it was useful for your research.
```
@misc{graphaugmentor,
	author={Dammann, Michael},
	title={graphaugmentor},
	year={2022},
	url={https://github.com/michaeldammann/GraphAugmentor},
}
```
