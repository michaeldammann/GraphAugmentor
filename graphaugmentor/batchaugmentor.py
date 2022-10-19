import numpy as np
from torch_geometric.data import Data, Batch
import random

class BatchAugmentor:
    def __init__(self, augmentfunclist, augmentfuncratiodict, probs=None):
        self.augmentfunclist = augmentfunclist
        self.augmentfuncratiodict = augmentfuncratiodict
        if probs == None:
            self.probs = [1./float(len(augmentfunclist)) for augfunc in augmentfunclist]
        else:
            self.probs = probs

    def augment_batch(self, databatch):
        all_graphs = Batch.to_data_list(databatch)
        augmentations = random.choices(self.augmentfunclist, self.probs, k=len(all_graphs))
        print(augmentations)
        return Batch.from_data_list([augmentations[i](graph, self.augmentfuncratiodict[augmentations[i]]) for i, graph in enumerate(all_graphs)])



