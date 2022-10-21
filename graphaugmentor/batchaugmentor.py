from torch_geometric.data import Batch
import random

class BatchAugmentor:
    def __init__(self, augmentfunclist, augmentfuncratiodict, probs=None):
        self.augmentfunclist = augmentfunclist
        self.augmentfuncratiodict = augmentfuncratiodict
        if probs == None:
            self.probs = [1./float(len(augmentfunclist)) for augfunc in augmentfunclist]
        else:
            self.probs = probs

    def augment_batch(self, databatch: Batch, num_augs: int=1) -> Batch:
        all_graphs = Batch.to_data_list(databatch)
        for i in range(num_augs):
            augmentations = random.choices(self.augmentfunclist, self.probs, k=len(all_graphs))
            all_graphs = [augmentations[i](graph, self.augmentfuncratiodict[augmentations[i]]) for i, graph in enumerate(all_graphs)]
        return Batch.from_data_list(all_graphs)


