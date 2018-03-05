from random import random
from glob import glob
from typing import List
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset, Subset
from skimage.io import imread


class TrophallaxisDataset(Dataset):
    def __init__(self, item_depth: int, transform=None):
        self.transformations = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.all_paths = sorted(glob("images/*/0/*.png"))
        self.item_depth = item_depth
        self.y_indices = self.indices_by_label("y")
        self.n_indices = self.indices_by_label("n")
        self.count = len(self.y_indices) + len(self.n_indices)

    def indices_by_label(self, label: str) -> List[int]:
        return [i for i, x in enumerate(self.all_paths) if x[-5] == label]

    def __getitem__(self, index):
        path = self.all_paths[index]
        label = 1 if path[-5] == "y" else 0
        before = [self.all_paths[i-1] for i in range(index, index - self.item_depth//2, -1)]
        after = [self.all_paths[i+1] for i in range(index, index + self.item_depth//2)]
        paths = [*before, path, *after]

        # randomly choose rotation
        if random() > 0.5:
            for i, p in enumerate(paths):
                split = p.split("/0/")
                paths[i] = split[0] + "/1/" + split[1]

        images = [imread(path) for path in paths]
        data = np.dstack(images) if len(images) > 1 else images[0]
        data = self.transformations(data)
        return (data, label)

    def __len__(self):
        return self.count

    def trainset(self, split_ratio=0.8):
        n_indices = [x for i, x in enumerate(self.n_indices) if i/len(self.n_indices) < split_ratio]
        y_indices = [x for i, x in enumerate(self.y_indices) if i/len(self.y_indices) < split_ratio]
        return Subset(dataset=self, indices=[*n_indices, *y_indices])

    def testset(self, split_ratio=0.8):
        n_indices = [x for i, x in enumerate(self.n_indices) if i/len(self.n_indices) >= split_ratio]
        y_indices = [x for i, x in enumerate(self.y_indices) if i/len(self.y_indices) >= split_ratio]
        return Subset(dataset=self, indices=[*n_indices, *y_indices])
