from random import random
from glob import glob
from typing import List
import numpy as np
import itertools

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset, Subset
from skimage.io import imread


class TrophallaxisDataset(Dataset):
    def __init__(self, item_depth: int, transform=None):
        self.transformations = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.all_paths = sorted(glob("images/*/0/*.png"))
        self.item_depth = item_depth
        self.y_indices = self._indices_by_label("y")
        self.n_indices = self._indices_by_label("n")
        self.count = len(self.y_indices) + len(self.n_indices)

        self.grouped_by_event = {}
        self.event_labels = []
        for i, path in enumerate(self.all_paths):
            folder = self._folder_index(path)
            if len(self.event_labels) <= folder:
                self.event_labels.append(False)
            if path[-5] == "y":
                self.event_labels[-1] = True
            if folder not in self.grouped_by_event:
                self.grouped_by_event[folder] = []
            self.grouped_by_event[folder].append(i)

    def _indices_by_label(self, label: str) -> List[int]:
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

    def split_index(self, split_ratio: float, indices) -> int:
        i = int(len(indices) * split_ratio)
        return i - int(self.all_paths[i].split("/")[-1].split("_")[0])

    def _folder_index(self, path:str) -> int:
        return int(path.split("/")[1].split("_")[0])

    def _get_subset(self, split_ratio, train: bool) -> Subset:
        y_events = [i for i,label in enumerate(self.event_labels) if label]
        n_events = [i for i,label in enumerate(self.event_labels) if not label]

        if train:
            y_events = y_events[:int(len(y_events)*split_ratio)]
            n_events = n_events[:int(len(n_events)*split_ratio)]
        else:
            y_events = y_events[int(len(y_events)*split_ratio):]
            n_events = n_events[int(len(n_events)*split_ratio):]

        print("ratio", len(y_events) / (len(y_events)+len(n_events)), "train" if train else "test")

        indices = [self.grouped_by_event[i] for i in y_events]
        indices += [self.grouped_by_event[i] for i in n_events]

        return Subset(dataset=self, indices=list(itertools.chain.from_iterable(indices)))

    def trainset(self, split_ratio=0.8) -> Subset:
        return self._get_subset(split_ratio=split_ratio, train=True)

    def testset(self, split_ratio=0.8) -> Subset:
        return self._get_subset(split_ratio=split_ratio, train=False)

    def subset_overlap(self, train: Subset, test: Subset) -> set:
        train_folders = set([self._folder_index(self.all_paths[i]) for i in train.indices])
        test_folders = set([self._folder_index(self.all_paths[i]) for i in test.indices])
        return train_folders & test_folders

