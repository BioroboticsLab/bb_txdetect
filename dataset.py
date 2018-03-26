import random
from glob import glob
from typing import List
import numpy as np
import itertools

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset, Subset
from skimage.io import imread
from skimage.transform import resize


class TrophallaxisDataset(Dataset):
    def __init__(self, item_depth: int, transform=None, image_size=(128,128), hide_tags=True):
        self.tags = "_hidden_tags" if hide_tags else ""
        self.transformations = transform if transform else transforms.Compose([transforms.ToTensor()])
        self.all_paths = sorted(glob("images/*/0{}/*.png".format(self.tags)))
        self.item_depth = item_depth
        self.y_indices = self._indices_by_label("y")
        self.n_indices = self._indices_by_label("n")
        self.count = len(self.y_indices) + len(self.n_indices)
        self.image_size = image_size

        self.grouped_by_event = {}
        self.event_labels = []
        for i, path in enumerate(self.all_paths):
            folder = self._folder_index(path)
            if len(self.event_labels) <= folder:
                self.event_labels.append(False)
            label_str = path[-5]
            if label_str == "y":
                self.event_labels[-1] = True
            if folder not in self.grouped_by_event:
                self.grouped_by_event[folder] = []
            if label_str != "u":
                self.grouped_by_event[folder].append(i)

    def _indices_by_label(self, label: str) -> List[int]:
        return [i for i, x in enumerate(self.all_paths) if x[-5] == label]

    def __getitem__(self, index):
        path = self.all_paths[index]
        label_str = path[-5]
        if label_str == "u":
            print(index, len(self.all_paths), self.item_depth, self.all_paths[index])
        assert label_str != "u"
        label = 1 if label_str == "y" else 0
        before = [self.all_paths[i-1] for i in range(index, index - self.item_depth//2, -1)]
        after = [self.all_paths[i+1] for i in range(index, index + self.item_depth//2)]
        paths = [*before, path, *after]

        # randomly choose rotation
        if random.random() > 0.5:
            for i, p in enumerate(paths):
                split = p.split("/0{}/".format(self.tags))
                paths[i] = split[0] + "/1{}/".format(self.tags) + split[1]

        images = [imread(path) for path in paths]
        assert self.image_size[0] == self.image_size[1]
        assert self.image_size[0] <= 128
        if self.image_size[0] < 128:
            images = [resize(img, self.image_size, mode="constant") for img in images]
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
        y_events = shuffle([i for i,label in enumerate(self.event_labels) if label])
        n_events = shuffle([i for i,label in enumerate(self.event_labels) if not label])

        if train:
            y_events = y_events[:int(len(y_events)*split_ratio)]
            n_events = n_events[:int(len(n_events)*split_ratio)]
        else:
            y_events = y_events[int(len(y_events)*split_ratio):]
            n_events = n_events[int(len(n_events)*split_ratio):]

        print("ratio", len(y_events) / (len(y_events)+len(n_events)), "train" if train else "test")

        events = shuffle([*y_events, *n_events])

        indices = [self.grouped_by_event[i] for i in events]

        return Subset(dataset=self, indices=list(itertools.chain.from_iterable(indices)))

    def trainset(self, split_ratio=0.8) -> Subset:
        return self._get_subset(split_ratio=split_ratio, train=True)

    def testset(self, split_ratio=0.8) -> Subset:
        return self._get_subset(split_ratio=split_ratio, train=False)

    def subset_overlap(self, train: Subset, test: Subset) -> set:
        train_folders = set([self._folder_index(self.all_paths[i]) for i in train.indices])
        test_folders = set([self._folder_index(self.all_paths[i]) for i in test.indices])
        return train_folders & test_folders


def shuffle(l: list, seed=42) -> list:
    random.seed(seed)
    return random.sample(l, len(l))
