import random
from glob import glob
from typing import List
import itertools
from time import time

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.dataset import Dataset, Subset
import torchvision.transforms as transforms
from tqdm import tqdm

from rotation import crop_to_128

from skimage.exposure import equalize_adapthist

IMG_FOLDER = "images_pad_16"

class TrophallaxisDataset(Dataset):
    def __init__(self, item_depth: int, transform=None, image_size=(128,128), 
                 random_crop_amplitude=0, clahe=True):
        self.transformations = transform if transform else transforms.Compose([transforms.ToTensor()])
        clahe_str = 'clahe' if clahe else ''
        self.all_paths = sorted(glob(img_folder + clahe_str + "/*/*.png"))
        self.item_depth = item_depth
        self.y_indices = self._indices_by_label("y")
        self.n_indices = self._indices_by_label("n")
        self.count = len(self.y_indices) + len(self.n_indices)
        self.image_size = image_size
        self.random_crop_amplitude = random_crop_amplitude

        self.grouped_by_event = {}
        self.event_labels = []
        for i, path in enumerate(self.all_paths):
            folder = _folder_index(path)
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
        assert label_str != "u"
        label = 1 if label_str == "y" else 0
        before = [self.all_paths[i-1] for i in range(index, index - self.item_depth//2, -1)]
        after = [self.all_paths[i+1] for i in range(index, index + self.item_depth//2)]
        paths = [*before, path, *after]
        invert = random.random() > 0.5
        if invert:
            paths = [p.replace(IMG_FOLDER, IMG_FOLDER + "_invert") for p in paths]

        if self.random_crop_amplitude > 0:
            x = random.random() * self.random_crop_amplitude
            x -= self.random_crop_amplitude/2
            y = random.random() * self.random_crop_amplitude 
            y -= self.random_crop_amplitude/2
        else:
            x = y = 0

        images = [crop_to_128(imread(path), x, y) for path in paths]

        assert images[0].shape[0] >= 32 and images[0].shape[1] >= 32
        assert self.image_size[0] == self.image_size[1]
        assert self.image_size[0] <= 128

        if self.image_size[0] < 128:
            images = [resize(img, self.image_size, mode="constant") for img in images]
        try:
            data = np.dstack(images) if len(images) > 1 else images[0]
        except ValueError:
            print("ValueError", index, path)
            for img in images:
                print(img.shape, index)
            raise
        data = self.transformations(data)
        return (data, label)

    def __len__(self):
        return self.count

    def _get_subset(self, split_ratio: float, train: bool, seed: int) -> Subset:
        y_events = shuffle([i for i,label in enumerate(self.event_labels) if label])
        n_events = shuffle([i for i,label in enumerate(self.event_labels) if not label])

        y_split = train_test_split(y_events, random_state=seed, train_size=split_ratio)
        n_split = train_test_split(n_events, random_state=seed, train_size=split_ratio)

        y_events = y_split[0 if train else 1]
        n_events = n_split[0 if train else 1]

        #if train:
        #    y_events = y_events[:int(len(y_events)*split_ratio)]
        #    n_events = n_events[:int(len(n_events)*split_ratio)]
        #else:
        #    y_events = y_events[int(len(y_events)*split_ratio):]
        #    n_events = n_events[int(len(n_events)*split_ratio):]

        
        with open("trainlog.txt", "a") as log:
            print("ratio:", len(y_events) / (len(y_events)+len(n_events)), 
                  "count:", (len(y_events)+len(n_events)),
                  "seed:", seed,
                  "train" if train else "test",
                  file=log)

        events = shuffle([*y_events, *n_events])

        indices = [self.grouped_by_event[i] for i in events]

        return Subset(dataset=self, indices=shuffle(list(itertools.chain.from_iterable(indices))))

    def trainset(self, split_ratio=0.8, seed=42) -> Subset:
        return self._get_subset(split_ratio=split_ratio, train=True, seed=seed)

    def testset(self, split_ratio=0.8, seed=42) -> Subset:
        return self._get_subset(split_ratio=split_ratio, train=False, seed=seed)

    def subset_overlap(self, train: Subset, test: Subset) -> set:
        train_folders = set([_folder_index(self.all_paths[i]) for i in train.indices])
        test_folders = set([_folder_index(self.all_paths[i]) for i in test.indices])
        return train_folders & test_folders


def _folder_index(path:str) -> int:
    return int(path.split("/")[1].split("_")[0])


def shuffle(l: list, seed=42) -> list:
    random.seed(seed)
    return random.sample(l, len(l))


def test_run():
    img_size = 128
    item_depth = 3
    ds = TrophallaxisDataset(item_depth=item_depth, image_size=(img_size,img_size),
                             random_crop_amplitude=8)
    trainset = ds.trainset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    testset = ds.testset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    assert len(ds.subset_overlap(train=trainset, test=testset)) == 0
    print("no overlap")

    tic = time()
    print("test")
    lab = []
    for _ in range(3):
        labels = []
        indices = []
        for i, data in enumerate(tqdm(testloader, 0)):
            for label in data[1]:
                labels.append(label)
        lab.append(labels)
    print("done after {} sec".format(time() - tic))
    return ds, lab

