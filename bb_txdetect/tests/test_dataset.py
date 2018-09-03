import unittest
from bb_txdetect import dataset
from bb_txdetect.dataset import TrophallaxisDataset
import torch


class PathsTestCase(unittest.TestCase):
    def test_folder_index(self):
        test_paths = [
            '/home/mi/mhocke/bb_txdetect_data/images/images_pad_16_invert/00586_n/021_n.png',
            '/home/mhocke/bb_txdetect_data/images/images_pad_16_invert/00586_n/021_n.png',
            '/home/mhocke/images/images_pad_16_invert/00586_n/021_n.png',
            '/home/mhocke/images/00586_n/021_n.png',
            '/home/mhocke/images/586_n/021_n.png',
            '/home/mhocke/images/586_y/001_n.png',
            '/home/mhocke/images/586_y/001_y.png',
            'images/586_y/001_y.png',
            '586_y/001_y.png',
        ]
        for i, path in enumerate(test_paths):
            with self.subTest(i=i):
                self.assertEqual(dataset._folder_index(path=path), 586)

    def test_load_data(self):
        item_depth = 3
        clahe = False
        rca = 8

        dataset = TrophallaxisDataset(item_depth=item_depth,
                                      random_crop_amplitude=rca, clahe=clahe,
                                      random_rotation_max=0,
                                      drop_frames_around_trophallaxis=0)
        trainset = dataset.trainset()
        testset = dataset.testset()
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                 shuffle=False, num_workers=2)

        self.assertFalse(dataset.subset_overlap(train=trainset, test=testset))

        for data in testloader:
            pass
