import unittest
import inspect
import torch
from txdetect.dataset import TrophallaxisDataset
from txdetect import smaller_net


class SmallerNetTestCase(unittest.TestCase):
    def test_smaller_nets_output_size(self):
        """test if all models in smaller_net have correct output size."""
        depth = 3
        ds = TrophallaxisDataset(item_depth=depth,
                                 random_crop_amplitude=0,
                                 clahe=False,
                                 random_rotation_max=0,
                                 drop_frames_around_trophallaxis=True)
        testset = ds.testset()
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                 shuffle=False, num_workers=2)
        criterion = torch.nn.CrossEntropyLoss()
        for data in testloader:
            break
        self.assertTrue(data)
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])

        net_classes = [tup[1] for tup in inspect.getmembers(smaller_net,
                                                            inspect.isclass)]

        expected_output_size = torch.Size([testloader.batch_size, 2])
        known_failure_classes = [smaller_net.SmallerNet4, smaller_net.SmallerNet5]

        for i, net_class in enumerate(net_classes):
            with self.subTest(i=i):
                net = net_class(in_channels=depth)
                optimizer = torch.optim.Adam(net.parameters())
                optimizer.zero_grad()
                outputs = net(inputs)
                if net_class not in known_failure_classes:
                    self.assertEqual(expected_output_size, outputs.shape)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

