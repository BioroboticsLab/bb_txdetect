import torch.nn as nn
import torch.nn.functional as F

class SmallerNet1(nn.Module):
    def __init__(self, in_channels=3):
        super(SmallerNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=5)
        self.fc1 = nn.Linear(in_features=self.conv3.out_channels * 12 * 12, out_features=120)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=84)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
