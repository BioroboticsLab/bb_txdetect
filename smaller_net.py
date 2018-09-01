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


class SmallerNet2(nn.Module):
    def __init__(self, in_channels=3):
        super(SmallerNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=48, kernel_size=5)
        self.fc1 = nn.Linear(in_features=self.conv4.out_channels * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=84)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SmallerNet3(nn.Module):
    def __init__(self, in_channels=3):
        super(SmallerNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=48, kernel_size=5)
        self.fc1 = nn.Linear(in_features=self.conv4.out_channels * 4 * 4, out_features=60)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SmallerNet4(nn.Module):
    def __init__(self, in_channels=3):
        super(SmallerNet4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=48, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=self.conv5.out_channels, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.fc1.in_features)
        return x


class SmallerNet4_1(nn.Module):
    def __init__(self, in_channels):
        super(SmallerNet4_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=48, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=self.conv5.out_channels, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)
        return x


class SmallerNet5(nn.Module):
    def __init__(self, in_channels=3):
        super(SmallerNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=48, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=self.conv5.out_channels, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.bn1(x)
        x = F.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.bn2(x)
        x = F.dropout(self.pool(F.relu(self.conv4(x))))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.fc1.in_features)
        return x


class SmallerNet6(nn.Module):
    def __init__(self, in_channels: int, model_parameters: [int] = [10, 16, 32, 48, 64]):
        super(SmallerNet6, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=model_parameters[0], kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=model_parameters[1], kernel_size=5)
        self.bn1 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=model_parameters[2], kernel_size=5)
        self.bn2 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=model_parameters[3], kernel_size=5)
        self.bn3 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=model_parameters[4], kernel_size=3)
        self.fc1 = nn.Linear(in_features=self.conv5.out_channels, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.fc1(x)
        return x


class SmallerNet7(nn.Module):
    def __init__(self, in_channels=3):
        super(SmallerNet7, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10,
                               kernel_size=5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(self.conv1.out_channels, out_channels=16,
                               kernel_size=5)

        self.bn1 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels,
                               out_channels=32, kernel_size=5)

        self.bn2 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels,
                               out_channels=48, kernel_size=5)

        self.bn3 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels,
                               out_channels=2, kernel_size=3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.bn2(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.bn3(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 2)
        return x
