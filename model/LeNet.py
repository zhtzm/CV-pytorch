from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))       # intput(b, 3, 32, 32) output(b, 16, 28, 28)
        X = self.pool1(X)               # output(b, 16, 14, 14)
        X = F.relu(self.conv2(X))       # output(b, 32, 10, 10)
        X = self.pool2(X)               # output(b, 32, 5, 5)
        X = X.view(-1, 32 * 5 * 5)      # output(b, 32*5*5)
        X = F.relu(self.fc1(X))         # output(b, 120)
        X = F.relu(self.fc2(X))         # output(b, 84)
        X = self.fc3(X)                 # output(b, 10)

        return X
