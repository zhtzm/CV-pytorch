import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(GoogLeNet, self).__init__()
        block3 = nn.Sequential(
            Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32),
            Inception(in_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        # (b, 480, 14, 14)
        block4 = nn.Sequential(
            Inception(in_channels=480, c1=192, c2=(96, 208), c3=(16, 48), c4=64),
            Inception(in_channels=512, c1=160, c2=(112, 224), c3=(24, 64), c4=64),
            Inception(in_channels=512, c1=128, c2=(128, 256), c3=(24, 64), c4=64),
            Inception(in_channels=512, c1=112, c2=(144, 288), c3=(32, 64), c4=64),
            Inception(in_channels=528, c1=256, c2=(160, 320), c3=(32, 128), c4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        # (b, 832, 7, 7)
        block5 = nn.Sequential(
            Inception(in_channels=832, c1=256, c2=(160, 320), c3=(32, 128), c4=128),
            Inception(in_channels=832, c1=384, c2=(192, 384), c3=(48, 128), c4=128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # (b, 1024, 1, 1)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            block3,
            block4,
            block5,
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, X):
        X = self.net(X)
        return X

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Inception(nn.Module):
    def __init__(self, in_channels, c1: int, c2: tuple, c3: tuple, c4: int):
        super(Inception, self).__init__()
        path1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
        self.p1 = nn.Sequential(
            path1_1,
            nn.ReLU(inplace=True)
        )

        path2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        path2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        self.p2 = nn.Sequential(
            path2_1,
            nn.ReLU(inplace=True),
            path2_2,
            nn.ReLU(inplace=True)
        )

        path3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        path3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        self.p3 = nn.Sequential(
            path3_1,
            nn.ReLU(inplace=True),
            path3_2,
            nn.ReLU(inplace=True)
        )

        path4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        path4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)
        self.p4 = nn.Sequential(
            path4_1,
            path4_2,
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        p1 = self.p1(X)
        p2 = self.p2(X)
        p3 = self.p3(X)
        p4 = self.p4(X)

        return torch.cat((p1, p2, p3, p4), dim=1)
    
if __name__ == '__main__':
    X = torch.rand(size=(1, 3, 224, 225))
    net = GoogLeNet(num_classes=5)
    for layer in net.net:
        X = layer(X)
        print('output shape\t', X.shape)
        