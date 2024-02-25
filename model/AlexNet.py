import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),     # input(b, 3, 224, 224) output(b, 48, 55, 55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                                              # output(b, 48, 27, 27)
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2),    # output(b, 128, 27, 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                                              # output(b, 128, 13, 13)
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1),   # output(b, 192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),   # ouput(b, 192, 13, 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),   # ouput(b, 128, 13, 13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                                               # output(b, 128, 6, 6)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128 * 6 * 6, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, X):
        X = self.features(X)                    # output(b, 256, 6, 6)
        X = torch.flatten(X, start_dim=1)       # output(b, 256 * 6 * 6)
        X = self.classifier(X)                  # output(b, num_classes)
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
