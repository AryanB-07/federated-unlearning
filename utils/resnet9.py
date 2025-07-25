import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.CELU(alpha=0.075)
    )


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet9, self).__init__()

        # Define layers and also expose standard names for compatibility
        self.conv1 = conv_bn(3, 64)                          # [N, 64, 32, 32]
        self.conv2 = conv_bn(64, 128, 5, 2, 2)              # [N, 128, 16, 16]

        self.layer1 = Residual(nn.Sequential(               # Acts like "layer1" in ResNet
            conv_bn(128, 128),
            conv_bn(128, 128)
        ))

        self.conv3 = nn.Sequential(                         # [N, 256, 8, 8]
            conv_bn(128, 256),
            nn.MaxPool2d(2)
        )

        self.layer2 = Residual(nn.Sequential(              # Acts like "layer2" in ResNet
            conv_bn(256, 256),
            conv_bn(256, 256)
        ))

        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))        # [N, 256, 1, 1]
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.conv3(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
