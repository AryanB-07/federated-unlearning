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
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = conv_bn(3, 64)
        self.conv2 = conv_bn(64, 128, kernel_size=5, stride=2, padding=2)

        self.res1 = Residual(nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128)
        ))

        self.conv3 = nn.Sequential(
            conv_bn(128, 256),
            nn.MaxPool2d(2)
        )

        self.res2 = Residual(nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256)
        ))

        self.conv4 = nn.Sequential(
            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()  # Flatten is now inside conv4
        )

        self.flatten = nn.Identity()  # No-op to satisfy external references
        self.fc = nn.Linear(128, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.res2(x)
        x = self.conv4(x)  # Already flattened here
        x = self.flatten(x)  # No-op
        x = self.fc(x)
        return x


def resnet9(num_classes=10):
    model = ResNet9(num_classes)
    model.embed_size = 128
    return model
