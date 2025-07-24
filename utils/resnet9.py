import torch.nn as nn
import torch.nn.functional as F

class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        self.base = nn.Sequential(*list(m.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
