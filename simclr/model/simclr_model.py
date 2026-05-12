import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.model(x).flatten(1)

        return x


class ProjHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim1 = 512
        self.dim2 = 128
        self.model = nn.Sequential(
            nn.Linear(1152, self.dim1), nn.BatchNorm1d(self.dim1), nn.ReLU(), nn.Linear(self.dim1, self.dim2)
        )

    def forward(self, x):
        x = self.model(x)

        return x


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.head = ProjHead()

    def forward(self, x):
        backbone_out = self.backbone(x)
        head_out = self.head(backbone_out)

        return head_out