import torch.nn as nn
import torchvision.models as models


class ResNet18BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)

        return x


class ProjHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim1 = int(self.cfg.MODEL.HEAD.dim1)
        self.dim2 = int(self.cfg.MODEL.HEAD.dim2)
        self.model = nn.Sequential(nn.Linear(self.dim1, self.dim1), nn.ReLU(), nn.Linear(self.dim1, self.dim2))

    def forward(self, x):
        x = self.model(x)

        return x


class SimCLR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = ResNet18BackBone()
        self.head = ProjHead(self.cfg)

    def forward(self, x):
        backbone_out = self.backbone(x)
        head_out = self.head(backbone_out)

        return head_out
