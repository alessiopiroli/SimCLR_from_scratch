import torch
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
    def __init__(self):
        super().__init__()
        pass
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, x):
        x = self.model(x)

        return x
    

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = ResNet18BackBone()
        self.head = ProjHead()

    def forward(self, x):
        backbone_out = self.backbone(x)
        head_out = self.head(backbone_out)

        return head_out