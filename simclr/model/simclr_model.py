import torch.nn as nn
import torchvision.models as models

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SimCLREncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            Conv(3, 32, 3, 2, 1),
            Conv(32, 64, 3, 2, 1),
            Conv(64, 128, 3, 2, 1),
            Conv(128, 256, 3, 2, 1),
            nn.MaxPool2d(2),
            nn.Flatten(1, -1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class SimCLRProjHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SimCLREncoder()
        self.head = SimCLRProjHead()

    def forward(self, x, encoder_out=False):
        h = self.encoder(x)

        if encoder_out:
            return h
        
        z = self.head(h)
        
        return z

if __name__ == "__main__":
    import torch
    x = torch.randn([1, 3, 32, 32])
    model = SimCLRModel().eval()
    out = model(x, encoder_out=True)

    print(f"Model output shape: {out.shape}")