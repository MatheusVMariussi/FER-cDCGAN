
import torch
import torch.nn as nn

# -------- Generator (Conditional) --------
class Generator(nn.Module):
    def __init__(self, nz=100, n_classes=7, ngf=64, img_size=48):
        super().__init__()
        self.nz = nz
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, n_classes)  # one-hot like

        # We concatenate z with class one-hot -> nz + n_classes
        in_dim = nz + n_classes

        # Map to (ngf*4, 6, 6) so that two doublings reach 24 and then 48
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf*4*6*6, bias=False),
            nn.BatchNorm1d(ngf*4*6*6),
            nn.ReLU(True)
        )

        self.net = nn.Sequential(
            # (ngf*4, 6, 6) -> (ngf*2, 12, 12)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # -> (ngf, 24, 24)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # -> (ngf//2, 48, 48)
            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),
            # -> (1, 48, 48)
            nn.Conv2d(ngf//2, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: (B, nz), labels: (B,)
        y = self.embed(labels)                       # (B, n_classes)
        x = torch.cat([z, y], dim=1)                 # (B, nz+n_classes)
        x = self.fc(x)                               # (B, ngf*4*6*6)
        x = x.view(x.size(0), -1, 6, 6)
        x = self.net(x)                              # (B, 1, 48, 48)
        return x

# -------- Discriminator (Conditional) --------
class Discriminator(nn.Module):
    def __init__(self, n_classes=7, ndf=64):
        super().__init__()
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, n_classes)

        # Spatially concatenate a class map [n_classes x H x W] to the image (1 x H x W)
        # So input channels = 1 + n_classes
        in_channels = 1 + n_classes

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),   # 48 -> 24
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),         # 24 -> 12
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),       # 12 -> 6
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 6, 1, 0, bias=False),           # 6 -> 1
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: (B,1,48,48); labels: (B,)
        y = self.embed(labels)               # (B, n_classes)
        y = y.unsqueeze(-1).unsqueeze(-1)    # (B, n_classes, 1, 1)
        y = y.expand(-1, -1, x.size(2), x.size(3))  # (B, n_classes, H, W)
        x = torch.cat([x, y], dim=1)         # (B, 1+n_classes, H, W)
        out = self.net(x)                    # (B,1,1,1)
        return out.view(-1, 1)
