import torch
import torch.nn as nn

# -------- Generator (Conditional) --------
class Generator(nn.Module):
    def __init__(self, nz=100, n_classes=7, ngf=64, img_size=48):
        super().__init__()
        self.nz = nz
        self.n_classes = n_classes
        
        # Usar n_classes como dimensão de embedding é um pouco estranho,
        # mas vamos manter para consistência.
        self.embed = nn.Embedding(n_classes, n_classes)

        # We concatenate z with class embedding -> nz + n_classes
        in_dim = nz + n_classes

        # Map to (ngf*4, 6, 6)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf*4*6*6, bias=False),
            nn.BatchNorm1d(ngf*4*6*6),
            nn.ReLU(True)
        )

        # ====================================================================
        # CORREÇÃO 1: Substituído ConvTranspose2d por Upsample + Conv2d
        # Isso elimina os artefatos de "QR code" (checkerboard).
        # ====================================================================
        self.net = nn.Sequential(
            # (ngf*4, 6, 6) -> (ngf*4, 12, 12)
            nn.Upsample(scale_factor=2, mode='nearest'),
            # -> (ngf*2, 12, 12)
            nn.Conv2d(ngf*4, ngf*2, 3, 1, 1, bias=False), # K=3,S=1,P=1 preserva tamanho
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            # -> (ngf*2, 24, 24)
            nn.Upsample(scale_factor=2, mode='nearest'),
            # -> (ngf, 24, 24)
            nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # -> (ngf, 48, 48)
            nn.Upsample(scale_factor=2, mode='nearest'),
            # -> (ngf//2, 48, 48)
            nn.Conv2d(ngf, ngf//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),
            
            # -> (1, 48, 48)
            nn.Conv2d(ngf//2, 1, 3, 1, 1, bias=False),
            nn.Tanh() # Mapeia para [-1, 1]
        )

    def forward(self, z, labels):
        # z: (B, nz), labels: (B,)
        y = self.embed(labels)                       # (B, n_classes)
        x = torch.cat([z, y], dim=1)                 # (B, nz+n_classes)
        x = self.fc(x)                               # (B, ngf*4*6*6)
        x = x.view(x.size(0), -1, 6, 6)              # Reshape to (B, ngf*4, 6, 6)
        x = self.net(x)                              # (B, 1, 48, 48)
        return x

# -------- Discriminator (Conditional) --------
class Discriminator(nn.Module):
    def __init__(self, n_classes=7, ndf=64):
        super().__init__()
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, n_classes)

        # ====================================================================
        # CORREÇÃO 2: A rede é dividida.
        # 1. O 'conv_stack' processa APENAS a imagem.
        # 2. O 'final_conv' processa as features combinadas.
        # ====================================================================

        # 1. Pilha de convolução da imagem (entrada 1x48x48)
        self.conv_stack = nn.Sequential(
            # (1, 48, 48) -> (ndf, 24, 24)
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*2, 12, 12)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*4, 6, 6)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Convolução final (classificador)
        # A entrada será (ndf*4) da imagem + (n_classes) do label
        in_channels_final = ndf*4 + n_classes
        self.final_conv = nn.Sequential(
            # -> (1, 1, 1)
            nn.Conv2d(in_channels_final, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # x: (B,1,48,48); labels: (B,)
        
        # 1. Processa a imagem para extrair features
        # (B, 1, 48, 48) -> (B, ndf*4, 6, 6)
        img_features = self.conv_stack(x)
        
        # 2. Processa o label para corresponder ao tamanho das features
        # (B,) -> (B, n_classes)
        y = self.embed(labels)
        # (B, n_classes) -> (B, n_classes, 1, 1)
        y = y.unsqueeze(-1).unsqueeze(-1)
        # (B, n_classes, 1, 1) -> (B, n_classes, 6, 6)
        y = y.expand(-1, -1, 6, 6)
        
        # 3. Concatena features da imagem e do label
        # (B, ndf*4, 6, 6) + (B, n_classes, 6, 6) -> (B, ndf*4 + n_classes, 6, 6)
        x = torch.cat([img_features, y], dim=1)
        
        # 4. Classifica
        out = self.final_conv(x) # (B,1,1,1)
        return out.view(-1, 1)