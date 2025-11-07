import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Calcula desvio padrão do minibatch
        std = torch.std(x, dim=0, keepdim=True).mean()
        # Replica para todas as posições
        std_map = std.repeat(batch_size, 1, height, width)
        return torch.cat([x, std_map], dim=1)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Reduz dimensionalidade para eficiência
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, HW, C')
        k = self.key(x).view(batch_size, -1, H * W)  # (B, C', HW)
        v = self.value(x).view(batch_size, -1, H * W)  # (B, C, HW)
        
        # Attention map
        attention = F.softmax(torch.bmm(q, k), dim=-1)  # (B, HW, HW)
        
        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (B, C, HW)
        out = out.view(batch_size, C, H, W)
        
        # Residual com peso aprendível
        return self.gamma * out + x

class SpectralNorm:
    def __init__(self, module, name='weight', power_iterations=1):
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = w.data.new(height).normal_(0, 1)
        v = w.data.new(width).normal_(0, 1)
        u = u / u.norm()
        v = v / v.norm()
        
        self.module.register_buffer(self.name + "_u", u)
        self.module.register_buffer(self.name + "_v", v)

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name)
        
        height = w.data.shape[0]
        w_mat = w.view(height, -1)
        
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(w_mat.t(), u), dim=0)
            u = F.normalize(torch.mv(w_mat, v), dim=0)
        
        sigma = torch.dot(u, torch.mv(w_mat, v))
        return w / sigma

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

def spectral_norm(module, name='weight', power_iterations=1):
    """Wrapper para aplicar Spectral Norm"""
    SpectralNorm(module, name, power_iterations)
    return module

class HeavyGenerator(nn.Module):
    def __init__(self, nz=256, n_classes=7, ngf=128, img_size=48):
        super().__init__()
        self.nz = nz
        self.n_classes = n_classes
        self.ngf = ngf

        self.embed_dim = 128
        self.embed = nn.Embedding(n_classes, self.embed_dim)
        
        # Mapping network (StyleGAN-inspired)
        in_dim = nz + self.embed_dim
        self.mapping = nn.Sequential(
            PixelNorm(),
            nn.Linear(in_dim, ngf * 8),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            nn.Linear(ngf * 8, ngf * 8),
            nn.LeakyReLU(0.2),
            PixelNorm(),
            nn.Linear(ngf * 8, ngf * 8),
            nn.LeakyReLU(0.2),
        )
        
        # Síntese inicial (4x4)
        self.init = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 16, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf * 16),
            nn.ReLU(True),
        )
        
        # Blocos de upsampling progressivos
        # 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 16, ngf * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(True),
        )
        
        # 8x8 -> 16x16
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        
        # Self-Attention em 16x16
        self.attn = SelfAttention(ngf * 4)
        
        # 16x16 -> 32x32
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
        )
        
        # 32x32 -> 48x48
        self.up4 = nn.Sequential(
            nn.Upsample(size=(48, 48), mode='bilinear', align_corners=False),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        )
        
        # Camada final para imagem
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.InstanceNorm2d):
                # Verifica se os parâmetros existem (affine=True)
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z, labels):
        # Mapping
        y = self.embed(labels)
        x = torch.cat([z, y], dim=1)
        x = self.mapping(x)
        
        # Reshape para convolução
        x = x.view(x.size(0), -1, 1, 1)
        
        # Síntese progressiva
        x = self.init(x)      # 4x4
        x = self.up1(x)       # 8x8
        x = self.up2(x)       # 16x16
        x = self.attn(x)      # Self-attention
        x = self.up3(x)       # 32x32
        x = self.up4(x)       # 48x48
        x = self.to_rgb(x)    # Imagem final
        
        return x

class HeavyDiscriminator(nn.Module):
    def __init__(self, n_classes=7, ndf=128):
        super().__init__()
        self.n_classes = n_classes
        self.ndf = ndf
        
        # Embedding para condicionamento
        self.embed_dim = ndf * 4
        self.embed = nn.Embedding(n_classes, self.embed_dim)
        
        # Blocos convolucionais
        # 48x48 -> 24x24
        self.block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(1, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 24x24 -> 12x12
        self.block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 12x12 -> 6x6
        self.block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Self-Attention em 6x6
        self.attn = SelfAttention(ndf * 4)
        
        # 6x6 -> 3x3
        self.block4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Minibatch StdDev
        self.minibatch_std = MinibatchStdDev()
        
        # Camadas finais com condicionamento
        self.final_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8 + 1, ndf * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Classificador com condicionamento
        self.classifier = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 8 + self.embed_dim, ndf * 4, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, 1, 1)),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.InstanceNorm2d):
                # Verifica se os parâmetros existem (affine=True)
                if m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # Processa imagem
        x = self.block1(x)    # 24x24
        x = self.block2(x)    # 12x12
        x = self.block3(x)    # 6x6
        x = self.attn(x)      # Self-attention
        x = self.block4(x)    # 3x3
        
        # Minibatch statistics
        x = self.minibatch_std(x)  # (B, ndf*8+1, 3, 3)
        x = self.final_conv(x)     # (B, ndf*8, 3, 3)
        
        # Embedding do label
        y_embed = self.embed(labels)  # (B, embed_dim)
        y_embed = y_embed.view(batch_size, -1, 1, 1)  # (B, embed_dim, 1, 1)
        y_embed = y_embed.repeat(1, 1, 3, 3)  # (B, embed_dim, 3, 3)
        
        # Concatena features + embedding
        x = torch.cat([x, y_embed], dim=1)  # (B, ndf*8+embed_dim, 3, 3)
        
        # Classificação final
        x = self.classifier(x) # (B, 1, 3, 3)
        
        return x.mean(dim=[2, 3]).view(-1, 1)