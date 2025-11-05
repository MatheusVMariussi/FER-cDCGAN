import torch
import torch.nn as nn

# -------- Generator (Melhorado) --------
class Generator(nn.Module):
    def __init__(self, nz=100, n_classes=7, ngf=64, img_size=48):
        super().__init__()
        self.nz = nz
        self.n_classes = n_classes
        self.embed_dim = 50
        self.embed = nn.Embedding(n_classes, self.embed_dim)
        in_dim = nz + self.embed_dim
        
        # Inicialização inicial mais robusta
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf*8*6*6, bias=False),
            nn.BatchNorm1d(ngf*8*6*6),
            nn.ReLU(True)
        )
        
        # Rede convolucional mais profunda e estável
        self.net = nn.Sequential(
            # 6x6 -> 12x12
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            
            # 12x12 -> 24x24
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            # 24x24 -> 48x48
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # Camada final
            nn.Conv2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z, labels):
        y = self.embed(labels)
        x = torch.cat([z, y], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 6, 6)
        x = self.net(x)
        return x

# -------- Discriminator (Projection - MELHORADO) --------
class Discriminator(nn.Module):
    def __init__(self, n_classes=7, ndf=64):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = 50
        
        self.embed = nn.Embedding(n_classes, self.embed_dim)
        
        # Rede convolucional MAIS PROFUNDA e SEM DROPOUT inicial
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
            
            # -> (ndf*8, 3, 3) - CAMADA EXTRA
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Feature layer mais robusta
        self.feature_layer = nn.Sequential(
            nn.Linear(ndf*8*3*3, ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)  # Dropout APENAS aqui
        )
        
        # Pontuação não-condicional
        self.unconditional_score = nn.Linear(ndf*8, 1)
        
        # Projeção condicional
        self.conditional_projection = nn.Linear(self.embed_dim, ndf*8)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels):
        # Features da imagem
        h = self.conv_stack(x)
        h = h.view(h.size(0), -1)
        h = self.feature_layer(h)
        
        # Embedding do label
        y = self.embed(labels)
        
        # Score não-condicional
        score_uncond = self.unconditional_score(h)
        
        # Score condicional (projeção)
        y_proj = self.conditional_projection(y)
        score_cond = (h * y_proj).sum(dim=1, keepdim=True)
        
        # Score total
        total_score = score_uncond + score_cond
        
        return torch.sigmoid(total_score)


# -------- DISCRIMINADOR ALTERNATIVO: PatchGAN --------
# Se o Projection não funcionar bem, use este
class PatchGANDiscriminator(nn.Module):
    def __init__(self, n_classes=7, ndf=64):
        super().__init__()
        self.n_classes = n_classes
        
        # Embedding do label projetado como canal adicional
        self.label_embedding = nn.Embedding(n_classes, 48*48)
        
        self.net = nn.Sequential(
            # (2, 48, 48) -> (ndf, 24, 24)
            nn.Conv2d(2, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (ndf*2, 12, 12)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (ndf*4, 6, 6)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (ndf*8, 3, 3)
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (1, 3, 3) - PatchGAN output
            nn.Conv2d(ndf*8, 1, 3, 1, 0),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels):
        # Cria canal de label
        label_map = self.label_embedding(labels)
        label_map = label_map.view(-1, 1, 48, 48)
        
        # Concatena imagem + label
        x_with_label = torch.cat([x, label_map], dim=1)
        
        # Retorna média dos patches
        return self.net(x_with_label).mean(dim=[2, 3])