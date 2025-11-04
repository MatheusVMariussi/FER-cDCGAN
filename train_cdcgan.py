
import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from types import SimpleNamespace

from datasets import FERFolder, FERCsv, DEFAULT_LABELS
from models import Generator, Discriminator
from utils import save_image_grid, ensure_dir

# =============================
# Configuração simples (edite aqui)
# =============================
# Defina abaixo os parâmetros padrão. Eles só são lidos quando o script é
# executado (em main()), não durante o import.
CONFIG = {
    # Dados: escolha UMA das opções abaixo
    # a) Dataset em pastas (recomendado):
    'data_root': 'data/fer2013/train',
    # b) CSV original do FER2013 (mantenha data_root=None se usar CSV):
    'csv_path': None,                 # exemplo: 'data/fer2013.csv'
    'usage': 'Training',              # split do CSV
    'labels': DEFAULT_LABELS,         # ordem das classes
    'img_size': 48,

    # Treinamento
    'epochs': 50,
    'batch_size': 256,
    'lr': 2e-4,
    'beta1': 0.5,
    'nz': 100,
    'ngf': 64,
    'ndf': 64,

    # Saídas
    'out_dir': 'checkpoints',
    'sample_dir': 'generated',
    'sample_every': 5,

    # Reprodutibilidade
    'seed': 42,
}

def get_config():
    return SimpleNamespace(**CONFIG)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataset(args):
    if args.data_root:
        ds = FERFolder(args.data_root, labels=args.labels, img_size=args.img_size)
    elif args.csv_path:
        ds = FERCsv(args.csv_path, usage=args.usage, labels=args.labels, img_size=args.img_size)
    else:
        raise ValueError("Provide either data_root or csv_path in CONFIG")
    return ds

def main():
    args = get_config()
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = build_dataset(args)
    n_classes = len(args.labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    netG = Generator(nz=args.nz, n_classes=n_classes, ngf=args.ngf, img_size=args.img_size).to(device)
    netD = Discriminator(n_classes=n_classes, ndf=args.ndf).to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(n_classes*8, args.nz, device=device)
    fixed_labels = torch.tensor([i for i in range(n_classes) for _ in range(8)], device=device)

    ensure_dir(args.out_dir); ensure_dir(args.sample_dir)

    for epoch in range(1, args.epochs+1):
        netG.train(); netD.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            bsz = real_imgs.size(0)
            # Real and fake labels
            real = torch.full((bsz,1), 1.0, device=device)
            fake = torch.full((bsz,1), 0.0, device=device)

            # ---- Train D ----
            netD.zero_grad()
            out_real = netD(real_imgs, labels)
            lossD_real = criterion(out_real, real)

            noise = torch.randn(bsz, args.nz, device=device)
            fake_labels = torch.randint(0, n_classes, (bsz,), device=device)
            fake_imgs = netG(noise, fake_labels)
            out_fake = netD(fake_imgs.detach(), fake_labels)
            lossD_fake = criterion(out_fake, fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # ---- Train G ----
            netG.zero_grad()
            out_fake2 = netD(fake_imgs, fake_labels)
            lossG = criterion(out_fake2, real)
            lossG.backward()
            optimizerG.step()

            pbar.set_postfix({'D': f"{lossD.item():.3f}", 'G': f"{lossG.item():.3f}"})

        # Sampling & checkpoint
        if (epoch % args.sample_every) == 0 or epoch==args.epochs:
            with torch.no_grad():
                samples = netG(fixed_noise, fixed_labels).cpu()
            save_path = os.path.join(args.sample_dir, f"samples_epoch_{epoch}.png")
            save_image(samples, save_path, nrow=8, normalize=True, value_range=(-1,1))
            torch.save(netG.state_dict(), os.path.join(args.out_dir, f"G_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(args.out_dir, f"D_epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()
