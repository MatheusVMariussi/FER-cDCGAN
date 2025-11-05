import os
import random
import time
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from types import SimpleNamespace

from datasets import FERFolder, FERCsv, DEFAULT_LABELS
from models import Generator, Discriminator  # ou PatchGANDiscriminator
from utils import ensure_dir

# =============================
# CONFIGURAÇÃO MELHORADA
# =============================
CONFIG = {
    # Dados
    'data_root': 'data/fer2013/train',
    'csv_path': None,
    'usage': 'Training',
    'labels': DEFAULT_LABELS,
    'img_size': 48,

    # Treinamento - AJUSTADO
    'epochs': 300,
    'batch_size': 128,           # REDUZIDO para melhor estabilidade
    'lr_g': 1e-4,                # REDUZIDO
    'lr_d': 2e-4,                # AUMENTADO (D precisa ser mais forte)
    'beta1': 0.5,
    'beta2': 0.999,
    'nz': 100,
    'ngf': 64,
    'ndf': 64,
    
    # Técnicas de estabilização
    'label_smoothing': 0.05,     # REDUZIDO (era 0.1)
    'label_noise': 0.05,         # Adiciona ruído aos labels
    'n_critic': 1,               # Treina D 1x por iteração
    'gradient_penalty': 0.0,     # 0 = desabilitado, 10.0 = GP ativo
    
    # Saídas
    'out_dir': 'checkpoints',
    'sample_dir': 'generated',
    'sample_every': 10,
    
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataset(args):
    if args.data_root:
        ds = FERFolder(args.data_root, labels=args.labels, img_size=args.img_size)
    elif args.csv_path:
        ds = FERCsv(args.csv_path, usage=args.usage, labels=args.labels, img_size=args.img_size)
    else:
        raise ValueError("Provide either data_root or csv_path in CONFIG")
    return ds

def setup_logging(out_dir):
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(log_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    """Calcula Gradient Penalty (WGAN-GP)"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = D(interpolates, labels)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    args = get_config()
    
    ensure_dir(args.out_dir)
    ensure_dir(args.sample_dir)
    
    logger = setup_logging(args.out_dir)
    logger.info("=== TREINAMENTO MELHORADO ===")
    logger.info(f"Configuração: {vars(args)}")

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")

    ds = build_dataset(args)
    n_classes = len(args.labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                       num_workers=4, drop_last=True, pin_memory=True)
    logger.info(f"Dataset: {len(ds)} amostras, {n_classes} classes.")

    netG = Generator(nz=args.nz, n_classes=n_classes, ngf=args.ngf, img_size=args.img_size).to(device)
    netD = Discriminator(n_classes=n_classes, ndf=args.ndf).to(device)
    # OU use: netD = PatchGANDiscriminator(n_classes=n_classes, ndf=args.ndf).to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    
    # Schedulers para diminuir LR ao longo do tempo
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)

    fixed_noise = torch.randn(n_classes*8, args.nz, device=device)
    fixed_labels = torch.tensor([i for i in range(n_classes) for _ in range(8)], device=device)

    logger.info("Iniciando treinamento...")
    
    for epoch in range(1, args.epochs+1):
        netG.train()
        netD.train()
        
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_d_real_acc = []
        epoch_d_fake_acc = []

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            bsz = real_imgs.size(0)

            # ==================
            # Train Discriminator
            # ==================
            for _ in range(args.n_critic):
                netD.zero_grad()
                
                # Labels com smoothing e noise
                real_labels = torch.ones(bsz, 1, device=device) * (1.0 - args.label_smoothing)
                fake_labels = torch.zeros(bsz, 1, device=device) + args.label_smoothing
                
                # Adiciona ruído aos labels ocasionalmente
                if args.label_noise > 0 and random.random() < args.label_noise:
                    real_labels, fake_labels = fake_labels, real_labels
                
                # Real images
                out_real = netD(real_imgs, labels)
                lossD_real = criterion(out_real, real_labels)
                
                # Fake images
                noise = torch.randn(bsz, args.nz, device=device)
                fake_labels_class = torch.randint(0, n_classes, (bsz,), device=device)
                fake_imgs = netG(noise, fake_labels_class).detach()
                out_fake = netD(fake_imgs, fake_labels_class)
                lossD_fake = criterion(out_fake, fake_labels)
                
                # Total D loss
                lossD = lossD_real + lossD_fake
                
                # Gradient Penalty (opcional)
                if args.gradient_penalty > 0:
                    gp = compute_gradient_penalty(netD, real_imgs, fake_imgs, fake_labels_class, device)
                    lossD = lossD + args.gradient_penalty * gp
                
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)  # Gradient clipping
                optimizerD.step()
                
                # Métricas de acurácia
                d_real_acc = (out_real > 0.5).float().mean().item()
                d_fake_acc = (out_fake < 0.5).float().mean().item()

            # ==================
            # Train Generator
            # ==================
            netG.zero_grad()
            
            noise = torch.randn(bsz, args.nz, device=device)
            gen_labels = torch.randint(0, n_classes, (bsz,), device=device)
            fake_imgs = netG(noise, gen_labels)
            out_gen = netD(fake_imgs, gen_labels)
            
            # G quer que D classifique como real
            lossG = criterion(out_gen, torch.ones(bsz, 1, device=device))
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
            optimizerG.step()
            
            # Métricas
            epoch_d_losses.append(lossD.item())
            epoch_g_losses.append(lossG.item())
            epoch_d_real_acc.append(d_real_acc)
            epoch_d_fake_acc.append(d_fake_acc)
            
            pbar.set_postfix({
                'D': f"{lossD.item():.3f}",
                'G': f"{lossG.item():.3f}",
                'D_real': f"{d_real_acc:.2f}",
                'D_fake': f"{d_fake_acc:.2f}"
            })

        pbar.close()
        
        # Atualiza learning rates
        schedulerD.step()
        schedulerG.step()
        
        # Estatísticas do epoch
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_real_acc = np.mean(epoch_d_real_acc)
        avg_d_fake_acc = np.mean(epoch_d_fake_acc)
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f} | "
            f"D_real_acc: {avg_d_real_acc:.3f} | D_fake_acc: {avg_d_fake_acc:.3f} | "
            f"LR_D: {schedulerD.get_last_lr()[0]:.6f} | LR_G: {schedulerG.get_last_lr()[0]:.6f}"
        )

        # Salvamento
        if (epoch % args.sample_every) == 0 or epoch == args.epochs:
            logger.info(f"Salvando checkpoint epoch {epoch}...")
            with torch.no_grad():
                netG.eval()
                samples = netG(fixed_noise, fixed_labels).cpu()
                netG.train()
            
            save_path = os.path.join(args.sample_dir, f"samples_epoch_{epoch}.png")
            save_image(samples, save_path, nrow=8, normalize=True, value_range=(-1, 1))
            
            torch.save({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
            }, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pth"))

    logger.info("Treinamento concluído!")

if __name__ == "__main__":
    main()