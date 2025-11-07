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
from models_heavy2 import HeavyGenerator, HeavyDiscriminator
from utils import ensure_dir

# CONFIGURAÇÃO HEAVY (WGAN-GP)
CONFIG = {
    # Dados
    'data_root': 'data/fer2013/train',
    'csv_path': None,
    'usage': 'Training',
    'labels': DEFAULT_LABELS,
    'img_size': 48,

    # Arquitetura HEAVY
    'nz': 256,
    'ngf': 128,
    'ndf': 128,
    
    # Treinamento WGAN-GP
    'epochs': 500,
    'batch_size': 64,
    'lr_g': 1e-4,
    'lr_d': 1e-4,
    'beta1': 0.0,            # Betas recomendadas para WGAN
    'beta2': 0.9,            # Betas recomendadas para WGAN
    'n_critic': 5,           # Treina D 5x mais que G
    'gp_weight': 10.0,       # Peso do Gradient Penalty
    
    'use_ema': True,
    'ema_decay': 0.999,
    
    # Saidas
    'out_dir': 'checkpoints_heavy',
    'sample_dir': 'generated_heavy',
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

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = D(interpolates, labels)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    args = get_config()
    
    ensure_dir(args.out_dir)
    ensure_dir(args.sample_dir)
    
    logger = setup_logging(args.out_dir)
    logger.info("=== TREINAMENTO HEAVY WGAN-GP ===")
    logger.info(f"Configuração: {vars(args)}")

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")

    ds = build_dataset(args)
    n_classes = len(args.labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                       num_workers=0, drop_last=True, pin_memory=True)
    logger.info(f"Dataset: {len(ds)} amostras, {n_classes} classes.")

    # Modelos HEAVY
    netG = HeavyGenerator(nz=args.nz, n_classes=n_classes, ngf=args.ngf, img_size=args.img_size).to(device)
    netD = HeavyDiscriminator(n_classes=n_classes, ndf=args.ndf).to(device)
    
    # Conta parâmetros
    g_params = sum(p.numel() for p in netG.parameters())
    d_params = sum(p.numel() for p in netD.parameters())
    logger.info(f"Generator: {g_params:,} parâmetros ({g_params/1e6:.2f}M)")
    logger.info(f"Discriminator: {d_params:,} parâmetros ({d_params/1e6:.2f}M)")

    # Optimizers (Sem BCELoss)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    
    # EMA do Generator
    if args.use_ema:
        ema = EMA(netG, decay=args.ema_decay)
        logger.info(f"EMA ativado com decay={args.ema_decay}")
    
    # Scheduler
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.995)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.995)

    fixed_noise = torch.randn(n_classes*8, args.nz, device=device)
    fixed_labels = torch.tensor([i for i in range(n_classes) for _ in range(8)], device=device)

    logger.info("Iniciando treinamento HEAVY...")
    
    global_step = 0
    
    for epoch in range(1, args.epochs+1):
        netG.train()
        netD.train()
        
        epoch_d_losses = []
        epoch_g_losses = []
        epoch_gp_penalties = []
        epoch_d_real_scores = []
        epoch_d_fake_scores = []

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            bsz = real_imgs.size(0)
            global_step += 1

            # Trainamento Discriminator
            netD.zero_grad()
            
            # Real
            out_real = netD(real_imgs, labels)
            lossD_real = -torch.mean(out_real) # WGAN: Quer maximizar D(real) -> minimizar -D(real)
            
            # Fake
            noise = torch.randn(bsz, args.nz, device=device)
            fake_labels_class = torch.randint(0, n_classes, (bsz,), device=device)
            fake_imgs = netG(noise, fake_labels_class).detach()
            
            out_fake = netD(fake_imgs, fake_labels_class)
            lossD_fake = torch.mean(out_fake) # WGAN: Quer minimizar D(fake)
            
            # Gradient Penalty (WGAN-GP)
            gradient_penalty = compute_gradient_penalty(netD, real_imgs, fake_imgs, labels)
            
            # Total D loss
            lossD = lossD_real + lossD_fake + (args.gp_weight * gradient_penalty)
            
            lossD.backward()
            optimizerD.step()
            
            # Métricas WGAN
            d_real_score = out_real.mean().item()
            d_fake_score = out_fake.mean().item()

            # Trainamento Generator
            lossG = torch.tensor(0.0) # Default se não treinar
            if global_step % args.n_critic == 0:
                netG.zero_grad()
                
                noise = torch.randn(bsz, args.nz, device=device)
                gen_labels = torch.randint(0, n_classes, (bsz,), device=device)
                fake_imgs_gen = netG(noise, gen_labels)
                out_gen = netD(fake_imgs_gen, gen_labels)
                
                # WGAN: Quer maximizar D(fake) -> minimizar -D(fake)
                lossG = -torch.mean(out_gen)
                
                lossG.backward()
                optimizerG.step()
                
                # Update EMA
                if args.use_ema:
                    ema.update()
            
            # Métricas do Batch
            epoch_d_losses.append(lossD.item())
            epoch_gp_penalties.append(gradient_penalty.item())
            epoch_d_real_scores.append(d_real_score)
            epoch_d_fake_scores.append(d_fake_score)
            if (global_step % args.n_critic == 0):
                epoch_g_losses.append(lossG.item())
                
            pbar.set_postfix({
                    'D_loss': f"{lossD.item():.2f}",
                    'G_loss': f"{lossG.item():.2f}",
                    'GP': f"{gradient_penalty.item():.2f}",
                    'D(real)': f"{d_real_score:.2f}",
                    'D(fake)': f"{d_fake_score:.2f}"
            })
        
        pbar.close()
        
        # Atualiza LR
        schedulerD.step()
        schedulerG.step()
        
        # Estatísticas do Epoch
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_loss = np.mean(epoch_g_losses) if len(epoch_g_losses) > 0 else 0.0
        avg_d_real = np.mean(epoch_d_real_scores)
        avg_d_fake = np.mean(epoch_d_fake_scores)
        avg_gp = np.mean(epoch_gp_penalties)
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"D_loss: {avg_d_loss:.3f} | G_loss: {avg_g_loss:.3f} | "
            f"D(real): {avg_d_real:.3f} | D(fake): {avg_d_fake:.3f} | "
            f"GP: {avg_gp:.3f} | "
            f"LR_D: {schedulerD.get_last_lr()[0]:.6f} | LR_G: {schedulerG.get_last_lr()[0]:.6f}"
        )

        # Salvamento
        if (epoch % args.sample_every) == 0 or epoch == args.epochs:
            logger.info(f"Salvando checkpoint epoch {epoch}...")
            
            # Usa EMA para gerar samples
            if args.use_ema:
                ema.apply_shadow()
            
            with torch.no_grad():
                netG.eval()
                samples = netG(fixed_noise, fixed_labels).cpu()
                netG.train()
            
            if args.use_ema:
                ema.restore()
            
            save_path = os.path.join(args.sample_dir, f"samples_epoch_{epoch}.png")
            save_image(samples, save_path, nrow=8, normalize=True, value_range=(-1, 1))
            
            # Salva o checkpoint completo
            checkpoint = {
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
            }
            if args.use_ema:
                checkpoint['ema_shadow'] = ema.shadow
            
            torch.save(checkpoint, os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pth"))
            
            # Salva apenas o gerador para facilitar a geração
            torch.save(netG.state_dict(), os.path.join(args.out_dir, f"G_epoch_{epoch}.pth"))
            if args.use_ema:
                ema.apply_shadow()
                torch.save(netG.state_dict(), os.path.join(args.out_dir, f"G_ema_epoch_{epoch}.pth"))
                ema.restore()


    logger.info("Treinamento concluído!")

if __name__ == "__main__":
    main()