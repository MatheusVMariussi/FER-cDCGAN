
import os
import torch
from torchvision.utils import save_image
from models import Generator
from utils import ensure_dir
from types import SimpleNamespace

DEFAULT_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

# =============================
# Configuração simples (edite aqui)
# =============================
CONFIG = {
    'checkpoint': 'checkpoints/G_epoch_50.pth',
    'out_dir': 'generated/epoch50',
    'num_per_class': 100,
    'labels': DEFAULT_LABELS,
    'nz': 100,
    'ngf': 64,
    'img_size': 48,
    'seed': 123,
}

def get_config():
    return SimpleNamespace(**CONFIG)

def main():
    args = get_config()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(args.labels)

    netG = Generator(nz=args.nz, n_classes=n_classes, ngf=args.ngf, img_size=args.img_size).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    ensure_dir(args.out_dir)

    for cls_idx, cls_name in enumerate(args.labels):
        cls_dir = os.path.join(args.out_dir, cls_name)
        ensure_dir(cls_dir)
        total = args.num_per_class
        bs = 64
        saved = 0
        while saved < total:
            cur = min(bs, total-saved)
            noise = torch.randn(cur, args.nz, device=device)
            labels = torch.full((cur,), cls_idx, dtype=torch.long, device=device)
            with torch.no_grad():
                imgs = netG(noise, labels).cpu()
            imgs = (imgs + 1)/2  # [0,1]
            for i in range(cur):
                save_image(imgs[i], os.path.join(cls_dir, f"{saved+i:06d}.png"))
            saved += cur
        print(f"Saved {saved} images for class '{cls_name}' to {cls_dir}")

if __name__ == "__main__":
    main()
