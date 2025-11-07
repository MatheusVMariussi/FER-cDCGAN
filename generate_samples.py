import os
import torch
from torchvision.utils import save_image
from models_heavy2 import HeavyGenerator
from utils import ensure_dir
from types import SimpleNamespace

DEFAULT_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

CONFIG = {
    'checkpoint': 'checkpoints_heavy/G_ema_epoch_500.pth', # Mude para o epoch que você quer
    'out_dir': 'generated_heavy/epoch500_ema', # Mude o nome do diretório
    
    'num_per_class': 100,
    'labels': DEFAULT_LABELS,
    'img_size': 48,
    'seed': 123,
    
    'nz': 256,
    'ngf': 128,
}

def get_config():
    return SimpleNamespace(**CONFIG)

def main():
    args = get_config()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(args.labels)

    netG = HeavyGenerator(
        nz=args.nz, 
        n_classes=n_classes, 
        ngf=args.ngf, 
        img_size=args.img_size
    ).to(device)
    
    try:
        state = torch.load(args.checkpoint, map_location=device)
        # Se salvou o checkpoint completo, procure 'netG_state_dict' ou 'ema_shadow'
        if 'netG_state_dict' in state:
            netG.load_state_dict(state['netG_state_dict'])
        elif 'ema_shadow' in state:
             # Se for o checkpoint completo, o EMA está em 'ema_shadow'
             # Precisamos carregar manualmente
            model_dict = netG.state_dict()
            for name, param in state['ema_shadow'].items():
                if name in model_dict:
                    model_dict[name].copy_(param)
        else:
            # Se salvou apenas o state_dict (como os novos G_epoch_X.pth)
            netG.load_state_dict(state)
            
        print(f"Pesos carregados de {args.checkpoint}")
            
    except Exception as e:
        print(f"Erro ao carregar checkpoint: {e}")
        print("Certifique-se que o 'checkpoint' aponta para um arquivo .pth válido.")
        return

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
            
            # Denorm de [-1, 1] para [0, 1]
            imgs = (imgs + 1) / 2
            
            for i in range(cur):
                img_path = os.path.join(cls_dir, f"{saved+i:06d}.png")
                save_image(imgs[i], img_path)
            saved += cur
        print(f"Salvou {saved} imagens para a classe '{cls_name}' em {cls_dir}")

if __name__ == "__main__":
    main()