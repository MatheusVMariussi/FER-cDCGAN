
import os
import csv
from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

DEFAULT_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

def get_default_transform(img_size=48):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),                # [0,1]
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])

class FERFolder(Dataset):
    def __init__(self, root: str, labels: Optional[List[str]]=None, img_size: int=48):
        self.root = root
        self.labels = labels or DEFAULT_LABELS
        self.class_to_idx = {c:i for i,c in enumerate(self.labels)}
        self.samples = []
        for c in self.labels:
            folder = os.path.join(root, c)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if fn.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                    self.samples.append((os.path.join(folder, fn), self.class_to_idx[c]))
        if len(self.samples)==0:
            raise RuntimeError(f"No images found under {root}.")
        self.transform = get_default_transform(img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Loop para garantir que uma imagem válida seja retornada
        while True:
            path, y = self.samples[idx]
            try:
                # Tenta abrir e carregar a imagem
                img = Image.open(path).convert('L')
                
                # Aplica a transformação
                img = self.transform(img)  # (1, H, W) in [-1,1]
                
                # Se tudo deu certo, retorna a imagem e o label
                return img, y
                
            except Exception as e:
                # Se falhar (ex: imagem corrompida), imprime um aviso
                print(f"AVISO: Falha ao carregar {path}. Erro: {e}")
                print("Tentando carregar uma imagem aleatória substituta...")
                
                # Tenta de novo com um índice aleatório
                idx = random.randint(0, len(self.samples) - 1)
