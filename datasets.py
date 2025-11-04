
import os
import csv
from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
        path, y = self.samples[idx]
        img = Image.open(path).convert('L')
        img = self.transform(img)  # (1, H, W) in [-1,1]
        return img, y

class FERCsv(Dataset):
    """Kaggle fer2013.csv loader: columns = [emotion, pixels, Usage]."""
    def __init__(self, csv_path: str, usage: str='Training', labels: Optional[List[str]]=None, img_size: int=48):
        self.csv_path = csv_path
        self.usage = usage
        self.labels = labels or DEFAULT_LABELS
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.img_size = img_size
        self._load()

    def _load(self):
        self.data = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Usage'] != self.usage:
                    continue
                emotion_idx = int(row['emotion'])
                pixels = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ')
                img = pixels.reshape(48,48)
                self.data.append((img, emotion_idx))

        if len(self.data)==0:
            raise RuntimeError(f"No rows for Usage='{self.usage}' in {self.csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, y = self.data[idx]
        img = Image.fromarray(img, mode='L').resize((self.img_size, self.img_size))
        img = self.transform(img) # (1,H,W) in [-1,1]
        return img, y
