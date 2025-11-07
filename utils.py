
import os
import math
import torch
import numpy as np
from PIL import Image

def denorm(x):
    return (x + 1.0) / 2.0

def save_image_grid(tensor, path, nrow=8):
    x = denorm(tensor.detach().cpu().clamp(-1,1))
    b, c, h, w = x.shape
    nrow = min(nrow, b)
    ncol = int(math.ceil(b / nrow))
    grid = torch.zeros((c, ncol*h, nrow*w))
    for idx in range(b):
        r = idx // nrow
        cidx = idx % nrow
        grid[:, r*h:(r+1)*h, cidx*w:(cidx+1)*w] = x[idx]
    arr = (grid.squeeze(0).numpy()*255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
