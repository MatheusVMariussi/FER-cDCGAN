
# FER2013 cDCGAN – Conditional GAN for Facial Expression Synthesis

This repo contains a **conditional DCGAN** (cDCGAN) to generate FER-like grayscale faces at **48×48** by emotion label.
It’s designed to **augment** FER2013 when data is scarce.

## Features
- cDCGAN with class conditioning in G and D
- Grayscale output (1×48×48) normalized to `[-1, 1]`
- Works with **image folders** (recommended) or the original **fer2013.csv**
- Checkpointing and fixed-noise sampling every few epochs
- Script to generate per-class images after training

## Project structure
```
FER-cDCGAN/
├── data/                  # put your dataset here (see below)
├── checkpoints/           # models will be saved here
├── generated/             # synthetic images saved here
├── models.py              # cDCGAN architectures (G and D)
├── datasets.py            # FER2013 loaders (folders or CSV)
├── train_cdcgan.py        # training script
├── generate_samples.py    # sample generation by class
├── utils.py               # misc helpers
└── requirements.txt
```

## Dataset layout options

### (A) Folder dataset (recommended)
```
data/fer2013/
├── train/
│   ├── angry/ *.png|jpg
│   ├── disgust/ ...
│   ├── fear/ ...
│   ├── happy/ ...
│   ├── neutral/ ...
│   ├── sad/ ...
│   └── surprise/ ...
└── val/ (optional, not used by GAN)
```

### (B) CSV dataset (original kaggle format)
Place `fer2013.csv` in `data/` and, at the top of `train_cdcgan.py` inside the `CONFIG` block, set:
- `csv_path = 'data/fer2013.csv'`
- `data_root = None`

## Quick start (folder dataset)

```bash
# 1) Edit the CONFIG block at the top of train_cdcgan.py
#    Set 'data_root' to your dataset path (or configure CSV).

# 2) Train (no CLI flags needed)
python train_cdcgan.py

# 3) Edit the CONFIG block at the top of generate_samples.py
#    Set 'checkpoint', 'out_dir', and 'num_per_class'.

# 4) Generate images (no CLI flags needed)
python generate_samples.py
```

## Notes
- Number of classes defaults to 7: `['angry','disgust','fear','happy','neutral','sad','surprise']`.
- If your dataset maps labels differently, update the `labels` list in the `CONFIG` block inside `train_cdcgan.py` (and `generate_samples.py`) in the same order as your folders.
- Images are saved as 8-bit PNG with values scaled back to `[0, 255]`.

## License
MIT (for this template). Verify FER2013 license/terms separately.
