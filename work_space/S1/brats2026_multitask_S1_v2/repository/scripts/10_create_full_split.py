from pathlib import Path
import random

random.seed(42)

ROOT = Path(
    "/root/autodl-tmp/brats2026/data/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
)

OUT_DIR = Path(
    "/root/autodl-tmp/brats2026/data/splits"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

cases = sorted(
    [
        x.name
        for x in ROOT.rglob("BraTS-MET-*")
        if x.is_dir()
    ]
)

random.shuffle(cases)

n = len(cases)

n_val = int(n * 0.1)

val_cases = cases[:n_val]

train_cases = cases[n_val:]

with open(
    OUT_DIR / "train_full.txt",
    "w"
) as f:

    for c in train_cases:
        f.write(c + "\n")

with open(
    OUT_DIR / "val_full.txt",
    "w"
) as f:

    for c in val_cases:
        f.write(c + "\n")

print("train =", len(train_cases))
print("val =", len(val_cases))
