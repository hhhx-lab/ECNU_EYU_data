import argparse
import os
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = "data/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
DEFAULT_SPLIT_DIR = "data/splits"


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train-root",
    default=os.environ.get("BRATS_TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
)
parser.add_argument(
    "--split-dir",
    default=os.environ.get("BRATS_SPLIT_DIR", DEFAULT_SPLIT_DIR)
)
parser.add_argument("--val-ratio", type=float, default=0.1)
parser.add_argument(
    "--seed",
    type=int,
    default=int(os.environ.get("BRATS_SPLIT_SEED", "42"))
)
args = parser.parse_args()

random.seed(args.seed)

ROOT = resolve_path(args.train_root)
OUT_DIR = resolve_path(args.split_dir)

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

n_val = int(n * args.val_ratio)

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
