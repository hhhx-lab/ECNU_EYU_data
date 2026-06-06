import os
import json
import random

data_root = "/share/home/zhaomrui/brats2025_data/raw_original/MICCAI-LH-BraTS2025-MET-Challenge-Training"
cases = [d for d in os.listdir(data_root) if d.startswith("BraTS-MET-") and os.path.isdir(os.path.join(data_root, d))]
random.seed(42)
random.shuffle(cases)
split = int(0.8 * len(cases))
train_cases = cases[:split]
val_cases = cases[split:]

data = [{"image": c, "label": c, "fold": 1} for c in train_cases] + [{"image": c, "label": c, "fold": 0} for c in val_cases]
with open("full_split.json", "w") as f:
    json.dump({"training": data}, f, indent=2)

print(f"Total cases: {len(cases)}, train: {len(train_cases)}, val: {len(val_cases)}")
