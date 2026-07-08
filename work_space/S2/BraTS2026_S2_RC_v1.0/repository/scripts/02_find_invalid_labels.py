import argparse
import os
from pathlib import Path
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = "data/extracted_full/MICCAI-LH-BraTS2025-MET-Challenge-Training"


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
args = parser.parse_args()

TRAIN_ROOT = resolve_path(args.train_root)

bad_cases = []

for seg_file in TRAIN_ROOT.rglob("*-seg.nii.gz"):

    arr = nib.load(seg_file).get_fdata(dtype=np.uint8)

    labels = np.unique(arr)

    illegal = [int(x) for x in labels if x not in [0, 1, 2, 3, 4]]

    if len(illegal) > 0:

        print(
            seg_file.parent.name,
            "illegal labels:",
            illegal
        )

        bad_cases.append(seg_file.parent.name)

print()
print("bad cases =", len(bad_cases))
