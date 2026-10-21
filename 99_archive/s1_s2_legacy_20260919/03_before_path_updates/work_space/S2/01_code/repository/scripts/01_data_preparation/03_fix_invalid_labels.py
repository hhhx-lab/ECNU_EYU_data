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
parser.add_argument(
    "--target-case",
    default=os.environ.get("BRATS_TARGET_CASE", "BraTS-MET-01094-002")
)
args = parser.parse_args()

TARGET_CASE = args.target_case
TRAIN_ROOT = resolve_path(args.train_root)
matches = list(TRAIN_ROOT.rglob(f"{TARGET_CASE}-seg.nii.gz"))

if not matches:
    raise FileNotFoundError(f"Cannot find seg for {TARGET_CASE} under {TRAIN_ROOT}")

seg_file = matches[0]

img = nib.load(seg_file)

arr = img.get_fdata()

n6 = np.sum(arr == 6)
n8 = np.sum(arr == 8)

arr[arr == 6] = 4
arr[arr == 8] = 4

new_img = nib.Nifti1Image(
    arr.astype(np.uint8),
    img.affine,
    img.header
)

nib.save(new_img, seg_file)

print(f"fixed case: {TARGET_CASE}")
print(f"6 -> 4 voxels: {n6}")
print(f"8 -> 4 voxels: {n8}")
