import argparse
import os
from pathlib import Path
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = "data/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"


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

ROOT = resolve_path(args.train_root)

cases = sorted(
    [
        x
        for x in ROOT.rglob("BraTS-MET-*")
        if x.is_dir()
    ]
)

print("cases =", len(cases))

for i, case_dir in enumerate(cases):

    seg_files = list(
        case_dir.glob("*-seg.nii.gz")
    )

    if len(seg_files) != 1:
        print("skip:", case_dir)
        continue

    seg_file = seg_files[0]

    img = nib.load(seg_file)

    seg = img.get_fdata()

    # Tumor task
    tumor = seg.copy()
    tumor[tumor == 4] = 0
    tumor = tumor.astype(np.uint8)

    nib.save(
        nib.Nifti1Image(
            tumor,
            img.affine,
            img.header
        ),
        case_dir / "tumor_label.nii.gz"
    )

    # RC task
    rc = np.zeros_like(
        seg,
        dtype=np.uint8
    )

    rc[seg == 4] = 1

    nib.save(
        nib.Nifti1Image(
            rc,
            img.affine,
            img.header
        ),
        case_dir / "rc_label.nii.gz"
    )

    if i % 100 == 0:
        print(i, case_dir.name)

print("done")
