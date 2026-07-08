import argparse
import os
from pathlib import Path

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

n_case = len(cases)

n_seg = 0
n_tumor = 0
n_rc = 0

bad_cases = []

for case_dir in cases:

    seg = list(case_dir.glob("*-seg.nii.gz"))

    tumor = list(case_dir.glob("tumor_label.nii.gz"))

    rc = list(case_dir.glob("rc_label.nii.gz"))

    n_seg += len(seg)
    n_tumor += len(tumor)
    n_rc += len(rc)

    if (
        len(seg) != 1
        or len(tumor) != 1
        or len(rc) != 1
    ):
        bad_cases.append(case_dir.name)

print("cases =", n_case)
print("seg =", n_seg)
print("tumor =", n_tumor)
print("rc =", n_rc)

print("bad =", len(bad_cases))

if len(bad_cases) > 0:
    print(bad_cases[:20])

print("audit finished")
