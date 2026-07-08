import argparse
import os
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = "data/extracted_full/MICCAI-LH-BraTS2025-MET-Challenge-Training"
DEFAULT_CORRECTED_ROOT = "data/corrected/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"


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
    "--corrected-root",
    default=os.environ.get("BRATS_CORRECTED_ROOT", DEFAULT_CORRECTED_ROOT)
)
args = parser.parse_args()

TRAIN_ROOT = resolve_path(args.train_root)
CORRECTED_ROOT = resolve_path(args.corrected_root)

for seg_file in CORRECTED_ROOT.glob("*-seg.nii.gz"):

    case_id = seg_file.name.replace("-seg.nii.gz", "")

    matches = list(TRAIN_ROOT.rglob(case_id))

    if len(matches) == 0:
        print(f"[NOT FOUND] {case_id}")
        continue

    case_dir = matches[0]

    target_seg = case_dir / f"{case_id}-seg.nii.gz"

    shutil.copy2(seg_file, target_seg)

    print(f"[UPDATED] {case_id}")

print("Done.")
