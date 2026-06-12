from pathlib import Path
import shutil

TRAIN_ROOT = Path(
    "/root/autodl-tmp/brats2026/data/extracted_full/MICCAI-LH-BraTS2025-MET-Challenge-Training"
)

CORRECTED_ROOT = Path(
    "/root/autodl-tmp/brats2026/data/corrected/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"
)

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
