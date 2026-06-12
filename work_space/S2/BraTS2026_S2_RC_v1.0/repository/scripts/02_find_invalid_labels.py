from pathlib import Path
import nibabel as nib
import numpy as np

TRAIN_ROOT = Path(
    "/root/autodl-tmp/brats2026/data/extracted_full/MICCAI-LH-BraTS2025-MET-Challenge-Training"
)

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
