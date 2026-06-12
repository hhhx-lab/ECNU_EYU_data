from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(
    "/root/autodl-tmp/brats2026/data/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
)

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
