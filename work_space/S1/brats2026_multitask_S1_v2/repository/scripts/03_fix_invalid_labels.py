from pathlib import Path
import nibabel as nib
import numpy as np

TARGET_CASE = "BraTS-MET-01094-002"

seg_file = next(
    Path(
        "/root/autodl-tmp/brats2026/data/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
    ).rglob(f"{TARGET_CASE}-seg.nii.gz")
)

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
