"""Generate attention masks from tumor segmentation files.

Converts full-resolution seg NIfTI files to binary attention masks
in latent space (64, 64, 40) for BBDM training.

Pipeline (mirrors MRI preprocessing exactly):
  1. Load seg NIfTI → binarize (> 0.5 → 1)
  2. Center-crop/pad to (256, 256, 160)   ← SAME transform as MRI
  3. 4x block max-pool to (64, 64, 40)     ← aligns with VAE 4x downsampling

Usage:
    python generate_attmask.py

Input:
    data/input/<subject_id>/<subject_id>-seg.nii.gz

Output:
    data/attention_masks/<subject_id>/<subject_id>_attmask_64_64_40.npy
"""
import os
import numpy as np
import pandas as pd
import configs
import synthesis.utils as utils


def center_crop_pad(image, new_shape):
    """Same operation as utils.resize_center_crop_pad (no affine)."""
    x, y, z = image.shape
    nx, ny, nz = new_shape
    new_image = np.zeros((nx, ny, nz), dtype=image.dtype)

    def get_slices(old, new):
        if old > new:
            start = (old - new) // 2
            return slice(start, start + new), slice(0, new)
        else:
            start = (new - old) // 2
            return slice(0, old), slice(start, start + old)

    xs_old, xs_new = get_slices(x, nx)
    ys_old, ys_new = get_slices(y, ny)
    zs_old, zs_new = get_slices(z, nz)
    new_image[xs_new, ys_new, zs_new] = image[xs_old, ys_old, zs_old]
    return new_image


def downsample_max_pool(image, factor=4):
    """4x block max-pooling. If any voxel in a block is tumor, output is tumor."""
    d, h, w = image.shape
    return image.reshape(d // factor, factor,
                         h // factor, factor,
                         w // factor, factor).max(axis=(1, 3, 5))


def main():
    csv_path = os.path.join(configs.PATH_DATA, "data_csv.csv")
    input_dir = configs.PATH_INPUT
    attmask_dir = os.path.join(configs.PATH_DATA, "attention_masks")
    image_shape = configs.SHAPE_PREPROCESS_IMG   # (256, 256, 160)
    latent_shape = (64, 64, 40)

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        s_id = row["id"]
        subject_input = os.path.join(input_dir, s_id)

        # Find seg file
        seg_file = None
        for f in sorted(os.listdir(subject_input)):
            if "-seg.nii" in f:
                seg_file = f
                break

        if seg_file is None:
            print(f"  SKIP {s_id}: no seg file found")
            continue

        # Step 1: load & binarize
        seg_path = os.path.join(subject_input, seg_file)
        seg, _ = utils.load_nifti(seg_path)
        seg = (seg > 0.5).astype(np.float32)

        # Step 2: center-crop/pad → (256, 256, 160), same as MRI
        seg = center_crop_pad(seg, image_shape)

        # Step 3: 4x block max-pool → (64, 64, 40), aligned with VAE latent grid
        seg_latent = downsample_max_pool(seg)

        # Save
        out_dir = os.path.join(attmask_dir, s_id)
        os.makedirs(out_dir, exist_ok=True)
        tag = "_".join(str(s) for s in latent_shape)
        out_path = os.path.join(out_dir, f"{s_id}_attmask_{tag}.npy")
        np.save(out_path, seg_latent)

        tumor_ratio = seg_latent.sum() / seg_latent.size
        print(f"  {s_id}: tumor_ratio = {tumor_ratio:.4f}  ->  {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
