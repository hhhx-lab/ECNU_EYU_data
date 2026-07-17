#!/usr/bin/env python
"""
Pre-compute per-lesion V-weight masks for BBDM/EncDec loss weighting.

For each subject, loads the BraTS seg, computes connected components
(individual lesions), and creates weighted masks where each lesion voxel
is set to its V weight:

    V_i = clamp(max_lesion_vol_in_patient / vol_i, 1, 5)

The weighted mask is saved in both image space (256,256,160) and
downsampled to latent space (64,64,40) for use during training.

Usage:
    python precompute_lesion_weights.py \
        --data_csv data/data_csv.csv \
        --data_dir data/input \
        --output_dir data/lesion_weights
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import configs
import synthesis.utils as utils


def load_seg(path, transform_path):
    return utils.load_segmentation_in_model_space(path, transform_path).astype(np.int16)


def compute_lesion_volumes_and_weights(seg):
    """Compute per-lesion volumes and V weights from seg.

    Returns:
        weight_map: np.ndarray (same shape as seg), each lesion voxel = V_i
        lesion_info: list of dicts with label, volume, V_weight
    """
    tumor_mask = (seg > 0).astype(np.int32)
    labeled, num_lesions = ndimage.label(tumor_mask)

    if num_lesions == 0:
        return np.zeros_like(seg, dtype=np.float32), []

    volumes = []
    for i in range(1, num_lesions + 1):
        vol = int(np.sum(labeled == i))
        volumes.append(vol)

    max_vol = max(volumes)

    weight_map = np.zeros_like(seg, dtype=np.float32)
    lesion_info = []

    for i, vol in enumerate(volumes, 1):
        v_weight = np.clip(max_vol / vol, 1.0, 5.0)
        weight_map[labeled == i] = v_weight
        lesion_info.append({
            "lesion_id": i,
            "volume_voxels": vol,
            "V_weight": float(v_weight),
        })

    return weight_map, lesion_info


def downsample_weight_map(weight_map, target_shape):
    """Downsample weight map from image space to target shape using
    max-pooling within each block (preserves lesion presence)."""
    source_shape = np.asarray(weight_map.shape, dtype=np.int64)
    target_shape_array = np.asarray(target_shape, dtype=np.int64)
    if np.all(source_shape % target_shape_array == 0):
        factor = source_shape // target_shape_array
        pooled = weight_map.reshape(
            target_shape[0], factor[0],
            target_shape[1], factor[1],
            target_shape[2], factor[2],
        ).max(axis=(1, 3, 5))
        pooled[pooled == 0] = 1.0
        return pooled.astype(np.float32, copy=False)

    factor = source_shape / target_shape_array
    result = np.zeros(target_shape, dtype=np.float32)
    for x in range(target_shape[0]):
        for y in range(target_shape[1]):
            for z in range(target_shape[2]):
                x0, x1 = int(x * factor[0]), int(min((x + 1) * factor[0], weight_map.shape[0]))
                y0, y1 = int(y * factor[1]), int(min((y + 1) * factor[1], weight_map.shape[1]))
                z0, z1 = int(z * factor[2]), int(min((z + 1) * factor[2], weight_map.shape[2]))
                block = weight_map[x0:x1, y0:y1, z0:z1]
                if block.max() > 0:
                    result[x, y, z] = block.max()
                else:
                    result[x, y, z] = 1.0  # healthy region
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/lesion_weights")
    parser.add_argument(
        "--transforms_dir",
        type=str,
        default="data/latents",
        help="Per-case spatial_transform.json root written by preprocess.py",
    )
    parser.add_argument("--latent_shapes", type=str, default="64_64_40",
                        help="Comma-separated latent shapes, e.g. '64_64_40,32_32_20'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse latent shapes
    latent_shapes = []
    for s in args.latent_shapes.split(","):
        dims = tuple(int(x) for x in s.strip().split("_"))
        latent_shapes.append(dims)

    # Load all subjects
    all_subjects = []
    with open(args.data_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_subjects.append(row)

    print(f"Total subjects: {len(all_subjects)}")
    stats = {"num_lesions": [], "min_vol": [], "max_vol": [],
             "V_min": [], "V_max": []}
    failures = []
    processed = 0

    for subj in tqdm(all_subjects, desc="Computing lesion weights"):
        s_id = subj["id"]
        try:
            seg_path = os.path.join(args.data_dir, s_id,
                                    os.path.basename(subj["seg"]))
            transform_path = os.path.join(
                args.transforms_dir, s_id, "spatial_transform.json"
            )
            if not os.path.isfile(transform_path):
                raise FileNotFoundError(
                    f"missing spatial transform for {s_id}: {transform_path}"
                )
            seg = load_seg(seg_path, transform_path)

            weight_map, lesion_info = compute_lesion_volumes_and_weights(seg)

            # Save image-space weight map
            out_subj_dir = os.path.join(args.output_dir, s_id)
            os.makedirs(out_subj_dir, exist_ok=True)

            img_path = os.path.join(out_subj_dir,
                                    f"{s_id}_lesion_weights_256_256_160.npy")
            np.save(img_path, weight_map)

            # Downsample and save for each latent shape
            for shape in latent_shapes:
                wm_ds = downsample_weight_map(weight_map, shape)
                shape_str = "_".join(str(x) for x in shape)
                lt_path = os.path.join(out_subj_dir,
                                       f"{s_id}_lesion_weights_{shape_str}.npy")
                np.save(lt_path, wm_ds)

            # Track stats
            if lesion_info:
                vols = [li["volume_voxels"] for li in lesion_info]
                vws = [li["V_weight"] for li in lesion_info]
                stats["num_lesions"].append(len(lesion_info))
                stats["min_vol"].append(min(vols))
                stats["max_vol"].append(max(vols))
                stats["V_min"].append(min(vws))
                stats["V_max"].append(max(vws))
            processed += 1

        except Exception as e:
            failures.append({"subject": s_id, "error": repr(e)})
            print(f"\n  [ERROR] {s_id}: {e}")

    # Print summary
    print(f"\nProcessed {processed}/{len(all_subjects)} subjects successfully.")
    if stats["num_lesions"]:
        print(f"  Lesions per subject:  mean={np.mean(stats['num_lesions']):.1f}  "
              f"median={np.median(stats['num_lesions']):.0f}  "
              f"max={max(stats['num_lesions'])}")
        print(f"  Lesion volume (voxels):  min={np.mean(stats['min_vol']):.0f}  "
              f"max={np.mean(stats['max_vol']):.0f}")
        print(f"  V weight range:  [{np.mean(stats['V_min']):.2f}, "
              f"{np.mean(stats['V_max']):.2f}]")

    if failures:
        raise RuntimeError(
            f"Lesion-weight generation failed for {len(failures)}/"
            f"{len(all_subjects)} subjects: {failures[:5]}"
        )
    if processed != len(all_subjects):
        raise RuntimeError(
            f"Lesion-weight count mismatch: processed={processed}, "
            f"expected={len(all_subjects)}"
        )
    print(f"\nLesion weight maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
