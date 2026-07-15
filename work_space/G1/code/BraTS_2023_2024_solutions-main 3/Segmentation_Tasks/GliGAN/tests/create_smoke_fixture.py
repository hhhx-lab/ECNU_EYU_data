#!/usr/bin/env python3
"""Create a tiny asymmetric BraTS-MET fixture for CPU/GPU pipeline tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def write_case(root: Path, case_id: str, lesion_start: tuple[int, int, int], seed: int) -> None:
    shape = (32, 28, 20)  # deliberately asymmetric to catch x/z swaps
    rng = np.random.default_rng(seed)
    brain_mask = np.zeros(shape, dtype=bool)
    brain_mask[2:31, 2:27, 1:19] = True
    base = np.zeros(shape, dtype=np.float32)
    base[brain_mask] = rng.normal(100.0, 12.0, size=int(brain_mask.sum()))

    seg = np.zeros(shape, dtype=np.uint8)
    x0, y0, z0 = lesion_start
    seg[x0:x0 + 3, y0:y0 + 3, z0:z0 + 3] = 3
    affine = np.array([
        [-1.0, 0.0, 0.0, 31.0],
        [0.0, 1.0, 0.0, -14.0],
        [0.0, 0.0, 1.2, -12.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    for modality, scale in (("t1n", 0.9), ("t1c", 1.0), ("t2w", 1.1), ("t2f", 1.2)):
        image = (base * scale + rng.normal(0.0, 0.5, size=shape) * brain_mask).astype(np.float32)
        nib.save(nib.Nifti1Image(image, affine), case_dir / f"{case_id}-{modality}.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine), case_dir / f"{case_id}-seg.nii.gz")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    dataset = output_root / "DataSet"
    dataset.mkdir(parents=True, exist_ok=True)

    cases = [
        ("BraTS-MET-99998-000", "train", (26, 12, 2), 11),
        ("BraTS-MET-99998-001", "train", (8, 5, 9), 13),
        ("BraTS-MET-99999-000", "val", (4, 18, 14), 19),
    ]
    for case_id, _split, start, seed in cases:
        write_case(dataset, case_id, start, seed)

    manifest_path = output_root / "fixture_split_manifest.csv"
    fieldnames = [
        "source_case_id", "patient_group", "split", "t2w_status",
        "allowed_as_v2_source", "t1n_path", "t1c_path", "t2w_path",
        "t2f_path", "seg_path", "label_source",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_id, split, _start, _seed in cases:
            case_dir = dataset / case_id
            writer.writerow({
                "source_case_id": case_id,
                "patient_group": case_id.rsplit("-", 1)[0],
                "split": split,
                "t2w_status": "authentic",
                "allowed_as_v2_source": str(split == "train"),
                "t1n_path": case_dir / f"{case_id}-t1n.nii.gz",
                "t1c_path": case_dir / f"{case_id}-t1c.nii.gz",
                "t2w_path": case_dir / f"{case_id}-t2w.nii.gz",
                "t2f_path": case_dir / f"{case_id}-t2f.nii.gz",
                "seg_path": case_dir / f"{case_id}-seg.nii.gz",
                "label_source": "fixture",
            })

    print(f"dataset={dataset}")
    print(f"split_manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
