#!/usr/bin/env python3
"""Build tumor_label.nii.gz and rc_label.nii.gz for every S1 case folder.

Fail-fast: any missing/ambiguous seg aborts the job so training never starts
on a silently incomplete view.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = "data/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root",
        default=os.environ.get("BRATS_TRAIN_ROOT", DEFAULT_TRAIN_ROOT),
    )
    args = parser.parse_args()

    root = resolve_path(args.train_root)
    if not root.is_dir():
        print(f"ERROR: train root does not exist: {root}", file=sys.stderr)
        return 1

    cases = sorted(path for path in root.rglob("BraTS-MET-*") if path.is_dir())
    print(f"cases = {len(cases)}")
    print(f"train_root = {root}")
    if not cases:
        print(f"ERROR: no BraTS-MET-* case folders under {root}", file=sys.stderr)
        return 1

    failures = []
    for i, case_dir in enumerate(cases):
        seg_files = list(case_dir.glob("*-seg.nii.gz"))
        if len(seg_files) != 1:
            failures.append(f"{case_dir.name}: expected 1 seg, found {len(seg_files)}")
            continue

        try:
            img = nib.load(str(seg_files[0]))
            seg = np.asanyarray(img.dataobj)
            # Tumor task: drop RC (label 4)
            tumor = seg.copy()
            tumor[tumor == 4] = 0
            tumor = tumor.astype(np.uint8)
            nib.save(
                nib.Nifti1Image(tumor, img.affine, img.header),
                str(case_dir / "tumor_label.nii.gz"),
            )
            # RC task: binary mask of label 4
            rc = np.zeros_like(seg, dtype=np.uint8)
            rc[seg == 4] = 1
            nib.save(
                nib.Nifti1Image(rc, img.affine, img.header),
                str(case_dir / "rc_label.nii.gz"),
            )
        except Exception as exc:  # noqa: BLE001 - surface any I/O/decode issue
            failures.append(f"{case_dir.name}: {exc}")
            continue

        if i % 100 == 0:
            print(i, case_dir.name)

    if failures:
        print(f"ERROR: failed to build labels for {len(failures)} cases", file=sys.stderr)
        for line in failures[:30]:
            print(f"  - {line}", file=sys.stderr)
        if len(failures) > 30:
            print(f"  ... and {len(failures) - 30} more", file=sys.stderr)
        return 1

    print("done")
    print(f"built tumor_label + rc_label for {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
