#!/usr/bin/env python3
"""Audit S1 training view: every case must have 4 modalities + seg + tumor/RC labels.

Exit code 1 if any case is incomplete, so Slurm never starts training on a bad view.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = "data/extracted/MICCAI-LH-BraTS2025-MET-Challenge-Training"
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


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
    parser.add_argument(
        "--split-dir",
        default=os.environ.get("BRATS_SPLIT_DIR", ""),
        help="If set, also verify every train/val case in split files exists under train-root",
    )
    args = parser.parse_args()

    root = resolve_path(args.train_root)
    if not root.is_dir():
        print(f"ERROR: train root does not exist: {root}", file=sys.stderr)
        return 1

    cases = sorted(path for path in root.rglob("BraTS-MET-*") if path.is_dir())
    n_case = len(cases)
    n_seg = 0
    n_tumor = 0
    n_rc = 0
    bad_cases = []

    for case_dir in cases:
        case = case_dir.name
        problems = []

        for mod in MODALITIES:
            mod_path = case_dir / f"{case}-{mod}.nii.gz"
            if not mod_path.exists():
                problems.append(f"missing {mod}")
            elif mod_path.is_symlink() and not mod_path.resolve().exists():
                problems.append(f"broken symlink {mod}")

        seg = list(case_dir.glob("*-seg.nii.gz"))
        tumor = list(case_dir.glob("tumor_label.nii.gz"))
        rc = list(case_dir.glob("rc_label.nii.gz"))

        n_seg += len(seg)
        n_tumor += len(tumor)
        n_rc += len(rc)

        if len(seg) != 1:
            problems.append(f"seg count={len(seg)}")
        elif seg[0].is_symlink() and not seg[0].resolve().exists():
            problems.append("broken symlink seg")
        if len(tumor) != 1:
            problems.append(f"tumor_label count={len(tumor)}")
        if len(rc) != 1:
            problems.append(f"rc_label count={len(rc)}")

        if problems:
            bad_cases.append(f"{case}: {', '.join(problems)}")

    print(f"train_root = {root}")
    print(f"cases = {n_case}")
    print(f"seg = {n_seg}")
    print(f"tumor = {n_tumor}")
    print(f"rc = {n_rc}")
    print(f"bad = {len(bad_cases)}")

    if n_case == 0:
        print("ERROR: no cases found", file=sys.stderr)
        return 1

    if bad_cases:
        print("ERROR: incomplete cases detected:", file=sys.stderr)
        for line in bad_cases[:30]:
            print(f"  - {line}", file=sys.stderr)
        if len(bad_cases) > 30:
            print(f"  ... and {len(bad_cases) - 30} more", file=sys.stderr)
        return 1

    if args.split_dir:
        split_dir = resolve_path(args.split_dir)
        case_names = {path.name for path in cases}
        for split_name in ("train_full.txt", "val_full.txt"):
            split_path = split_dir / split_name
            if not split_path.is_file():
                print(f"ERROR: missing split file: {split_path}", file=sys.stderr)
                return 1
            listed = [
                line.strip()
                for line in split_path.read_text().splitlines()
                if line.strip()
            ]
            if not listed:
                print(f"ERROR: empty split file: {split_path}", file=sys.stderr)
                return 1
            missing = [name for name in listed if name not in case_names]
            if missing:
                print(
                    f"ERROR: {split_name} has {len(missing)} cases not present in view",
                    file=sys.stderr,
                )
                print("  examples:", missing[:10], file=sys.stderr)
                return 1
            print(f"{split_name}: {len(listed)} cases OK")

    print("audit finished: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
