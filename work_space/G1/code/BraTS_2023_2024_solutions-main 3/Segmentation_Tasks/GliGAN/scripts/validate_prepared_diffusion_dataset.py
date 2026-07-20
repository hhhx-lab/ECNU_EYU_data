#!/usr/bin/env python3
"""Validate the fixed G1 Diffusion V3 train/validation view."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--membership", required=True, type=Path)
    parser.add_argument("--lesions", required=True, type=Path)
    parser.add_argument("--expected-train", required=True, type=int)
    parser.add_argument("--expected-val", required=True, type=int)
    args = parser.parse_args()

    membership = read_rows(args.membership)
    expected_counts = {"train": args.expected_train, "val": args.expected_val}
    case_counts = Counter(row["split"].strip().lower() for row in membership)
    if dict(case_counts) != expected_counts:
        raise ValueError(f"Case counts {dict(case_counts)} != {expected_counts}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in membership:
        group_splits[row["patient_group"]].add(row["split"].strip().lower())
        for key in ("t1n_path", "t1c_path", "t2w_path", "t2f_path", "seg_path"):
            source = Path(row[key])
            if not source.is_file():
                raise FileNotFoundError(f"Missing source file: {source}")
    leaks = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaks:
        raise ValueError(f"Patient groups cross splits: {leaks[:10]}")

    lesions = read_rows(args.lesions)
    if not lesions:
        raise ValueError("Lesion CSV is empty")
    lesion_counts = Counter(row["split"].strip().lower() for row in lesions)
    if set(lesion_counts) != {"train", "val"}:
        raise ValueError(f"Unexpected lesion splits: {dict(lesion_counts)}")
    if any(int(row["n_voxels"]) <= 0 for row in lesions):
        raise ValueError("Lesion CSV contains a non-positive n_voxels value")

    print(
        "DIFFUSION_DATASET_CONTRACT_PASS "
        f"cases={dict(case_counts)} lesions={dict(lesion_counts)} overlap=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
