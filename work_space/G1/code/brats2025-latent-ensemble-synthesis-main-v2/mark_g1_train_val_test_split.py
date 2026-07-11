#!/usr/bin/env python3
"""Write deterministic patient-grouped train/val/test splits into data_csv.csv."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, default=Path("data/data_csv.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--output-summary", type=Path, default=Path("data/g1_split_summary.json"))
    parser.add_argument("--membership-csv", type=Path, default=Path("data/g1_split_membership.csv"))
    return parser.parse_args()


def patient_group(case_id: str) -> str:
    """Keep all BraTS-MET-xxxxx-yyy records from one patient in one split."""
    prefix, separator, suffix = case_id.rpartition("-")
    return prefix if separator and suffix.isdigit() else case_id


def main() -> None:
    args = parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0:
        raise ValueError("Split fractions must be non-negative.")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    with args.data_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "id" not in fieldnames:
        raise ValueError(f"{args.data_csv} must contain an 'id' column.")
    if not rows:
        raise ValueError(f"{args.data_csv} is empty.")
    if "split" not in fieldnames:
        fieldnames.append("split")

    ids = [str(row["id"]) for row in rows]
    duplicates = [case_id for case_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate case IDs in {args.data_csv}: {duplicates[:10]}")

    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for case_id in sorted(ids):
        grouped_ids[patient_group(case_id)].append(case_id)
    groups = sorted(grouped_ids)
    random.Random(args.seed).shuffle(groups)

    requested_holdouts = int(args.test_fraction > 0) + int(args.val_fraction > 0)
    if len(groups) <= requested_holdouts:
        raise ValueError(
            f"Need more than {requested_holdouts} patient groups to keep train/val/test non-empty."
        )

    target_test_cases = round(len(ids) * args.test_fraction)
    target_val_cases = round(len(ids) * args.val_fraction)
    if args.test_fraction > 0:
        target_test_cases = max(1, target_test_cases)
    if args.val_fraction > 0:
        target_val_cases = max(1, target_val_cases)
    split_by_group: dict[str, str] = {}
    assigned_cases = Counter()
    for group in groups:
        group_size = len(grouped_ids[group])
        if assigned_cases["test"] < target_test_cases:
            split = "test"
        elif assigned_cases["val"] < target_val_cases:
            split = "val"
        else:
            split = "train"
        split_by_group[group] = split
        assigned_cases[split] += group_size

    split_by_id = {
        case_id: split_by_group[group]
        for group, case_ids in grouped_ids.items()
        for case_id in case_ids
    }
    for row in rows:
        row["split"] = split_by_id[str(row["id"])]

    temp_csv = args.data_csv.with_suffix(args.data_csv.suffix + ".tmp")
    with temp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_csv.replace(args.data_csv)

    args.membership_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.membership_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "patient_group", "split"])
        writer.writeheader()
        for row in rows:
            case_id = str(row["id"])
            writer.writerow(
                {
                    "id": case_id,
                    "patient_group": patient_group(case_id),
                    "split": row["split"],
                }
            )

    split_counts = Counter(str(row["split"]) for row in rows)
    group_counts = Counter(split_by_group.values())
    summary = {
        "seed": args.seed,
        "grouping_rule": "remove final numeric case suffix",
        "val_fraction_requested": args.val_fraction,
        "test_fraction_requested": args.test_fraction,
        "total_cases": len(ids),
        "total_patient_groups": len(groups),
        "case_counts": dict(sorted(split_counts.items())),
        "patient_group_counts": dict(sorted(group_counts.items())),
        "actual_case_fractions": {
            split: split_counts[split] / len(ids) for split in ("train", "val", "test")
        },
        "data_csv": str(args.data_csv),
        "membership_csv": str(args.membership_csv),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
