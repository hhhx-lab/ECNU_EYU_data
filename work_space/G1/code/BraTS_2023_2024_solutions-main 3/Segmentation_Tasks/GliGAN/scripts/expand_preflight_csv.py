#!/usr/bin/env python3
"""Expand a smoke lesion CSV to exercise the requested full batch size."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError("CSV header is missing")
    train = next((row for row in rows if row["split"] == "train"), None)
    val = next((row for row in rows if row["split"] == "val"), None)
    if train is None or val is None:
        raise ValueError("Preflight CSV must contain train and val rows")

    expanded = []
    for index in range(args.batch_size):
        row = dict(train)
        row["lesion_id"] = f"{train['lesion_id']}_preflight_{index}"
        expanded.append(row)
    expanded.append(val)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded)
    print(f"PREFLIGHT_CSV_PASS train_rows={args.batch_size} val_rows=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
