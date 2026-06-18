#!/usr/bin/env python3
"""Mark G1 preprocess CSV rows as train/val using the G2 fixed fold.

Run this after `python preprocess.py` has generated `data/data_csv.csv`.
It preserves the CSV columns and only rewrites the `split` column.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
G2_RESULTS = PROJECT_ROOT / "work_space" / "G2" / "results"
DEFAULT_SPLIT = G2_RESULTS / "splits" / "splits_final_fold0_realval.json"
DEFAULT_MAPPING = G2_RESULTS / "manifests" / "nnunet_case_mapping_realonly.csv"
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "data_csv.csv"


def load_val_ids(split_path: Path, mapping_path: Path) -> set[str]:
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    fold0 = split_data[0] if isinstance(split_data, list) else split_data
    val_nnunet = set(fold0["val"])

    val_case_ids: set[str] = set()
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("nnunet_case_id") in val_nnunet and row.get("source_case_id"):
                val_case_ids.add(row["source_case_id"])
    return val_case_ids


def rewrite_csv(csv_path: Path, val_case_ids: set[str], output_path: Path | None) -> tuple[int, int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "split" not in fieldnames:
        fieldnames.append("split")

    train_count = 0
    val_count = 0
    for row in rows:
        if row.get("id") in val_case_ids:
            row["split"] = "val"
            val_count += 1
        else:
            row["split"] = "train"
            train_count += 1

    target = output_path or csv_path
    if target == csv_path:
        backup = csv_path.with_suffix(csv_path.suffix + ".bak_before_g2_split")
        backup.write_text(csv_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return train_count, val_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply G2 fixed fold0 train/val split to G1 data_csv.csv.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV))
    parser.add_argument("--split-json", default=str(DEFAULT_SPLIT))
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output-csv", default="", help="Optional output path. Defaults to in-place rewrite with backup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_ids = load_val_ids(Path(args.split_json), Path(args.mapping_csv))
    output = Path(args.output_csv) if args.output_csv else None
    train_count, val_count = rewrite_csv(Path(args.csv_path), val_ids, output)
    print(f"train={train_count}")
    print(f"val={val_count}")
    print(f"val_ids_from_g2={len(val_ids)}")
    print(f"csv={output or Path(args.csv_path)}")


if __name__ == "__main__":
    main()
