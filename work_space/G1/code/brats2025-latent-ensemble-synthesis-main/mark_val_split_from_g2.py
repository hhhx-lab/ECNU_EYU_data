#!/usr/bin/env python3
"""Mark G1 preprocess CSV rows as train/val/test using the G2 fixed split.

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
DEFAULT_SPLIT = G2_RESULTS / "splits" / "splits_final_train_val_test.json"
LEGACY_SPLIT = G2_RESULTS / "splits" / "splits_final_fold0_realval.json"
DEFAULT_MAPPING = G2_RESULTS / "manifests" / "nnunet_case_mapping_realonly.csv"
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "data_csv.csv"


def load_split_ids(split_path: Path, mapping_path: Path) -> dict[str, set[str]]:
    if not split_path.exists() and split_path == DEFAULT_SPLIT and LEGACY_SPLIT.exists():
        split_path = LEGACY_SPLIT

    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    fold0 = split_data[0] if isinstance(split_data, list) else split_data
    split_nnunet = {
        "train": set(fold0.get("train", [])),
        "val": set(fold0.get("val", [])),
        "test": set(fold0.get("test", [])),
    }
    if not split_nnunet["test"] and split_path.name == LEGACY_SPLIT.name:
        # Historical two-way split: old val is the locked internal test.
        split_nnunet["test"] = split_nnunet["val"]
        split_nnunet["val"] = set()

    split_case_ids = {"train": set(), "val": set(), "test": set()}
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            nnunet_case_id = row.get("nnunet_case_id")
            source_case_id = row.get("source_case_id")
            if not nnunet_case_id or not source_case_id:
                continue
            for split_name, nnunet_ids in split_nnunet.items():
                if nnunet_case_id in nnunet_ids:
                    split_case_ids[split_name].add(source_case_id)
                    break
    return split_case_ids


def rewrite_csv(csv_path: Path, split_case_ids: dict[str, set[str]], output_path: Path | None) -> dict[str, int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "split" not in fieldnames:
        fieldnames.append("split")

    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        case_id = row.get("id")
        if case_id in split_case_ids.get("test", set()):
            row["split"] = "test"
            counts["test"] += 1
        elif case_id in split_case_ids.get("val", set()):
            row["split"] = "val"
            counts["val"] += 1
        else:
            row["split"] = "train"
            counts["train"] += 1

    target = output_path or csv_path
    if target == csv_path:
        backup = csv_path.with_suffix(csv_path.suffix + ".bak_before_g2_split")
        backup.write_text(csv_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply G2 fixed train/val/test split to G1 data_csv.csv.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV))
    parser.add_argument("--split-json", default=str(DEFAULT_SPLIT))
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output-csv", default="", help="Optional output path. Defaults to in-place rewrite with backup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_ids = load_split_ids(Path(args.split_json), Path(args.mapping_csv))
    output = Path(args.output_csv) if args.output_csv else None
    counts = rewrite_csv(Path(args.csv_path), split_ids, output)
    print(f"train={counts['train']}")
    print(f"val={counts['val']}")
    print(f"test={counts['test']}")
    print(f"val_ids_from_g2={len(split_ids['val'])}")
    print(f"test_ids_from_g2={len(split_ids['test'])}")
    print(f"csv={output or Path(args.csv_path)}")


if __name__ == "__main__":
    main()
