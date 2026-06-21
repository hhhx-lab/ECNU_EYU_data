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


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "work_space" / "G1").exists() and (parent / "work_space" / "G2").exists():
            return parent
    raise RuntimeError(f"Could not locate ECNU_EYU_data project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
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


def rewrite_csv(
    csv_path: Path,
    split_case_ids: dict[str, set[str]],
    output_path: Path | None,
    allow_unmatched_as_train: bool,
) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "split" not in fieldnames:
        fieldnames.append("split")

    counts = {"train": 0, "val": 0, "test": 0}
    matched_ids = split_case_ids.get("train", set()) | split_case_ids.get("val", set()) | split_case_ids.get("test", set())
    unmatched: list[str] = []
    for row in rows:
        case_id = row.get("id")
        if case_id in split_case_ids.get("test", set()):
            row["split"] = "test"
            counts["test"] += 1
        elif case_id in split_case_ids.get("val", set()):
            row["split"] = "val"
            counts["val"] += 1
        elif case_id in split_case_ids.get("train", set()):
            row["split"] = "train"
            counts["train"] += 1
        else:
            unmatched.append(case_id or "")
            row["split"] = "train"
            counts["train"] += 1

    if unmatched and not allow_unmatched_as_train:
        preview = ", ".join(unmatched[:20])
        raise SystemExit(
            f"{len(unmatched)} CSV case ids are not present in the G2 split/mapping; "
            f"stop before rewriting split. examples: {preview}"
        )

    target = output_path or csv_path
    if target == csv_path:
        backup = csv_path.with_suffix(csv_path.suffix + ".bak_before_g2_split")
        backup.write_text(csv_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "counts": counts,
        "unmatched": unmatched,
        "matched_id_count": len(matched_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply G2 fixed train/val/test split to G1 data_csv.csv.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV))
    parser.add_argument("--split-json", default=str(DEFAULT_SPLIT))
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output-csv", default="", help="Optional output path. Defaults to in-place rewrite with backup.")
    parser.add_argument(
        "--allow-unmatched-as-train",
        action="store_true",
        help="Keep old conservative behavior: rows missing from G2 mapping are written as train. Default fails fast.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_ids = load_split_ids(Path(args.split_json), Path(args.mapping_csv))
    output = Path(args.output_csv) if args.output_csv else None
    result = rewrite_csv(Path(args.csv_path), split_ids, output, args.allow_unmatched_as_train)
    counts = result["counts"]
    print(f"train={counts['train']}")
    print(f"val={counts['val']}")
    print(f"test={counts['test']}")
    print(f"val_ids_from_g2={len(split_ids['val'])}")
    print(f"test_ids_from_g2={len(split_ids['test'])}")
    print(f"matched_ids_from_g2={result['matched_id_count']}")
    print(f"unmatched_ids={len(result['unmatched'])}")
    print(f"csv={output or Path(args.csv_path)}")


if __name__ == "__main__":
    main()
