#!/usr/bin/env python3
"""Build the fixed S2 completion-online split from a G2 materialized dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="ascii")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-split-dir", required=True, type=Path)
    parser.add_argument("--expected-train", type=int, default=1035)
    parser.add_argument("--expected-val", type=int, default=103)
    parser.add_argument("--expected-test", type=int, default=104)
    parser.add_argument("--expected-completions", type=int, default=212)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    split_payload = read_json(dataset_dir / "g2_fixed_split.json")
    split = split_payload[0] if isinstance(split_payload, list) else split_payload
    train = [str(value) for value in split["train"]]
    val = [str(value) for value in split["val"]]
    test = [str(value) for value in split["test"]]
    expected = {
        "train": args.expected_train,
        "val": args.expected_val,
        "test": args.expected_test,
    }
    actual = {"train": len(train), "val": len(val), "test": len(test)}
    if actual != expected:
        raise ValueError(f"Split counts {actual} != {expected}")
    if len(set(train) | set(val) | set(test)) != sum(actual.values()):
        raise ValueError("Train/val/test IDs overlap or contain duplicates")

    rows = [
        row for row in read_csv(dataset_dir / "g2_materialization_manifest.csv")
        if row["modality"] == "seg"
    ]
    by_id = {row["nnunet_case_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Materialization manifest contains duplicate case IDs")
    if set(by_id) != set(train) | set(val) | set(test):
        raise ValueError("Materialization manifest and fixed split IDs differ")

    train_types = Counter(by_id[case_id]["row_type"] for case_id in train)
    val_types = Counter(by_id[case_id]["row_type"] for case_id in val)
    test_types = Counter(by_id[case_id]["row_type"] for case_id in test)
    if train_types.get("real_with_completion_t2w", 0) != args.expected_completions:
        raise ValueError(
            f"Expected {args.expected_completions} train completions, got {dict(train_types)}")
    if set(train_types) - {"real", "real_with_completion_t2w"}:
        raise ValueError(f"Unexpected train row types: {dict(train_types)}")
    if val_types != {"real": args.expected_val}:
        raise ValueError(f"Validation must be authentic-only: {dict(val_types)}")
    if test_types != {"real": args.expected_test}:
        raise ValueError(f"Locked test must be authentic-only: {dict(test_types)}")

    output = args.output_split_dir.expanduser().resolve()
    atomic_lines(output / "train_fixed.txt", train)
    atomic_lines(output / "val_fixed.txt", val)
    atomic_lines(output / "test_locked.txt", test)
    summary = {
        "dataset_dir": str(dataset_dir),
        "counts": actual,
        "train_row_types": dict(train_types),
        "val_row_types": dict(val_types),
        "test_row_types": dict(test_types),
        "patient_overlap": 0,
    }
    (output / "completion_online_split_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "COMPLETION_ONLINE_SPLIT_PASS "
        f"train={len(train)} val={len(val)} test={len(test)} "
        f"completions={args.expected_completions}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
