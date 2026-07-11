#!/usr/bin/env python3
"""Validate G1 V2 placement, fake-T2W isolation, and internal split counts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--expected-fake", type=int, default=265)
    parser.add_argument("--expected-total", type=int, default=1030)
    parser.add_argument("--expected-train", type=int, default=823)
    parser.add_argument("--expected-val", type=int, default=103)
    parser.add_argument("--expected-test", type=int, default=104)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def has_suffix(case_dir: Path, suffix: str) -> bool:
    return any(
        path.name.endswith(f"-{suffix}.nii.gz")
        or path.name.endswith(f"-{suffix}.nii")
        for path in case_dir.iterdir()
        if path.is_file()
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    placement = read_csv(data_dir / "g1_v2_data_placement_manifest.csv")
    data_rows = read_csv(data_dir / "data_csv.csv")
    label_rows = read_csv(data_dir / "g1_v2_label_filter_report.csv")

    errors: list[str] = []
    fake_rows = [
        row
        for row in placement
        if row.get("t2w_is_fake_by_gzip_header", "").lower() == "true"
    ]
    input_rows = [
        row for row in placement if row.get("target") == "input" and row.get("status") == "placed"
    ]
    inference_rows = [
        row
        for row in placement
        if row.get("target") == "input_inference" and row.get("status") == "placed"
    ]
    input_ids = {row["case_id"] for row in input_rows}
    inference_ids = {row["case_id"] for row in inference_rows}
    csv_ids = {row["id"] for row in data_rows}

    if len(fake_rows) != args.expected_fake:
        errors.append(f"fake_count:{len(fake_rows)}!={args.expected_fake}")
    if len(inference_ids) != args.expected_fake:
        errors.append(f"inference_count:{len(inference_ids)}!={args.expected_fake}")
    if input_ids & inference_ids:
        errors.append(f"train_inference_overlap:{sorted(input_ids & inference_ids)[:10]}")
    fake_ids = {row["case_id"] for row in fake_rows}
    if fake_ids != inference_ids:
        errors.append("fake_ids_do_not_match_inference_ids")
    if fake_ids & csv_ids:
        errors.append(f"fake_t2w_leaked_into_csv:{sorted(fake_ids & csv_ids)[:10]}")
    if len(data_rows) != args.expected_total:
        errors.append(f"csv_total:{len(data_rows)}!={args.expected_total}")

    split_counts = Counter(row.get("split", "") for row in data_rows)
    expected_splits = {
        "train": args.expected_train,
        "val": args.expected_val,
        "test": args.expected_test,
    }
    if dict(split_counts) != expected_splits:
        errors.append(f"split_counts:{dict(split_counts)}!={expected_splits}")

    label_by_id = {row["case_id"]: row for row in label_rows}
    excluded_ids = {
        case_id for case_id, row in label_by_id.items() if row.get("status") == "excluded"
    }
    accepted_ids = {
        case_id for case_id, row in label_by_id.items() if row.get("status") == "accepted"
    }
    if accepted_ids != csv_ids:
        errors.append("accepted_label_ids_do_not_match_csv")
    if excluded_ids != {"BraTS-MET-01094-002"}:
        errors.append(f"unexpected_label_exclusions:{sorted(excluded_ids)}")
    corrected = label_by_id.get("BraTS-MET-01184-002", {})
    if corrected.get("status") != "accepted" or corrected.get("label_source") != "corrected":
        errors.append("BraTS-MET-01184-002_not_using_corrected_label")

    for case_id in sorted(csv_ids):
        case_dir = data_dir / "input" / case_id
        required = (*MODALITIES, "seg")
        missing = [suffix for suffix in required if not has_suffix(case_dir, suffix)]
        if missing:
            errors.append(f"input_missing:{case_id}:{missing}")
    for case_id in sorted(inference_ids):
        case_dir = data_dir / "input_inference" / case_id
        required = ("t1n", "t1c", "t2f", "seg")
        missing = [suffix for suffix in required if not has_suffix(case_dir, suffix)]
        if missing:
            errors.append(f"inference_missing:{case_id}:{missing}")
        if has_suffix(case_dir, "t2w"):
            errors.append(f"inference_contains_t2w:{case_id}")

    summary = {
        "status": "failed" if errors else "passed",
        "training_candidates_before_label_filter": len(input_ids),
        "fake_t2w_inference_cases": len(inference_ids),
        "effective_csv_cases": len(data_rows),
        "split_counts": dict(split_counts),
        "excluded_label_cases": sorted(excluded_ids),
        "corrected_label_case": "BraTS-MET-01184-002",
        "error_count": len(errors),
        "errors": errors,
    }
    output = data_dir / "g1_v2_preparation_validation.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise RuntimeError(f"G1 V2 preparation validation failed; see {output}")


if __name__ == "__main__":
    main()
