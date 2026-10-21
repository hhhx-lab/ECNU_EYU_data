#!/usr/bin/env python3
"""Exclude cases with unreadable or out-of-contract segmentation labels from G1 V3."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


ALLOWED_LABELS = {0, 1, 2, 3, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, default=Path("data/data_csv.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/input"))
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("data/g1_v3_label_filter_report.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/g1_v3_label_filter_summary.json"),
    )
    parser.add_argument("--max-excluded-fraction", type=float, default=0.01)
    return parser.parse_args()


def illegal_label_values(path: Path) -> list[int | float]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if np.issubdtype(data.dtype, np.integer):
        min_label = int(data.min())
        max_label = int(data.max())
        if min_label >= min(ALLOWED_LABELS) and max_label <= max(ALLOWED_LABELS):
            return []
        return sorted(
            int(value) for value in np.unique(data) if int(value) not in ALLOWED_LABELS
        )

    invalid = ~np.isfinite(data) | ~np.isin(data, tuple(ALLOWED_LABELS))
    if not invalid.any():
        return []
    return [float(value) for value in np.unique(data[invalid])]


def main() -> None:
    args = parse_args()
    if not 0 <= args.max_excluded_fraction < 1:
        raise ValueError("--max-excluded-fraction must be in [0, 1).")
    with args.data_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"Input CSV is empty: {args.data_csv}")
    for column in ("id", "seg"):
        if column not in fieldnames:
            raise ValueError(f"Input CSV is missing required column: {column}")

    accepted = []
    report = []
    for index, row in enumerate(rows, start=1):
        case_id = str(row["id"])
        seg_path = args.data_dir / case_id / str(row["seg"])
        resolved_path = ""
        label_source = "raw"
        illegal: list[int | float] = []
        error = ""
        try:
            resolved_path = str(seg_path.resolve(strict=True))
            if "corrected-labels" in resolved_path:
                label_source = "corrected"
            illegal = illegal_label_values(seg_path)
        except Exception as exc:
            error = repr(exc)

        status = "accepted" if not illegal and not error else "excluded"
        reason = "" if status == "accepted" else (
            f"illegal_seg_labels:{illegal}" if illegal else f"seg_read_error:{error}"
        )
        report.append(
            {
                "case_id": case_id,
                "status": status,
                "reason": reason,
                "label_source": label_source,
                "seg_path": str(seg_path),
                "resolved_seg_path": resolved_path,
            }
        )
        if status == "accepted":
            accepted.append(row)
        if index % 100 == 0 or index == len(rows):
            print(f"Checked labels: {index}/{len(rows)}")

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)

    excluded = [row for row in report if row["status"] == "excluded"]
    excluded_fraction = len(excluded) / len(rows)
    summary = {
        "allowed_labels": sorted(ALLOWED_LABELS),
        "input_cases": len(rows),
        "accepted_cases": len(accepted),
        "excluded_cases": len(excluded),
        "excluded_fraction": excluded_fraction,
        "excluded_case_ids": [row["case_id"] for row in excluded],
        "report_csv": str(args.report_csv),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if excluded_fraction > args.max_excluded_fraction:
        raise RuntimeError(
            f"Excluded fraction {excluded_fraction:.4%} exceeds "
            f"limit {args.max_excluded_fraction:.4%}; CSV was not changed."
        )
    if not accepted:
        raise RuntimeError("Every case failed segmentation label validation.")

    temp_csv = args.data_csv.with_suffix(args.data_csv.suffix + ".tmp")
    with temp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accepted)
    os.replace(temp_csv, args.data_csv)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in excluded:
        print(f"EXCLUDED: {row['case_id']}: {row['reason']}")


if __name__ == "__main__":
    main()
