#!/usr/bin/env python3
"""Fail-fast validation for the G1 V2 training CSV and linked NIfTI files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
REQUIRED_COLUMNS = ("id", "seg", *MODALITIES, "split")
ALLOWED_LABELS = {0, 1, 2, 3, 4}


def patient_group(case_id: str) -> str:
    prefix, separator, suffix = case_id.rpartition("-")
    return prefix if separator and suffix.isdigit() else case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, default=Path("data/data_csv.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/input"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/g1_v2_dataset_validation.json"),
    )
    parser.add_argument(
        "--skip-label-values",
        action="store_true",
        help="Skip reading segmentation arrays; header/path checks still run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.data_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    missing_columns = sorted(set(REQUIRED_COLUMNS) - columns)
    errors: list[dict[str, str]] = []
    if missing_columns:
        errors.append({"case_id": "", "error": f"missing_columns:{','.join(missing_columns)}"})
    if not rows:
        errors.append({"case_id": "", "error": "empty_csv"})

    ids = [str(row.get("id", "")) for row in rows]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    for case_id in duplicates:
        errors.append({"case_id": case_id, "error": "duplicate_case_id"})

    split_counts = Counter(str(row.get("split", "")) for row in rows)
    for split in ("train", "val", "test"):
        if split_counts[split] == 0:
            errors.append({"case_id": "", "error": f"empty_split:{split}"})
    unknown_splits = sorted(set(split_counts) - {"train", "val", "test"})
    for split in unknown_splits:
        errors.append({"case_id": "", "error": f"unknown_split:{split}"})

    group_splits: dict[str, set[str]] = {}
    for row in rows:
        case_id = str(row.get("id", ""))
        group_splits.setdefault(patient_group(case_id), set()).add(
            str(row.get("split", ""))
        )
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            errors.append(
                {
                    "case_id": group,
                    "error": f"patient_group_split_leakage:{sorted(splits)}",
                }
            )

    checked = 0
    for row in rows:
        case_id = str(row.get("id", ""))
        case_dir = args.data_dir / case_id
        images = {}
        for field in (*MODALITIES, "seg"):
            filename = str(row.get(field, ""))
            path = case_dir / filename
            if not filename:
                errors.append({"case_id": case_id, "error": f"empty_path:{field}"})
                continue
            if not path.is_file():
                errors.append({"case_id": case_id, "error": f"missing_file:{field}:{path}"})
                continue
            try:
                images[field] = nib.load(str(path))
            except Exception as exc:
                errors.append({"case_id": case_id, "error": f"invalid_nifti:{field}:{exc}"})

        if len(images) != 5:
            continue
        shapes = {field: tuple(image.shape) for field, image in images.items()}
        if len(set(shapes.values())) != 1:
            errors.append({"case_id": case_id, "error": f"shape_mismatch:{shapes}"})

        reference_affine = images["t1n"].affine
        for field in (*MODALITIES[1:], "seg"):
            if not np.allclose(reference_affine, images[field].affine, atol=1e-4):
                errors.append({"case_id": case_id, "error": f"affine_mismatch:{field}"})

        if not args.skip_label_values:
            seg_data = np.asanyarray(images["seg"].dataobj)
            illegal = []
            if np.issubdtype(seg_data.dtype, np.integer):
                min_label = int(seg_data.min())
                max_label = int(seg_data.max())
                if min_label < min(ALLOWED_LABELS) or max_label > max(ALLOWED_LABELS):
                    illegal = sorted(
                        int(value)
                        for value in np.unique(seg_data)
                        if int(value) not in ALLOWED_LABELS
                    )
            else:
                invalid_mask = ~np.isfinite(seg_data) | ~np.isin(seg_data, tuple(ALLOWED_LABELS))
                if invalid_mask.any():
                    illegal = [float(value) for value in np.unique(seg_data[invalid_mask])]
            if illegal:
                errors.append({"case_id": case_id, "error": f"illegal_seg_labels:{illegal}"})
        checked += 1

    summary = {
        "status": "failed" if errors else "passed",
        "data_csv": str(args.data_csv.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "total_rows": len(rows),
        "checked_cases": checked,
        "split_counts": dict(sorted(split_counts.items())),
        "label_values_checked": not args.skip_label_values,
        "error_count": len(errors),
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error['case_id']}: {error['error']}")
        raise RuntimeError(
            f"G1 V2 dataset validation failed with {len(errors)} error(s); see {args.output_json}."
        )


if __name__ == "__main__":
    main()
