#!/usr/bin/env python3
"""Remove lesion rows that can produce an all-zero MRI crop in any modality."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITY_COLUMNS = ("scan_t1c", "scan_t1n", "scan_t2w", "scan_t2f")
AXES = (
    ("x_extreme_min", "x_extreme_max"),
    ("y_extreme_min", "y_extreme_max"),
    ("z_extreme_min", "z_extreme_max"),
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def allowed_offsets(lower: int, upper: int, limit: int, target_size: int) -> range:
    size = upper - lower
    margin = min(8, max(0, (target_size - size) // 2 - 2))
    minimum = max(-margin, -lower)
    maximum = min(margin, limit - upper)
    return range(minimum, maximum + 1)


def crop_bounds(lower: int, upper: int, limit: int, offset: int, target_size: int) -> tuple[int, int]:
    size = upper - lower
    pad = (target_size - size) / 2
    correction = -0.5 if pad < 0 else 0.5
    start = lower + offset - int(pad)
    stop = upper + offset + int(pad + correction)
    return max(0, start), min(limit, stop)


def integral_mask(volume: np.ndarray) -> np.ndarray:
    mask = np.asarray(volume != 0, dtype=np.int32)
    integral = mask.cumsum(0).cumsum(1).cumsum(2)
    return np.pad(integral, ((1, 0), (1, 0), (1, 0)))


def box_sum(
    integral: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    z0: np.ndarray,
    z1: np.ndarray,
) -> np.ndarray:
    return (
        integral[x1, y1, z1]
        - integral[x0, y1, z1]
        - integral[x1, y0, z1]
        - integral[x1, y1, z0]
        + integral[x0, y0, z1]
        + integral[x0, y1, z0]
        + integral[x1, y0, z0]
        - integral[x0, y0, z0]
    )


def minimum_possible_crop_nonzero(
    volume: np.ndarray,
    row: dict[str, str],
    target_size: int,
    integral: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    bounds = [(int(row[lower]), int(row[upper])) for lower, upper in AXES]
    limits = volume.shape
    offsets = [allowed_offsets(lo, hi, limit, target_size) for (lo, hi), limit in zip(bounds, limits)]
    axis_bounds = [
        [crop_bounds(lo, hi, limit, offset, target_size) for offset in axis_offsets]
        for (lo, hi), limit, axis_offsets in zip(bounds, limits, offsets)
    ]
    if integral is None:
        integral = integral_mask(volume)
    x0 = np.asarray([value[0] for value in axis_bounds[0]])[:, None, None]
    x1 = np.asarray([value[1] for value in axis_bounds[0]])[:, None, None]
    y0 = np.asarray([value[0] for value in axis_bounds[1]])[None, :, None]
    y1 = np.asarray([value[1] for value in axis_bounds[1]])[None, :, None]
    z0 = np.asarray([value[0] for value in axis_bounds[2]])[None, None, :]
    z1 = np.asarray([value[1] for value in axis_bounds[2]])[None, None, :]
    counts = box_sum(integral, x0, x1, y0, y1, z0, z1)
    return int(counts.min()), integral


def validate_group(
    rows: list[dict[str, str]],
    target_size: int,
) -> dict[int, dict[str, object]]:
    rejected: dict[int, dict[str, object]] = {}
    volumes: dict[str, np.ndarray] = {}
    affines: dict[str, np.ndarray] = {}
    for column in MODALITY_COLUMNS:
        path = Path(rows[0][column])
        image = nib.load(str(path))
        volume = np.asanyarray(image.dataobj)
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError(f"Invalid volume for {column}: {path}")
        volumes[column] = volume
        affines[column] = np.asarray(image.affine)
    shapes = {volume.shape for volume in volumes.values()}
    if len(shapes) != 1:
        raise ValueError(f"Modality shapes differ for {rows[0]['patient_id']}: {shapes}")
    reference_affine = affines[MODALITY_COLUMNS[0]]
    if any(not np.allclose(affine, reference_affine, atol=1e-4, rtol=0) for affine in affines.values()):
        raise ValueError(f"Modality affines differ for {rows[0]['patient_id']}")

    integral_cache: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        bbox = tuple(slice(int(row[lower]), int(row[upper])) for lower, upper in AXES)
        invalid_modalities: list[str] = []
        minimum_counts: dict[str, int] = {}
        for column, volume in volumes.items():
            if np.any(volume[bbox] != 0):
                continue
            minimum, integral = minimum_possible_crop_nonzero(
                volume, row, target_size, integral_cache.get(column))
            integral_cache[column] = integral
            minimum_counts[column.removeprefix("scan_")] = minimum
            if minimum == 0:
                invalid_modalities.append(column.removeprefix("scan_"))
        if invalid_modalities:
            rejected[index] = {
                "patient_id": row["patient_id"],
                "patient_group": row["patient_group"],
                "lesion_id": row["lesion_id"],
                "split": row["split"],
                "reason": "possible_all_zero_crop",
                "modalities": ";".join(invalid_modalities),
                "minimum_nonzero_voxels": json.dumps(minimum_counts, sort_keys=True),
            }
    return rejected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesions", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=64)
    args = parser.parse_args()

    fieldnames, rows = read_csv(args.lesions)
    required = {"patient_id", "patient_group", "lesion_id", "split", "patient_n_crops", *MODALITY_COLUMNS}
    required.update(value for pair in AXES for value in pair)
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Lesion CSV is missing columns: {missing}")

    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = tuple(row[column] for column in MODALITY_COLUMNS)
        grouped[key].append((index, row))

    rejected_by_global_index: dict[int, dict[str, object]] = {}
    for grouped_rows in grouped.values():
        local_rows = [row for _, row in grouped_rows]
        local_rejected = validate_group(local_rows, args.target_size)
        for local_index, audit_row in local_rejected.items():
            rejected_by_global_index[grouped_rows[local_index][0]] = audit_row

    kept = [row for index, row in enumerate(rows) if index not in rejected_by_global_index]
    counts: dict[str, int] = defaultdict(int)
    balance_column = "patient_group" if "patient_group" in fieldnames else "patient_id"
    for row in kept:
        counts[row[balance_column]] += 1
    for row in kept:
        row["patient_n_crops"] = str(counts[row[balance_column]])

    args.backup.parent.mkdir(parents=True, exist_ok=True)
    if not args.backup.exists():
        shutil.copy2(args.lesions, args.backup)
    write_csv_atomic(args.lesions, fieldnames, kept)
    audit_rows = [rejected_by_global_index[index] for index in sorted(rejected_by_global_index)]
    write_csv_atomic(
        args.audit,
        ["patient_id", "patient_group", "lesion_id", "split", "reason", "modalities", "minimum_nonzero_voxels"],
        audit_rows,
    )
    summary = {
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "rejected_rows": len(audit_rows),
        "target_size": args.target_size,
        "backup": str(args.backup),
        "audit": str(args.audit),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "LESION_SCAN_CONTENT_QC_PASS "
        f"input={len(rows)} kept={len(kept)} rejected={len(audit_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
