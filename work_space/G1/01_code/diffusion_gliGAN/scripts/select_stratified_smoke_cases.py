#!/usr/bin/env python3
"""Freeze a deterministic patient-level Diffusion smoke cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VALID_LABELS = {0, 1, 2, 3, 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    return rows, fieldnames


def filter_lesion_rows(
    lesion_rows: Iterable[dict[str, str]], selected_ids: set[str]
) -> list[dict[str, str]]:
    return [row for row in lesion_rows if str(row["patient_id"]) in selected_ids]


def _assign_burden_strata(features: list[dict[str, object]]) -> None:
    ordered = sorted(
        features, key=lambda row: (float(row["burden_mm3"]), str(row["patient_id"]))
    )
    count = len(ordered)
    for index, row in enumerate(ordered):
        fraction = index / max(count, 1)
        if fraction < 1 / 3:
            row["burden_stratum"] = "low"
        elif fraction < 2 / 3:
            row["burden_stratum"] = "mid"
        else:
            row["burden_stratum"] = "high"


def _selection_counts(selected: list[dict[str, object]]) -> dict[str, int]:
    return {
        "rc": sum(bool(row["has_rc"]) for row in selected),
        "tiny_small": sum(bool(row["has_tiny_small"]) for row in selected),
        "regular": sum(
            not bool(row["has_rc"]) and not bool(row["has_tiny_small"])
            for row in selected
        ),
    }


def select_smoke_cases(
    features: list[dict[str, object]],
    *,
    case_count: int = 20,
    min_rc: int = 8,
    min_tiny_small: int = 8,
    regular_count: int = 4,
) -> list[dict[str, object]]:
    if len(features) < case_count:
        raise ValueError(
            f"Not enough eligible cases: eligible={len(features)} requested={case_count}"
        )
    if len({str(row["patient_id"]) for row in features}) != len(features):
        raise ValueError("Duplicate patient_id values in case features")
    if case_count < regular_count:
        raise ValueError("case_count must be at least regular_count")

    candidates = [dict(row) for row in features]
    _assign_burden_strata(candidates)
    global_median = statistics.median(float(row["burden_mm3"]) for row in candidates)
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selected_groups: set[str] = set()
    selected_strata: set[str] = set()

    def add(row: dict[str, object], reason: str) -> None:
        patient_id = str(row["patient_id"])
        if patient_id in selected_ids:
            return
        chosen = dict(row)
        reasons = list(chosen.get("selection_reasons", []))
        reasons.append(reason)
        chosen["selection_reasons"] = reasons
        selected.append(chosen)
        selected_ids.add(patient_id)
        selected_groups.add(str(chosen.get("patient_group", "")))
        selected_strata.add(str(chosen["burden_stratum"]))

    regular_candidates = [
        row
        for row in candidates
        if not bool(row["has_rc"]) and not bool(row["has_tiny_small"])
    ]
    while len(selected) < regular_count and regular_candidates:
        regular_candidates.sort(
            key=lambda row: (
                str(row.get("patient_group", "")) in selected_groups,
                abs(float(row["burden_mm3"]) - global_median),
                str(row["patient_id"]),
            )
        )
        add(regular_candidates.pop(0), "regular_median")
    if _selection_counts(selected)["regular"] < regular_count:
        raise ValueError(
            f"Insufficient regular cases: required={regular_count} "
            f"available={len(regular_candidates) + _selection_counts(selected)['regular']}"
        )

    while True:
        counts = _selection_counts(selected)
        rc_deficit = max(0, min_rc - counts["rc"])
        small_deficit = max(0, min_tiny_small - counts["tiny_small"])
        if rc_deficit == 0 and small_deficit == 0:
            break
        if len(selected) >= case_count:
            raise ValueError("Risk quotas cannot fit within requested case_count")
        risk_candidates = [
            row
            for row in candidates
            if str(row["patient_id"]) not in selected_ids
            and (
                (rc_deficit > 0 and bool(row["has_rc"]))
                or (small_deficit > 0 and bool(row["has_tiny_small"]))
            )
        ]
        if not risk_candidates:
            raise ValueError(
                f"Risk quota unavailable: rc_deficit={rc_deficit} "
                f"tiny_small_deficit={small_deficit}"
            )
        risk_candidates.sort(
            key=lambda row: (
                -int(rc_deficit > 0 and bool(row["has_rc"])),
                -int(small_deficit > 0 and bool(row["has_tiny_small"])),
                str(row["burden_stratum"]) in selected_strata,
                str(row.get("patient_group", "")) in selected_groups,
                -int(row.get("rc_tiny_count", 0)),
                -int(row.get("tiny_count", 0)),
                str(row["patient_id"]),
            )
        )
        add(risk_candidates[0], "risk_quota")

    while len(selected) < case_count:
        remaining = [
            row
            for row in candidates
            if str(row["patient_id"]) not in selected_ids
        ]
        remaining.sort(
            key=lambda row: (
                str(row["burden_stratum"]) in selected_strata,
                str(row.get("patient_group", "")) in selected_groups,
                abs(float(row["burden_mm3"]) - global_median),
                str(row["patient_id"]),
            )
        )
        add(remaining[0], "diversity_fill")

    counts = _selection_counts(selected)
    if counts["rc"] < min_rc or counts["tiny_small"] < min_tiny_small:
        raise AssertionError(f"Risk quotas not met after selection: {counts}")
    if counts["regular"] < regular_count:
        raise AssertionError(f"Regular quota not met after selection: {counts}")
    if {str(row["burden_stratum"]) for row in selected} != {"low", "mid", "high"}:
        raise ValueError("Selected cohort does not cover low/mid/high burden strata")

    for row in selected:
        reasons = list(row.get("selection_reasons", []))
        if bool(row["has_rc"]):
            reasons.append("rc")
        if bool(row["has_tiny_small"]):
            reasons.append("tiny_small")
        reasons.append(f"burden_{row['burden_stratum']}")
        row["selection_reasons"] = sorted(set(reasons))
    return selected


def summarize_label(patient_row: dict[str, str], lesion_count_csv: int) -> dict[str, object]:
    import nibabel as nib
    import numpy as np
    from scipy import ndimage

    label_path = Path(patient_row["seg_path"])
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing segmentation: {label_path}")
    image = nib.load(str(label_path))
    label = np.rint(image.get_fdata()).astype(np.int16)
    unique_labels = {int(value) for value in np.unique(label)}
    illegal = sorted(unique_labels - VALID_LABELS)
    if illegal:
        raise ValueError(f"Illegal labels for {patient_row['source_case_id']}: {illegal}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    voxel_volume = math.prod(spacing)
    structure = ndimage.generate_binary_structure(3, 3)

    lesion_labels, lesion_count = ndimage.label(label > 0, structure=structure)
    lesion_voxels = np.bincount(lesion_labels.ravel())[1:]
    lesion_volumes = [float(value) * voxel_volume for value in lesion_voxels]

    rc_labels, _ = ndimage.label(label == 4, structure=structure)
    rc_voxels = np.bincount(rc_labels.ravel())[1:]
    rc_volumes = [float(value) * voxel_volume for value in rc_voxels]
    return {
        "patient_id": str(patient_row["source_case_id"]).removeprefix("BraTS-MET-"),
        "source_case_id": str(patient_row["source_case_id"]),
        "patient_group": str(patient_row.get("patient_group", "")),
        "seg_path": str(label_path),
        "spacing": list(spacing),
        "voxel_volume_mm3": voxel_volume,
        "burden_mm3": float(np.count_nonzero(label)) * voxel_volume,
        "connected_lesion_count": int(lesion_count),
        "lesion_count_csv": int(lesion_count_csv),
        "tiny_count": sum(volume < 27.0 for volume in lesion_volumes),
        "small_count": sum(27.0 <= volume <= 275.0 for volume in lesion_volumes),
        "large_count": sum(volume > 275.0 for volume in lesion_volumes),
        "has_tiny_small": any(volume <= 275.0 for volume in lesion_volumes),
        "has_rc": bool(np.any(label == 4)),
        "rc_voxels": int(np.count_nonzero(label == 4)),
        "rc_tiny_count": sum(volume < 27.0 for volume in rc_volumes),
        "rc_small_count": sum(27.0 <= volume <= 275.0 for volume in rc_volumes),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a stratified patient-level Diffusion smoke cohort."
    )
    parser.add_argument("--lesions-csv", required=True, type=Path)
    parser.add_argument("--membership-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--selection-json", required=True, type=Path)
    parser.add_argument("--selection-csv", required=True, type=Path)
    parser.add_argument("--expected-val-count", type=int, default=103)
    parser.add_argument("--case-count", type=int, default=20)
    parser.add_argument("--min-rc", type=int, default=8)
    parser.add_argument("--min-tiny-small", type=int, default=8)
    parser.add_argument("--regular-count", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_paths = (args.output_csv, args.selection_json, args.selection_csv)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite frozen smoke outputs: {existing}")

    lesion_rows, lesion_fields = read_csv_rows(args.lesions_csv)
    membership_rows, _ = read_csv_rows(args.membership_csv)
    val_lesions = [row for row in lesion_rows if row.get("split") == "val"]
    val_membership = [row for row in membership_rows if row.get("split") == "val"]
    if len(val_membership) != args.expected_val_count:
        raise ValueError(
            f"Fixed validation count mismatch: expected={args.expected_val_count} "
            f"actual={len(val_membership)}"
        )

    lesion_counts = Counter(str(row["patient_id"]) for row in val_lesions)
    membership_by_id = {
        str(row["source_case_id"]).removeprefix("BraTS-MET-"): row
        for row in val_membership
    }
    if len(membership_by_id) != len(val_membership):
        raise ValueError("Duplicate source_case_id values in validation membership")

    unknown_lesion_ids = sorted(set(lesion_counts) - set(membership_by_id))
    if unknown_lesion_ids:
        raise ValueError(f"Val lesion rows outside fixed validation: {unknown_lesion_ids}")

    features: list[dict[str, object]] = []
    negative_cases: list[str] = []
    for patient_id, membership_row in sorted(membership_by_id.items()):
        summary = summarize_label(membership_row, lesion_counts.get(patient_id, 0))
        if float(summary["burden_mm3"]) <= 0:
            if patient_id in lesion_counts:
                raise ValueError(f"Zero-label case unexpectedly has lesion rows: {patient_id}")
            negative_cases.append(str(membership_row["source_case_id"]))
            continue
        if patient_id not in lesion_counts:
            raise ValueError(f"Positive-label case is missing from lesions CSV: {patient_id}")
        features.append(summary)

    selected = select_smoke_cases(
        features,
        case_count=args.case_count,
        min_rc=args.min_rc,
        min_tiny_small=args.min_tiny_small,
        regular_count=args.regular_count,
    )
    selected_ids = {str(row["patient_id"]) for row in selected}
    smoke_rows = filter_lesion_rows(val_lesions, selected_ids)
    represented_ids = {str(row["patient_id"]) for row in smoke_rows}
    if represented_ids != selected_ids:
        raise AssertionError("Frozen smoke CSV does not represent every selected patient")

    feature_fields = [
        "patient_id",
        "source_case_id",
        "patient_group",
        "burden_stratum",
        "burden_mm3",
        "connected_lesion_count",
        "lesion_count_csv",
        "tiny_count",
        "small_count",
        "large_count",
        "has_tiny_small",
        "has_rc",
        "rc_voxels",
        "rc_tiny_count",
        "rc_small_count",
        "voxel_volume_mm3",
        "spacing",
        "selection_reasons",
        "seg_path",
    ]
    serializable_selected = []
    for row in selected:
        serialized = dict(row)
        serialized["spacing"] = json.dumps(row["spacing"], separators=(",", ":"))
        serialized["selection_reasons"] = ";".join(row["selection_reasons"])
        serializable_selected.append(serialized)

    write_csv(args.output_csv, lesion_fields, smoke_rows)
    write_csv(args.selection_csv, feature_fields, serializable_selected)
    counts = _selection_counts(selected)
    summary = {
        "schema_version": 1,
        "status": "frozen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "fixed_103_val",
        "fixed_val_count": len(val_membership),
        "lesion_positive_val_count": len(features),
        "lesion_negative_val_count": len(negative_cases),
        "lesion_negative_source_case_ids": negative_cases,
        "smoke_case_count": len(selected),
        "smoke_lesion_row_count": len(smoke_rows),
        "requirements": {
            "min_rc": args.min_rc,
            "min_tiny_small": args.min_tiny_small,
            "regular_count": args.regular_count,
            "burden_strata": ["low", "mid", "high"],
        },
        "observed": {
            "rc": counts["rc"],
            "tiny_small": counts["tiny_small"],
            "regular": counts["regular"],
            "burden_strata": sorted(
                {str(row["burden_stratum"]) for row in selected}
            ),
            "patient_groups": sorted(
                {str(row.get("patient_group", "")) for row in selected}
            ),
        },
        "selected_source_case_ids": [str(row["source_case_id"]) for row in selected],
        "source_files": {
            "lesions_csv": str(args.lesions_csv.resolve()),
            "lesions_csv_sha256": sha256_file(args.lesions_csv),
            "membership_csv": str(args.membership_csv.resolve()),
            "membership_csv_sha256": sha256_file(args.membership_csv),
        },
        "outputs": {
            "smoke_csv": str(args.output_csv.resolve()),
            "selection_csv": str(args.selection_csv.resolve()),
        },
    }
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    args.selection_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
