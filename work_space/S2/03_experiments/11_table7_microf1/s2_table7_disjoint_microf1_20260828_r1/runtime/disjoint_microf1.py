#!/usr/bin/env python3
"""Compute mutually exclusive size-stratified lesion-instance micro-F1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


REGIONS: dict[str, tuple[int, ...]] = {
    "et": (3,),
    "rc": (4,),
    "tc": (1, 3),
    "wt": (1, 2, 3),
}
ALLOWED_LABELS = {0, 1, 2, 3, 4}
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def micro_f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return (2.0 * tp / denominator) if denominator else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nifti_id(path: Path) -> str:
    require(path.name.endswith(".nii.gz"), f"not a NIfTI filename: {path}")
    return path.name[: -len(".nii.gz")]


def discover(root: Path) -> dict[str, Path]:
    require(root.is_dir(), f"missing directory: {root}")
    paths = sorted(root.glob("*.nii.gz"))
    result = {nifti_id(path): path for path in paths}
    require(len(paths) == len(result), f"duplicate NIfTI IDs in {root}")
    for path in paths:
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}")
    return result


def read_labels(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    require(data.ndim == 3, f"expected a 3D label map: {path} shape={data.shape}")
    require(np.isfinite(data).all(), f"nonfinite label values: {path}")
    rounded = np.rint(data)
    require(np.array_equal(data, rounded), f"non-integer label values: {path}")
    labels = rounded.astype(np.uint8, copy=False)
    values = {int(value) for value in np.unique(labels)}
    require(values <= ALLOWED_LABELS, f"unexpected labels in {path}: {sorted(values)}")
    return image, labels


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labeled, count = ndimage.label(mask, structure=CONNECTIVITY)
    sizes = np.bincount(labeled.ravel(), minlength=count + 1)[1:].astype(np.int64)
    return labeled, sizes


def dice_matrix(
    reference_components: np.ndarray,
    prediction_components: np.ndarray,
    reference_sizes: np.ndarray,
    prediction_sizes: np.ndarray,
) -> np.ndarray:
    n_reference = len(reference_sizes)
    n_prediction = len(prediction_sizes)
    result = np.zeros((n_reference, n_prediction), dtype=np.float64)
    if n_reference == 0 or n_prediction == 0:
        return result
    overlap = (reference_components > 0) & (prediction_components > 0)
    if not np.any(overlap):
        return result
    encoded = (
        reference_components[overlap].astype(np.int64) * (n_prediction + 1)
        + prediction_components[overlap].astype(np.int64)
    )
    keys, intersections = np.unique(encoded, return_counts=True)
    reference_index = keys // (n_prediction + 1) - 1
    prediction_index = keys % (n_prediction + 1) - 1
    valid = (reference_index >= 0) & (prediction_index >= 0)
    reference_index = reference_index[valid]
    prediction_index = prediction_index[valid]
    intersections = intersections[valid]
    result[reference_index, prediction_index] = (
        2.0
        * intersections
        / (reference_sizes[reference_index] + prediction_sizes[prediction_index])
    )
    return result


def match_components(
    reference_components: np.ndarray,
    prediction_components: np.ndarray,
    reference_sizes: np.ndarray,
    prediction_sizes: np.ndarray,
    overlap_threshold: float,
) -> list[tuple[int, int, float]]:
    """Return maximum-cardinality matches, with total DSC as tie-breaker."""
    n_reference = len(reference_sizes)
    n_prediction = len(prediction_sizes)
    if n_reference == 0 or n_prediction == 0:
        return []
    require(
        n_reference * n_prediction <= 25_000_000,
        "component matching matrix exceeds the audited 25M-cell safety bound",
    )
    dsc = dice_matrix(
        reference_components,
        prediction_components,
        reference_sizes,
        prediction_sizes,
    )
    qualifying = dsc >= overlap_threshold
    if not np.any(qualifying):
        return []
    cardinality_weight = float(min(n_reference, n_prediction) + 1)
    reward = np.where(qualifying, cardinality_weight + dsc, 0.0)
    rows, columns = linear_sum_assignment(reward, maximize=True)
    matches = [
        (int(row), int(column), float(dsc[row, column]))
        for row, column in zip(rows, columns)
        if qualifying[row, column]
    ]
    require(len({row for row, _, _ in matches}) == len(matches), "reference reused")
    require(len({column for _, column, _ in matches}) == len(matches), "prediction reused")
    return sorted(matches)


def empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def add_count(counts: dict[str, int], event: str) -> None:
    counts[event] += 1


def with_f1(counts: dict[str, int]) -> dict[str, int | float | None]:
    return {**counts, "micro_f1": micro_f1(counts["tp"], counts["fp"], counts["fn"])}


def evaluate_region(
    reference: np.ndarray,
    prediction: np.ndarray,
    labels: Iterable[int],
    overlap_threshold: float,
) -> dict[str, object]:
    reference_components, reference_sizes = component_labels(np.isin(reference, tuple(labels)))
    prediction_components, prediction_sizes = component_labels(np.isin(prediction, tuple(labels)))
    matches = match_components(
        reference_components,
        prediction_components,
        reference_sizes,
        prediction_sizes,
        overlap_threshold,
    )
    matched_reference = {row for row, _, _ in matches}
    matched_prediction = {column for _, column, _ in matches}
    return {
        "reference_sizes": reference_sizes,
        "prediction_sizes": prediction_sizes,
        "matches": matches,
        "unmatched_reference": sorted(set(range(len(reference_sizes))) - matched_reference),
        "unmatched_prediction": sorted(set(range(len(prediction_sizes))) - matched_prediction),
    }


def classify_counts(evaluation: dict[str, object], threshold: int) -> dict[str, dict[str, int]]:
    counts = {"all": empty_counts(), "small": empty_counts(), "large": empty_counts()}
    reference_sizes = evaluation["reference_sizes"]
    prediction_sizes = evaluation["prediction_sizes"]
    assert isinstance(reference_sizes, np.ndarray)
    assert isinstance(prediction_sizes, np.ndarray)

    for reference_index, _, _ in evaluation["matches"]:
        bucket = "large" if int(reference_sizes[reference_index]) >= threshold else "small"
        add_count(counts["all"], "tp")
        add_count(counts[bucket], "tp")
    for reference_index in evaluation["unmatched_reference"]:
        bucket = "large" if int(reference_sizes[reference_index]) >= threshold else "small"
        add_count(counts["all"], "fn")
        add_count(counts[bucket], "fn")
    for prediction_index in evaluation["unmatched_prediction"]:
        bucket = "large" if int(prediction_sizes[prediction_index]) >= threshold else "small"
        add_count(counts["all"], "fp")
        add_count(counts[bucket], "fp")
    for event in ("tp", "fp", "fn"):
        require(
            counts["all"][event] == counts["small"][event] + counts["large"][event],
            f"non-additive {event} at threshold {threshold}",
        )
    return counts


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--overlap-threshold", default=0.2, type=float)
    parser.add_argument("--thresholds", default="27,50")
    args = parser.parse_args()

    thresholds = sorted({int(item) for item in args.thresholds.split(",")})
    require(thresholds == [27, 50], f"threshold contract drift: {thresholds}")
    require(args.overlap_threshold == 0.2, "overlap threshold drift")
    prediction_paths = discover(args.prediction_root.resolve())
    reference_paths = discover(args.reference_root.resolve())
    require(len(reference_paths) == args.expected_count, "reference count drift")
    require(set(prediction_paths) == set(reference_paths), "prediction/reference ID mismatch")
    output_root = args.output_root.resolve()
    require(not output_root.exists(), f"exclusive output exists: {output_root}")
    output_root.mkdir(parents=True)

    case_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    pooled: dict[tuple[int, str, str], dict[str, int]] = defaultdict(empty_counts)
    region_matching_totals = {
        region: {"reference_components": 0, "prediction_components": 0, "accepted_matches": 0}
        for region in REGIONS
    }

    for case_id in sorted(reference_paths):
        reference_image, reference = read_labels(reference_paths[case_id])
        prediction_image, prediction = read_labels(prediction_paths[case_id])
        require(reference.shape == prediction.shape, f"shape mismatch: {case_id}")
        require(
            np.allclose(reference_image.affine, prediction_image.affine, rtol=0, atol=1e-5),
            f"affine mismatch: {case_id}",
        )
        for region, labels in REGIONS.items():
            evaluation = evaluate_region(reference, prediction, labels, args.overlap_threshold)
            reference_sizes = evaluation["reference_sizes"]
            prediction_sizes = evaluation["prediction_sizes"]
            assert isinstance(reference_sizes, np.ndarray)
            assert isinstance(prediction_sizes, np.ndarray)
            matches = evaluation["matches"]
            totals = region_matching_totals[region]
            totals["reference_components"] += len(reference_sizes)
            totals["prediction_components"] += len(prediction_sizes)
            totals["accepted_matches"] += len(matches)

            for reference_index, prediction_index, dsc in matches:
                event_rows.append({
                    "case_id": case_id,
                    "region": region,
                    "outcome": "tp",
                    "reference_component": reference_index + 1,
                    "prediction_component": prediction_index + 1,
                    "reference_voxels": int(reference_sizes[reference_index]),
                    "prediction_voxels": int(prediction_sizes[prediction_index]),
                    "dsc": f"{dsc:.17g}",
                })
            for reference_index in evaluation["unmatched_reference"]:
                event_rows.append({
                    "case_id": case_id,
                    "region": region,
                    "outcome": "fn",
                    "reference_component": reference_index + 1,
                    "prediction_component": "",
                    "reference_voxels": int(reference_sizes[reference_index]),
                    "prediction_voxels": "",
                    "dsc": "",
                })
            for prediction_index in evaluation["unmatched_prediction"]:
                event_rows.append({
                    "case_id": case_id,
                    "region": region,
                    "outcome": "fp",
                    "reference_component": "",
                    "prediction_component": prediction_index + 1,
                    "reference_voxels": "",
                    "prediction_voxels": int(prediction_sizes[prediction_index]),
                    "dsc": "",
                })

            for threshold in thresholds:
                counts = classify_counts(evaluation, threshold)
                row: dict[str, object] = {
                    "case_id": case_id,
                    "region": region,
                    "size_threshold_voxels": threshold,
                    "reference_components": len(reference_sizes),
                    "prediction_components": len(prediction_sizes),
                    "accepted_matches": len(matches),
                }
                for group in ("all", "small", "large"):
                    for event in ("tp", "fp", "fn"):
                        value = counts[group][event]
                        row[f"{group}_{event}"] = value
                        pooled[(threshold, region, group)][event] += value
                    value = micro_f1(
                        counts[group]["tp"], counts[group]["fp"], counts[group]["fn"]
                    )
                    row[f"{group}_f1"] = "" if value is None else f"{value:.17g}"
                case_rows.append(row)

    case_fields = [
        "case_id", "region", "size_threshold_voxels", "reference_components",
        "prediction_components", "accepted_matches",
    ] + [f"{group}_{metric}" for group in ("all", "small", "large") for metric in ("tp", "fp", "fn", "f1")]
    event_fields = [
        "case_id", "region", "outcome", "reference_component", "prediction_component",
        "reference_voxels", "prediction_voxels", "dsc",
    ]
    case_path = output_root / "case_level_disjoint_counts.csv"
    event_path = output_root / "component_events.csv"
    write_csv(case_path, case_rows, case_fields)
    write_csv(event_path, event_rows, event_fields)

    regions_payload: dict[str, object] = {}
    summary_rows: list[dict[str, object]] = []
    for region in REGIONS:
        threshold_payload: dict[str, object] = {}
        for threshold in thresholds:
            groups: dict[str, object] = {}
            for group in ("all", "small", "large"):
                counts = pooled[(threshold, region, group)]
                payload = with_f1(counts)
                groups[group] = payload
                summary_rows.append({
                    "region": region,
                    "size_threshold_voxels": threshold,
                    "group": group,
                    **payload,
                })
            for event in ("tp", "fp", "fn"):
                require(
                    groups["all"][event] == groups["small"][event] + groups["large"][event],
                    f"pooled non-additive {region}/{threshold}/{event}",
                )
            threshold_payload[str(threshold)] = {"groups": groups, "additivity_pass": True}
        regions_payload[region] = {
            "matching_totals": region_matching_totals[region],
            "thresholds": threshold_payload,
        }
    summary_csv = output_root / "disjoint_microf1_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        ["region", "size_threshold_voxels", "group", "tp", "fp", "fn", "micro_f1"],
    )
    case_ids = sorted(reference_paths)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(case_ids),
        "case_ids_sha256": hashlib.sha256(("\n".join(case_ids) + "\n").encode()).hexdigest(),
        "contract": {
            "regions": {key: list(value) for key, value in REGIONS.items()},
            "connected_components": "3D 26-connectivity",
            "matching": "maximum-cardinality one-to-one among DSC>=0.2; maximum-total-DSC tie-break",
            "overlap_threshold_dsc": args.overlap_threshold,
            "thresholds_voxels": thresholds,
            "size_rule": "small < T; large >= T",
            "tp_size_source": "reference",
            "fn_size_source": "reference",
            "fp_size_source": "prediction",
            "aggregation": "global micro pooling across all cases",
            "f1_formula": "2TP/(2TP+FP+FN)",
        },
        "regions": regions_payload,
        "artifacts": {
            "case_level_disjoint_counts.csv": sha256_file(case_path),
            "component_events.csv": sha256_file(event_path),
            "disjoint_microf1_summary.csv": sha256_file(summary_csv),
        },
    }
    require(all(math.isfinite(row["micro_f1"]) for row in summary_rows if row["micro_f1"] is not None), "nonfinite F1")
    summary_json = output_root / "DISJOINT_MICROF1_SUMMARY.json"
    with summary_json.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

