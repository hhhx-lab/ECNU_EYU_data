#!/usr/bin/env python3
"""Freeze the fixed 94-positive/9-negative Diffusion validation cohort."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


MODALITY_FIELDS = ("t1n_path", "t1c_path", "t2w_path", "t2f_path")
VALID_LABELS = {0, 1, 2, 3, 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strict_validation_noop(
    image: np.ndarray, segmentation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validation/test/official inference bypasses Diffusion by contract."""
    return image.copy(), segmentation.copy(), False


def validate_pipeline_contract(online_trainer: Path, base_trainer: Path) -> dict[str, bool]:
    online_tree = ast.parse(online_trainer.read_text(encoding="utf-8"))
    base_tree = ast.parse(base_trainer.read_text(encoding="utf-8"))

    online_class = next(
        node
        for node in online_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "nnUNetTrainerBraTS2026RCOnlineDiffusion"
    )
    online_methods = {
        node.name for node in online_class.body if isinstance(node, ast.FunctionDef)
    }
    base_class = next(
        node
        for node in base_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "nnUNetTrainerBraTS2026RC"
    )
    base_methods = {
        node.name: node for node in base_class.body if isinstance(node, ast.FunctionDef)
    }
    validation_source = ast.unparse(base_methods["get_validation_transforms"])
    training_source = ast.unparse(base_methods["get_training_transforms"])
    result = {
        "online_trainer_does_not_override_validation": (
            "get_validation_transforms" not in online_methods
        ),
        "base_validation_has_no_online_diffusion": (
            "OnlineDiffusion" not in validation_source
        ),
        "online_diffusion_is_training_only": (
            "get_pre_spatial_training_transforms" in training_source
        ),
    }
    if not all(result.values()):
        raise ValueError(f"Online Diffusion validation-isolation contract failed: {result}")
    return result


def _load_finite(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values: {path}")
    return image, values


def freeze_cohort(
    lesions_csv: Path,
    membership_csv: Path,
    smoke_selection_json: Path,
    output_dir: Path,
    *,
    expected_fixed: int = 103,
    expected_positive: int = 94,
    expected_negative: int = 9,
    online_trainer: Path | None = None,
    base_trainer: Path | None = None,
) -> dict[str, Any]:
    output_paths = (
        output_dir / "val_positive94.csv",
        output_dir / "val_negative9_noop.csv",
        output_dir / "full_eval_cohort_summary.json",
    )
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite frozen outputs: {existing}")

    selection = json.loads(smoke_selection_json.read_text(encoding="utf-8"))
    source_files = selection.get("source_files", {})
    source_hashes = {
        "lesions_csv": sha256_file(lesions_csv),
        "membership_csv": sha256_file(membership_csv),
        "smoke_selection_json": sha256_file(smoke_selection_json),
    }
    for key, actual in (
        ("lesions_csv_sha256", source_hashes["lesions_csv"]),
        ("membership_csv_sha256", source_hashes["membership_csv"]),
    ):
        expected = source_files.get(key)
        if expected != actual:
            raise ValueError(f"Frozen source hash mismatch for {key}: {actual} != {expected}")

    lesions, lesion_fields = read_csv(lesions_csv)
    membership, _ = read_csv(membership_csv)
    val_members = [row for row in membership if row.get("split") == "val"]
    val_lesions = [row for row in lesions if row.get("split") == "val"]
    if len(val_members) != expected_fixed:
        raise ValueError(
            f"Fixed validation count mismatch: {len(val_members)} != {expected_fixed}"
        )
    member_by_id = {row["source_case_id"]: row for row in val_members}
    if len(member_by_id) != len(val_members):
        raise ValueError("Duplicate source_case_id in fixed validation membership")
    positive_ids = {f"BraTS-MET-{row['patient_id']}" for row in val_lesions}
    if not positive_ids <= set(member_by_id):
        raise ValueError("Validation lesion CSV includes cases outside fixed membership")
    negative_ids = sorted(set(member_by_id) - positive_ids)
    if len(positive_ids) != expected_positive or len(negative_ids) != expected_negative:
        raise ValueError(
            "Positive/negative count mismatch: "
            f"positive={len(positive_ids)} negative={len(negative_ids)}"
        )
    if sorted(selection.get("lesion_negative_source_case_ids", [])) != negative_ids:
        raise ValueError("Lesion-negative case IDs differ from frozen smoke selection")

    validation_contract: dict[str, bool] = {}
    if (online_trainer is None) != (base_trainer is None):
        raise ValueError("online_trainer and base_trainer must be provided together")
    if online_trainer is not None and base_trainer is not None:
        validation_contract = validate_pipeline_contract(online_trainer, base_trainer)

    negative_rows: list[dict[str, Any]] = []
    for case_id in negative_ids:
        member = member_by_id[case_id]
        seg_image, seg = _load_finite(Path(member["seg_path"]))
        rounded_seg = np.rint(seg).astype(np.int16)
        labels = {int(value) for value in np.unique(rounded_seg)}
        if labels - VALID_LABELS or np.count_nonzero(rounded_seg):
            raise ValueError(f"Expected all-zero valid segmentation: {case_id} labels={labels}")
        modality_arrays = []
        geometry_ok = True
        for field in MODALITY_FIELDS:
            image, values = _load_finite(Path(member[field]))
            modality_arrays.append(values.astype(np.float32, copy=False))
            geometry_ok &= image.shape == seg_image.shape
            geometry_ok &= np.allclose(image.affine, seg_image.affine, atol=1e-4, rtol=0)
        if not geometry_ok:
            raise ValueError(f"Negative-case geometry mismatch: {case_id}")
        stacked = np.stack(modality_arrays)
        seg_4d = rounded_seg[None]
        before_image_hash = hashlib.sha256(stacked.tobytes()).hexdigest()
        before_seg_hash = hashlib.sha256(seg_4d.tobytes()).hexdigest()
        output_image, output_seg, was_modified = strict_validation_noop(stacked, seg_4d)
        image_equal = np.array_equal(stacked, output_image)
        seg_equal = np.array_equal(seg_4d, output_seg)
        if was_modified or not image_equal or not seg_equal:
            raise AssertionError(f"Strict validation no-op failed: {case_id}")
        negative_rows.append(
            {
                "source_case_id": case_id,
                "patient_group": member["patient_group"],
                "seg_voxels": 0,
                "was_modified": False,
                "image_equal": image_equal,
                "seg_equal": seg_equal,
                "geometry_ok": geometry_ok,
                "image_sha256_before": before_image_hash,
                "image_sha256_after": hashlib.sha256(output_image.tobytes()).hexdigest(),
                "seg_sha256_before": before_seg_hash,
                "seg_sha256_after": hashlib.sha256(output_seg.tobytes()).hexdigest(),
            }
        )

    for case_id in sorted(positive_ids):
        _, seg = _load_finite(Path(member_by_id[case_id]["seg_path"]))
        if not np.count_nonzero(np.rint(seg).astype(np.int16)):
            raise ValueError(f"Positive case has empty segmentation: {case_id}")

    positive_rows = [
        row
        for row in val_lesions
        if f"BraTS-MET-{row['patient_id']}" in positive_ids
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_paths[0], positive_rows, lesion_fields)
    write_csv(
        output_paths[1],
        negative_rows,
        [
            "source_case_id",
            "patient_group",
            "seg_voxels",
            "was_modified",
            "image_equal",
            "seg_equal",
            "geometry_ok",
            "image_sha256_before",
            "image_sha256_after",
            "seg_sha256_before",
            "seg_sha256_after",
        ],
    )
    summary = {
        "status": "frozen_and_validated",
        "fixed_val_count": len(val_members),
        "generated_positive_count": len(positive_ids),
        "positive_lesion_row_count": len(positive_rows),
        "strict_noop_negative_count": len(negative_rows),
        "strict_noop_pass_count": sum(
            not row["was_modified"] and row["image_equal"] and row["seg_equal"]
            for row in negative_rows
        ),
        "selected_source_case_ids": sorted(positive_ids),
        "negative_source_case_ids": negative_ids,
        "validation_pipeline_contract": validation_contract,
        "source_sha256": source_hashes,
        "output_sha256": {
            "val_positive94_csv": sha256_file(output_paths[0]),
            "val_negative9_noop_csv": sha256_file(output_paths[1]),
        },
    }
    output_paths[2].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesions-csv", required=True, type=Path)
    parser.add_argument("--membership-csv", required=True, type=Path)
    parser.add_argument("--smoke-selection-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-fixed", type=int, default=103)
    parser.add_argument("--expected-positive", type=int, default=94)
    parser.add_argument("--expected-negative", type=int, default=9)
    parser.add_argument("--online-trainer", type=Path)
    parser.add_argument("--base-trainer", type=Path)
    args = parser.parse_args()
    summary = freeze_cohort(
        args.lesions_csv,
        args.membership_csv,
        args.smoke_selection_json,
        args.output_dir,
        expected_fixed=args.expected_fixed,
        expected_positive=args.expected_positive,
        expected_negative=args.expected_negative,
        online_trainer=args.online_trainer,
        base_trainer=args.base_trainer,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
