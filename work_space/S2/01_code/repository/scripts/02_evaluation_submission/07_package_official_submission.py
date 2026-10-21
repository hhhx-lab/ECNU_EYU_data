#!/usr/bin/env python3
"""Validate S2 Task 1 predictions and build the official submission archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_ID_PATTERN = re.compile(r"^BraTS-MET-[0-9]{5}-[0-9]{3}$")
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
ALLOWED_LABELS = {0, 1, 2, 3, 4}


def load_imaging_modules():
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "nibabel and numpy are required; run this script in the brats2026_s2 environment"
        ) from exc
    return nib, np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_source_cases(source_root: Path, expected_count: int) -> list[Path]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Official validation directory does not exist: {source_root}")
    if expected_count <= 0:
        raise ValueError(f"expected_count must be positive, got {expected_count}")
    case_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    invalid_ids = [path.name for path in case_dirs if not CASE_ID_PATTERN.fullmatch(path.name)]
    if invalid_ids:
        raise ValueError(f"Unexpected official validation case IDs: {invalid_ids[:10]}")
    if len(case_dirs) != expected_count:
        raise ValueError(
            "Official validation case-count mismatch: "
            f"expected={expected_count}, actual={len(case_dirs)}"
        )
    for case_dir in case_dirs:
        case_id = case_dir.name
        expected_files = {f"{case_id}-{modality}.nii.gz" for modality in MODALITIES}
        actual_files = {
            path.name
            for path in case_dir.iterdir()
            if path.is_file() and path.name.endswith(".nii.gz")
        }
        if actual_files != expected_files:
            raise ValueError(
                f"Invalid official validation case {case_id}: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"unexpected_nifti={sorted(actual_files - expected_files)}"
            )
    return case_dirs


def geometry_signature(image: Any, nib: Any, np: Any) -> dict[str, Any]:
    affine = np.asarray(image.affine, dtype=float)
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape={image.shape}")
    return {
        "shape": tuple(int(value) for value in image.shape),
        "spacing": tuple(float(value) for value in image.header.get_zooms()[:3]),
        "origin": tuple(float(value) for value in affine[:3, 3]),
        "orientation": tuple(str(value) for value in nib.aff2axcodes(affine)),
        "affine": affine,
    }


def assert_same_geometry(
    case_id: str,
    candidate_name: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    np: Any,
) -> None:
    if candidate["shape"] != reference["shape"]:
        raise ValueError(
            f"{case_id} {candidate_name} array dimensions differ: "
            f"reference={reference['shape']}, candidate={candidate['shape']}"
        )
    if not np.allclose(candidate["spacing"], reference["spacing"], rtol=0, atol=1e-5):
        raise ValueError(
            f"{case_id} {candidate_name} voxel spacing differs: "
            f"reference={reference['spacing']}, candidate={candidate['spacing']}"
        )
    if not np.allclose(candidate["origin"], reference["origin"], rtol=0, atol=1e-4):
        raise ValueError(
            f"{case_id} {candidate_name} image origin differs: "
            f"reference={reference['origin']}, candidate={candidate['origin']}"
        )
    if candidate["orientation"] != reference["orientation"]:
        raise ValueError(
            f"{case_id} {candidate_name} spatial orientation differs: "
            f"reference={reference['orientation']}, candidate={candidate['orientation']}"
        )
    if not np.allclose(candidate["affine"], reference["affine"], rtol=0, atol=1e-4):
        raise ValueError(f"{case_id} {candidate_name} affine differs from the source image")


def validate_case(case_dir: Path, prediction_path: Path) -> dict[str, object]:
    nib, np = load_imaging_modules()
    case_id = case_dir.name
    reference_image = nib.load(str(case_dir / f"{case_id}-t1n.nii.gz"))
    reference_geometry = geometry_signature(reference_image, nib, np)

    for modality in MODALITIES[1:]:
        source_image = nib.load(str(case_dir / f"{case_id}-{modality}.nii.gz"))
        assert_same_geometry(
            case_id,
            f"source {modality}",
            reference_geometry,
            geometry_signature(source_image, nib, np),
            np,
        )

    prediction_image = nib.load(str(prediction_path))
    prediction_geometry = geometry_signature(prediction_image, nib, np)
    assert_same_geometry(
        case_id,
        "prediction",
        reference_geometry,
        prediction_geometry,
        np,
    )

    prediction = np.asanyarray(prediction_image.dataobj)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{case_id} prediction contains NaN or infinite values")
    if np.issubdtype(prediction.dtype, np.integer):
        rounded = prediction
    else:
        rounded = np.rint(prediction)
        if not np.array_equal(prediction, rounded):
            raise ValueError(f"{case_id} prediction contains non-integer label values")
    labels = {int(value) for value in np.unique(rounded)}
    illegal_labels = sorted(labels - ALLOWED_LABELS)
    if illegal_labels:
        raise ValueError(f"{case_id} prediction contains illegal labels: {illegal_labels}")

    nonzero_voxels = int(np.count_nonzero(rounded))
    return {
        "case_id": case_id,
        "prediction_filename": prediction_path.name,
        "shape": "x".join(str(value) for value in prediction_geometry["shape"]),
        "spacing": json.dumps(prediction_geometry["spacing"]),
        "origin": json.dumps(prediction_geometry["origin"]),
        "orientation": "".join(prediction_geometry["orientation"]),
        "dtype": str(prediction_image.get_data_dtype()),
        "labels": ";".join(str(value) for value in sorted(labels)),
        "nonzero_voxels": nonzero_voxels,
        "empty_prediction": nonzero_voxels == 0,
        "file_size_bytes": prediction_path.stat().st_size,
        "sha256": sha256(prediction_path),
    }


def validate_and_package(
    source_root: Path,
    prediction_dir: Path,
    output_zip: Path,
    manifest_path: Path,
    summary_path: Path,
    *,
    expected_count: int = 179,
) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    prediction_dir = prediction_dir.expanduser().resolve()
    output_zip = output_zip.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"S2 prediction directory does not exist: {prediction_dir}")

    case_dirs = discover_source_cases(source_root, expected_count)
    expected_names = {f"{case_dir.name}.nii.gz" for case_dir in case_dirs}
    actual_names = {
        path.name
        for path in prediction_dir.iterdir()
        if path.is_file() and path.name.endswith(".nii.gz")
    }
    if actual_names != expected_names:
        raise ValueError(
            "Official prediction filename/coverage mismatch: "
            f"missing={sorted(expected_names - actual_names)[:10]}, "
            f"unexpected={sorted(actual_names - expected_names)[:10]}, "
            f"expected_count={len(expected_names)}, actual_count={len(actual_names)}"
        )

    rows = [
        validate_case(case_dir, prediction_dir / f"{case_dir.name}.nii.gz")
        for case_dir in case_dirs
    ]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    temporary_zip = output_zip.with_name(f".{output_zip.name}.tmp")
    if temporary_zip.exists():
        temporary_zip.unlink()
    try:
        with zipfile.ZipFile(
            temporary_zip,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as archive:
            for case_dir in case_dirs:
                filename = f"{case_dir.name}.nii.gz"
                archive.write(prediction_dir / filename, arcname=filename)
        with zipfile.ZipFile(temporary_zip, mode="r") as archive:
            archive_names = archive.namelist()
            if set(archive_names) != expected_names or len(archive_names) != expected_count:
                raise RuntimeError("Submission ZIP content does not match the official case set")
            if any("/" in name or "\\" in name for name in archive_names):
                raise RuntimeError("Submission ZIP must contain NIfTI files at the archive root")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Submission ZIP integrity check failed: {bad_member}")
        temporary_zip.replace(output_zip)
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()

    empty_cases = [str(row["case_id"]) for row in rows if row["empty_prediction"]]
    summary: dict[str, object] = {
        "status": "pass",
        "submission_role": "BraTS_2026_Task1_official_validation_file_prediction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "prediction_dir": str(prediction_dir),
        "output_zip": str(output_zip),
        "manifest_path": str(manifest_path),
        "expected_case_count": expected_count,
        "prediction_count": len(rows),
        "allowed_labels": sorted(ALLOWED_LABELS),
        "empty_prediction_count": len(empty_cases),
        "empty_prediction_case_ids": empty_cases,
        "filename_rule": "BraTS-MET-xxxxx-xxx.nii.gz at archive root",
        "geometry_checks": [
            "array_dimensions",
            "voxel_spacing",
            "image_origin",
            "spatial_orientation",
            "affine",
        ],
        "zip_size_bytes": output_zip.stat().st_size,
        "zip_sha256": sha256(output_zip),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate 179 Task 1 predictions and build a Synapse-ready ZIP."
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--expected-count", type=int, default=179)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest or args.output_zip.with_suffix(".manifest.csv")
    summary_path = args.summary or args.output_zip.with_suffix(".validation.json")
    summary = validate_and_package(
        args.src,
        args.predictions,
        args.output_zip,
        manifest_path,
        summary_path,
        expected_count=args.expected_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
