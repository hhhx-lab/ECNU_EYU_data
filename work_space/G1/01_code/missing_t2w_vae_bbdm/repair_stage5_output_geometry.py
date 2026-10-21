#!/usr/bin/env python3
"""Audit or repair Stage-5 synthesized T2W geometry without resampling voxels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--synthetic-root", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument(
        "--repair-output-root",
        help="Write geometry-fixed copies here; omit to require inputs already aligned.",
    )
    parser.add_argument("--expected-cases", type=int, default=103)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_case_file(case_dir: Path, case_id: str, modality: str) -> Path:
    candidates = [
        case_dir / f"{case_id}-{modality}.nii.gz",
        case_dir / f"{modality}.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(case_dir.glob(f"*{modality}*.nii.gz"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve {modality} for {case_id}: {matches}")


def matrix_text(matrix: np.ndarray) -> str:
    return json.dumps(np.asarray(matrix, dtype=float).round(8).tolist(), separators=(",", ":"))


def geometry_matches(reference: nib.spatialimages.SpatialImage, image: nib.spatialimages.SpatialImage) -> bool:
    return (
        reference.shape == image.shape
        and np.allclose(reference.affine, image.affine, atol=1e-4, rtol=0.0)
        and np.allclose(
            reference.header.get_zooms()[:3],
            image.header.get_zooms()[:3],
            atol=1e-5,
            rtol=0.0,
        )
    )


def write_geometry_fixed(
    generated: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    destination: Path,
) -> float:
    data = generated.get_fdata(dtype=np.float32)
    header = generated.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(data.shape)
    header.set_zooms(reference.header.get_zooms()[: len(data.shape)])
    fixed = nib.Nifti1Image(data, reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    fixed.set_qform(qform if qform is not None else reference.affine, int(qcode))
    fixed.set_sform(sform if sform is not None else reference.affine, int(scode))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.nii.gz")
    nib.save(fixed, str(temporary))
    temporary.replace(destination)

    reloaded = nib.load(str(destination))
    if not geometry_matches(reference, reloaded):
        raise ValueError(f"repaired geometry still mismatches: {destination}")
    repaired_data = reloaded.get_fdata(dtype=np.float32)
    max_abs_difference = float(np.max(np.abs(repaired_data - data)))
    if max_abs_difference > 1e-7:
        raise ValueError(
            f"geometry repair changed voxel values for {destination}: {max_abs_difference}"
        )
    return max_abs_difference


def run(args: argparse.Namespace) -> dict[str, object]:
    real_root = Path(args.real_root).resolve()
    synthetic_root = Path(args.synthetic_root).resolve()
    metrics_csv = Path(args.metrics_csv).resolve()
    audit_root = Path(args.audit_root).resolve()
    repair_root = Path(args.repair_output_root).resolve() if args.repair_output_root else None

    with metrics_csv.open(encoding="utf-8-sig", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    if len(metric_rows) != args.expected_cases:
        raise ValueError(
            f"metrics case count {len(metric_rows)} != expected {args.expected_cases}"
        )
    case_ids = [row["subject"].strip() for row in metric_rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("metrics.csv contains duplicate subject IDs")

    if audit_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"audit root already exists: {audit_root}")
        shutil.rmtree(audit_root)
    if repair_root is not None and repair_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"repair output already exists: {repair_root}")
        shutil.rmtree(repair_root)
    audit_root.mkdir(parents=True)
    if repair_root is not None:
        repair_root.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    mismatch_count = 0
    repaired_count = 0
    for case_id in case_ids:
        case_dir = real_root / case_id
        reference_path = find_case_file(case_dir, case_id, "t2w")
        generated_path = synthetic_root / f"{case_id}-t2w.nii.gz"
        if not generated_path.is_file():
            raise FileNotFoundError(generated_path)
        reference = nib.load(str(reference_path))
        generated = nib.load(str(generated_path))
        if reference.shape != generated.shape:
            raise ValueError(
                f"{case_id}: shape mismatch reference={reference.shape}, generated={generated.shape}"
            )
        generated_data = generated.get_fdata(dtype=np.float32)
        if not np.isfinite(generated_data).all():
            raise ValueError(f"{case_id}: generated image contains NaN/Inf")
        matched_before = geometry_matches(reference, generated)
        mismatch_count += int(not matched_before)
        max_abs_difference = 0.0
        repaired = False
        destination = ""
        if repair_root is not None:
            destination_path = repair_root / generated_path.name
            max_abs_difference = write_geometry_fixed(generated, reference, destination_path)
            repaired = True
            repaired_count += 1
            destination = str(destination_path)
        rows.append(
            {
                "case_id": case_id,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "output_path": destination,
                "shape": "x".join(str(value) for value in generated.shape),
                "geometry_matched_before": matched_before,
                "repaired": repaired,
                "max_abs_voxel_difference_after_repair": max_abs_difference,
                "reference_affine": matrix_text(reference.affine),
                "generated_affine_before": matrix_text(generated.affine),
            }
        )

    audit_csv = audit_root / "geometry_audit.csv"
    with audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "case_count": len(rows),
        "geometry_mismatch_before_count": mismatch_count,
        "repaired_count": repaired_count,
        "repair_mode": repair_root is not None,
        "voxel_resampling_performed": False,
        "max_abs_voxel_difference_after_repair": max(
            float(row["max_abs_voxel_difference_after_repair"]) for row in rows
        ),
        "source_synthetic_root": str(synthetic_root),
        "repair_output_root": str(repair_root) if repair_root is not None else None,
    }
    (audit_root / "geometry_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if mismatch_count and repair_root is None:
        raise ValueError(
            f"{mismatch_count}/{len(rows)} generated images have mismatched geometry; "
            "rerun with --repair-output-root"
        )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
