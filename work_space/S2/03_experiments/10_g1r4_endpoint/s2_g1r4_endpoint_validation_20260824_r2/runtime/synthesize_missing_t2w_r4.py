#!/usr/bin/env python3
"""Synthesize missing T2W with the frozen G1 r4 Ensemble contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

import configs
import synthesis.pipeline as pipeline
import synthesis.utils as utils


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    json.loads(path.read_text(encoding="utf-8"))


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path, expected_count: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == expected_count, f"manifest count {len(rows)} != {expected_count}")
    ids = [row["source_case_id"].strip() for row in rows]
    require(len(ids) == len(set(ids)), "manifest contains duplicate source IDs")
    require(all(not row.get("t2w_source_path", "").strip() for row in rows), "missing cohort exposes a source T2W path")
    require(all(row.get("source_t2w_allowed", "").strip().lower() == "false" for row in rows), "source T2W is not forbidden")
    return rows


def resolve_source(row: dict[str, str], modality: str, source_root: Path) -> Path:
    path = Path(row[f"{modality}_source_path"]).resolve()
    require(path.is_relative_to(source_root), f"{modality} escaped source root: {path}")
    require(path.is_file() and path.stat().st_size > 0, f"missing {modality}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--spatial-audit", required=True, type=Path)
    parser.add_argument("--geometry-audit-json", required=True, type=Path)
    parser.add_argument("--geometry-audit-csv", required=True, type=Path)
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--vae-path", required=True, type=Path)
    parser.add_argument("--encdec-path", required=True, type=Path)
    parser.add_argument("--bbdm-path", required=True, type=Path)
    parser.add_argument("--vae-sha256", required=True)
    parser.add_argument("--encdec-sha256", required=True)
    parser.add_argument("--bbdm-sha256", required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    require(source_root.is_dir(), f"missing source root: {source_root}")
    require(not output_root.exists(), f"exclusive output already exists: {output_root}")
    for target in (args.spatial_audit, args.geometry_audit_json, args.geometry_audit_csv, args.run_json):
        require(not target.exists(), f"refusing to overwrite audit: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest.resolve(), args.expected_count)

    checkpoints = {
        "vae": (args.vae_path.resolve(), args.vae_sha256),
        "encdec": (args.encdec_path.resolve(), args.encdec_sha256),
        "bbdm": (args.bbdm_path.resolve(), args.bbdm_sha256),
    }
    for name, (path, expected_sha) in checkpoints.items():
        require(path.is_file(), f"missing {name} checkpoint: {path}")
        require(sha256_file(path) == expected_sha, f"{name} checkpoint SHA drift")
    configured_checkpoints = {
        "vae": Path(configs.PATH_NAME_WEIGHTS_VAE).resolve(),
        "encdec": Path(configs.PATH_NAME_WEIGHTS_ENCDEC).resolve(),
        "bbdm": Path(configs.PATH_NAME_WEIGHTS_BBDM).resolve(),
    }
    for name, configured_path in configured_checkpoints.items():
        require(configured_path == checkpoints[name][0], f"{name} configured checkpoint path drift")
    require(float(configs.BBDM_S) == 0.01, f"BBDM s drift: {configs.BBDM_S}")
    require(torch.cuda.is_available(), "CUDA is required")
    require(torch.cuda.device_count() == 1, f"expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    output_root.mkdir(parents=True, exist_ok=False)

    models = pipeline.prepare_synthesis_models("ensamble", device)
    spatial_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    source_bindings: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    for row in rows:
        case_id = row["source_case_id"].strip()
        source_paths = {
            modality: resolve_source(row, modality, source_root)
            for modality in ("t1n", "t1c", "t2f", "seg")
        }
        prepared = utils.prepare_subject_space(
            [source_paths[modality] for modality in configs.AVAILABLE_MODALITIES],
            seg_path=source_paths["seg"],
            target_shape=configs.SHAPE_PREPROCESS_IMG,
            base_spacing_mm=1.0,
            margin_mm=5.0,
        )
        latents = [pipeline.encode_image(image, models["vae"]) for image in prepared["images"]]
        synthesis_data = {"latens_list": latents}
        encdec_latent = pipeline.run_encdec_synthesis(
            synthesis_data, device, unet=models["unet_encdec"]
        )
        bbdm_latent = pipeline.run_bbdm_synthesis(
            synthesis_data,
            device,
            models=(models["unet_bbdm"], models["conditions_model"], models["noise_scheduler"]),
        )
        encdec_image = pipeline.decode_latents(encdec_latent, models["vae"])
        bbdm_image = pipeline.decode_latents(bbdm_latent, models["vae"])
        ensemble_image = utils.combine_images([encdec_image, bbdm_image], combination_type="mean")
        native_image = pipeline.restore_generated_image(ensemble_image, prepared, "t2w")
        require(np.isfinite(native_image).all(), f"nonfinite synthesized T2W: {case_id}")
        require(float(np.max(native_image) - np.min(native_image)) > 0.0, f"constant synthesized T2W: {case_id}")
        output_path = output_root / f"{case_id}-t2w.nii.gz"
        utils.save_nifti(native_image, prepared["native_affine"], str(output_path))

        output_image = nib.load(str(output_path))
        reference = nib.load(str(source_paths["seg"]))
        shape_match = tuple(output_image.shape) == tuple(reference.shape)
        affine_match = bool(np.allclose(output_image.affine, reference.affine, rtol=0.0, atol=1e-5))
        require(shape_match and affine_match, f"native geometry mismatch: {case_id}")
        transform = prepared["transform"]
        spatial_rows.append(
            {
                "subject": case_id,
                "spatial_preprocessing": "foreground_centered_isotropic_resample_v1",
                "native_shape": "x".join(map(str, transform.native_shape)),
                "target_shape": "x".join(map(str, transform.target_shape)),
                "target_spacing_mm": transform.target_spacing_mm,
                "foreground_voxel_count": transform.foreground_voxel_count,
                "lesion_voxel_count": transform.lesion_voxel_count,
                "foreground_outside_voxel_count": prepared["foreground_support_audit"]["outside_voxel_count"],
                "lesion_outside_voxel_count": prepared["lesion_support_audit"]["outside_voxel_count"] if prepared["lesion_support_audit"] else 0,
                "reference_segmentation_used_for_support_audit": True,
                "reference_segmentation_is_model_input": False,
            }
        )
        geometry_rows.append(
            {
                "subject": case_id,
                "shape_match": shape_match,
                "affine_match": affine_match,
                "repaired": False,
                "voxel_resampling_performed_after_restore": False,
                "output_sha256": sha256_file(output_path),
            }
        )
        source_bindings.append(
            {
                "subject": case_id,
                "source_t2w_read": False,
                **{f"{modality}_path": str(path) for modality, path in source_paths.items()},
                **{f"{modality}_sha256": sha256_file(path) for modality, path in source_paths.items()},
            }
        )

    require(len(list(output_root.glob("*.nii.gz"))) == args.expected_count, "synthesized output count drift")
    require(all(int(row["foreground_outside_voxel_count"]) == 0 for row in spatial_rows), "foreground escaped FOV")
    require(all(int(row["lesion_outside_voxel_count"]) == 0 for row in spatial_rows), "lesion escaped FOV")
    write_csv_exclusive(args.spatial_audit, spatial_rows)
    write_csv_exclusive(args.geometry_audit_csv, geometry_rows)
    geometry_payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "case_count": len(geometry_rows),
        "geometry_mismatch_before_count": 0,
        "repaired_count": 0,
        "repair_mode": False,
        "voxel_resampling_performed": False,
        "source_t2w_read_count": 0,
        "rows": geometry_rows,
    }
    write_json_exclusive(args.geometry_audit_json, geometry_payload)
    run_payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "cohort": args.cohort,
        "case_count": args.expected_count,
        "synthesis_type": "ensamble",
        "spatial_preprocessing": "foreground_centered_isotropic_resample_v1",
        "native_geometry_restored": True,
        "source_t2w_read": False,
        "bbdm_s": 0.01,
        "configured_checkpoint_binding_validated": True,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": torch.cuda.get_device_name(0),
        "checkpoint_bindings": {
            name: {"path": str(path), "sha256": expected_sha}
            for name, (path, expected_sha) in checkpoints.items()
        },
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "output_root": str(output_root),
        "spatial_audit_sha256": sha256_file(args.spatial_audit),
        "geometry_audit_sha256": sha256_file(args.geometry_audit_json),
        "source_bindings": source_bindings,
    }
    write_json_exclusive(args.run_json, run_payload)
    print(json.dumps({"status": "pass", "cohort": args.cohort, "case_count": args.expected_count}, sort_keys=True))


if __name__ == "__main__":
    main()
