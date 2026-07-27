#!/usr/bin/env python3
"""Prepare an immutable, two-shard official S2 inference run without packaging."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch


EXPECTED_CHECKPOINT_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_SELECTION_SHA256 = "380467ce9157ffc31b58baaa6d1182e7882cd4c68555ee1a0048304cc708be5f"
EXPECTED_SELECTION_MARKER_SHA256 = "84b424f9120c1738c2adcffbb1e477fdc3d1077382e9138e5cb012ebd1477a08"
EXPECTED_INFERENCE_SHA256 = "fa367326d18d6e03ef3b313daca7213af43a96ecfa818fe29e64c52d75df9b38"
EXPECTED_INFERENCE_TRAINER_SHA256 = "c7c464c9ec283d75ab7d2f353d50025dc52e8b59dcc0d1f0256abdff7b78df13"
EXPECTED_TRANSFER_AUDIT_SHA256 = "8894ccbd8ad92f3e6991d566fd8fc5328a789b04707f3689ad2555e47a308d9e"
EXPECTED_TRANSFER_MARKER_SHA256 = "adc785045156079565ebe4abf842e8a2c0b9e39366fb189f4b827742354f8c71"
EXPECTED_TRANSFER_MANIFEST_SHA256 = "47e4a87c7d079d8005efe7df2d3e4d53232c295fa3afa478b6648bffa2ac5d51"
CASE_PATTERN = re.compile(r"^BraTS-MET-[0-9]{5}-[0-9]{3}$")
MODALITIES = ("t1n", "t1c", "t2w", "t2f")
CHANNELS = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}
SHARDS = ("gpu0", "gpu1")


class PreparationError(RuntimeError):
    """Raised when an immutable inference prerequisite does not match."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def source_manifest(cases: list[Path]) -> tuple[str, int, int]:
    rows: list[str] = []
    total_bytes = 0
    for case_dir in cases:
        for modality in MODALITIES:
            path = case_dir / f"{case_dir.name}-{modality}.nii.gz"
            size = path.stat().st_size
            rows.append(f"{case_dir.name}/{path.name}\t{size}\t{sha256_file(path)}")
            total_bytes += size
    payload = "\n".join(rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(rows), total_bytes


def discover_cases(source_root: Path, expected_count: int) -> list[Path]:
    require(source_root.is_dir(), f"Missing official source root: {source_root}")
    cases = sorted(path for path in source_root.iterdir() if path.is_dir())
    require(len(cases) == expected_count, f"Official case count mismatch: {len(cases)}/{expected_count}")
    for case_dir in cases:
        require(CASE_PATTERN.fullmatch(case_dir.name) is not None, f"Invalid case ID: {case_dir.name}")
        expected_names = {f"{case_dir.name}-{modality}.nii.gz" for modality in MODALITIES}
        actual_names = {
            path.name
            for path in case_dir.iterdir()
            if path.is_file() and path.name.endswith(".nii.gz")
        }
        require(actual_names == expected_names, f"Official modality contract failed: {case_dir.name}")
        for name in expected_names:
            require((case_dir / name).stat().st_size > 0, f"Empty source image: {case_dir / name}")
    return cases


def validate_geometry_and_weight(case_dir: Path) -> int:
    signatures: list[tuple[tuple[int, ...], np.ndarray, np.ndarray]] = []
    for modality in MODALITIES:
        image = nib.load(str(case_dir / f"{case_dir.name}-{modality}.nii.gz"))
        shape = tuple(int(value) for value in image.shape)
        require(len(shape) == 3, f"Expected 3D image: {case_dir.name} {modality}")
        zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
        affine = np.asarray(image.affine, dtype=np.float64)
        signatures.append((shape, zooms, affine))
    reference_shape, reference_zooms, reference_affine = signatures[0]
    for shape, zooms, affine in signatures[1:]:
        require(shape == reference_shape, f"Source shape mismatch: {case_dir.name}")
        require(
            np.allclose(zooms, reference_zooms, rtol=0, atol=1e-5),
            f"Source spacing mismatch: {case_dir.name}",
        )
        require(
            np.allclose(affine, reference_affine, rtol=0, atol=1e-4),
            f"Source affine mismatch: {case_dir.name}",
        )
    return math.prod(signatures[0][0])


def greedy_shards(cases: list[Path]) -> tuple[dict[str, list[Path]], dict[str, int]]:
    weighted = [(validate_geometry_and_weight(case_dir), case_dir) for case_dir in cases]
    assignments = {name: [] for name in SHARDS}
    totals = {name: 0 for name in SHARDS}
    for weight, case_dir in sorted(weighted, key=lambda item: (-item[0], item[1].name)):
        shard = min(SHARDS, key=lambda name: (totals[name], name))
        assignments[shard].append(case_dir)
        totals[shard] += weight
    for shard in SHARDS:
        assignments[shard].sort(key=lambda path: path.name)
    require(sum(len(value) for value in assignments.values()) == len(cases), "Shard count mismatch")
    require(
        not ({path.name for path in assignments["gpu0"]} & {path.name for path in assignments["gpu1"]}),
        "Shard overlap detected",
    )
    return assignments, totals


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_inputs(
    repository: Path,
    source_root: Path,
    run_root: Path,
    expected_count: int,
) -> dict[str, Any]:
    prepare_script = repository / "scripts/06_prepare_official_validation.py"
    require(prepare_script.is_file(), f"Missing input preparation script: {prepare_script}")
    module = load_module(prepare_script, "prepare_official_validation_locked")
    summary = module.prepare_official_validation(
        source_root,
        run_root / "input_all",
        run_root / "official_validation_input_manifest.csv",
        run_root / "official_validation_preparation.json",
        expected_count=expected_count,
        mode="symlink",
        clean=False,
    )
    require(summary.get("status") == "pass", "Input preparation did not pass")
    require(summary.get("case_count") == expected_count, "Prepared case count mismatch")
    require(summary.get("nifti_count") == expected_count * 4, "Prepared NIfTI count mismatch")
    return {
        "script": str(prepare_script),
        "script_sha256": sha256_file(prepare_script),
        "summary_sha256": sha256_file(run_root / "official_validation_preparation.json"),
        "manifest_sha256": sha256_file(run_root / "official_validation_input_manifest.csv"),
        "case_id_sha256": summary["case_id_sha256"],
    }


def build_deployment_model(
    model_root: Path,
    checkpoint: Path,
    dataset_json: Path,
    plans_json: Path,
) -> dict[str, Any]:
    require(sha256_file(checkpoint) == EXPECTED_CHECKPOINT_SHA256, "Selected checkpoint SHA drift")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(checkpoint_data.get("trainer_name") == "nnUNetTrainerBraTS2026RCFocalCompletionFineTune", "Checkpoint trainer drift")
    require(checkpoint_data.get("current_epoch") == 200, "Checkpoint epoch drift")
    init_args = checkpoint_data.get("init_args")
    require(isinstance(init_args, dict), "Checkpoint init_args missing")
    dataset_value = load_json(dataset_json)
    plans_value = load_json(plans_json)
    require(init_args.get("dataset_json") == dataset_value, "dataset.json differs from checkpoint")
    require(init_args.get("plans") == plans_value, "plans.json differs from checkpoint")
    require(init_args.get("configuration") == "3d_fullres", "Checkpoint configuration drift")
    require(init_args.get("fold") == 0, "Checkpoint fold drift")

    model_root.mkdir(parents=True, exist_ok=False)
    (model_root / "fold_0").mkdir()
    (model_root / "dataset.json").symlink_to(dataset_json.resolve())
    (model_root / "plans.json").symlink_to(plans_json.resolve())
    (model_root / "fold_0/checkpoint_final.pth").symlink_to(checkpoint.resolve())
    return {
        "model_root": str(model_root),
        "trainer": checkpoint_data["trainer_name"],
        "epoch": checkpoint_data["current_epoch"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dataset_json_path": str(dataset_json),
        "dataset_json_sha256": sha256_file(dataset_json),
        "plans_json_path": str(plans_json),
        "plans_json_sha256": sha256_file(plans_json),
        "checkpoint_metadata_matches_json": True,
    }


def materialize_shards(
    run_root: Path,
    assignments: dict[str, list[Path]],
    voxel_totals: dict[str, int],
) -> dict[str, Any]:
    all_ids: set[str] = set()
    result: dict[str, Any] = {}
    for shard in SHARDS:
        input_root = run_root / f"input_{shard}"
        input_root.mkdir()
        rows = []
        for case_dir in assignments[shard]:
            case_id = case_dir.name
            require(case_id not in all_ids, f"Duplicate shard case: {case_id}")
            all_ids.add(case_id)
            for channel in CHANNELS.values():
                name = f"{case_id}_{channel}.nii.gz"
                (input_root / name).symlink_to(Path("../input_all") / name)
            rows.append(case_id)
        manifest_path = run_root / f"{shard}_case_ids.txt"
        manifest_path.write_text("".join(f"{value}\n" for value in rows), encoding="utf-8")
        result[shard] = {
            "case_count": len(rows),
            "nifti_count": len(rows) * 4,
            "estimated_voxels": voxel_totals[shard],
            "case_ids_sha256": sha256_file(manifest_path),
            "case_ids_path": str(manifest_path),
            "input_root": str(input_root),
            "output_root": str(run_root / f"output_{shard}"),
            "physical_gpu": int(shard[-1]),
        }
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    run_root = args.run_root.resolve()
    selection_root = args.selection_root.resolve()
    repository = args.repository.resolve()
    runtime_env = args.runtime_env.resolve()
    checkpoint = args.checkpoint.resolve()
    dataset_json = args.dataset_json.resolve()
    plans_json = args.plans_json.resolve()

    require(not run_root.exists(), f"Run root already exists: {run_root}")
    require(source_root.is_dir(), f"Source root missing: {source_root}")
    require(selection_root.is_dir(), f"Selection root missing: {selection_root}")
    require(repository.is_dir(), f"Repository missing: {repository}")
    require(runtime_env.is_dir(), f"Runtime environment missing: {runtime_env}")
    require(Path(sys.executable).resolve() == (runtime_env / "bin/python").resolve(), "Wrong runtime Python")
    for path in (checkpoint, dataset_json, plans_json):
        require(path.is_file(), f"Missing model prerequisite: {path}")

    selection_path = selection_root / "MODEL_SELECTION.json"
    selection_marker = selection_root / "MODEL_SELECTION_COMPLETE.ok"
    require(sha256_file(selection_path) == EXPECTED_SELECTION_SHA256, "Model selection SHA drift")
    require(sha256_file(selection_marker) == EXPECTED_SELECTION_MARKER_SHA256, "Selection marker SHA drift")
    selection = load_json(selection_path)
    require(selection.get("status") == "pass", "Model selection status failed")
    require(selection.get("selected_model") == "E", "Official inference is not bound to original E")
    require(selection.get("selected_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256, "Selection checkpoint drift")

    transfer_audit_path = source_root.parent / "TRANSFER_AUDIT.json"
    transfer_marker_path = source_root.parent / "TRANSFER_COMPLETE.ok"
    require(sha256_file(transfer_audit_path) == EXPECTED_TRANSFER_AUDIT_SHA256, "Transfer audit SHA drift")
    require(sha256_file(transfer_marker_path) == EXPECTED_TRANSFER_MARKER_SHA256, "Transfer marker SHA drift")
    transfer_audit = load_json(transfer_audit_path)
    require(transfer_audit.get("status") == "pass", "Transfer audit status failed")
    require(transfer_audit.get("case_count") == args.expected_count, "Transfer case count drift")
    require(transfer_audit.get("nifti_count") == args.expected_count * 4, "Transfer NIfTI count drift")
    require(transfer_audit.get("manifests_byte_identical") is True, "Transfer manifests differ")
    require(
        transfer_audit.get("remote_manifest_sha256") == EXPECTED_TRANSFER_MANIFEST_SHA256,
        "Transfer manifest identity drift",
    )

    inference_script = repository / "inference_frozen.py"
    inference_trainer = repository / "custom_nnunet/nnUNetTrainerBraTS2026RC_inference.py"
    require(sha256_file(inference_script) == EXPECTED_INFERENCE_SHA256, "Frozen inference script drift")
    require(sha256_file(inference_trainer) == EXPECTED_INFERENCE_TRAINER_SHA256, "Inference trainer drift")

    cases = discover_cases(source_root, args.expected_count)
    assignments, voxel_totals = greedy_shards(cases)
    run_root.mkdir(parents=True, mode=0o755)
    try:
        (run_root / "logs").mkdir()
        input_evidence = materialize_inputs(repository, source_root, run_root, args.expected_count)
        model_evidence = build_deployment_model(
            run_root / "model",
            checkpoint,
            dataset_json,
            plans_json,
        )
        shard_evidence = materialize_shards(run_root, assignments, voxel_totals)
        source_digest, source_nifti_count, source_bytes = source_manifest(cases)

        contract: dict[str, Any] = {
            "schema_version": 1,
            "status": "prepared",
            "generated_at_utc": utc_now(),
            "role": "BraTS_2026_S2_Task1_official_179_inference",
            "run_root": str(run_root),
            "source_root": str(source_root),
            "expected_case_count": args.expected_count,
            "source_nifti_count": source_nifti_count,
            "source_total_bytes": source_bytes,
            "source_manifest_sha256": source_digest,
            "source_transfer": {
                "audit": str(transfer_audit_path),
                "audit_sha256": EXPECTED_TRANSFER_AUDIT_SHA256,
                "completion_marker": str(transfer_marker_path),
                "completion_marker_sha256": EXPECTED_TRANSFER_MARKER_SHA256,
                "local_remote_manifest_sha256": EXPECTED_TRANSFER_MANIFEST_SHA256,
                "local_remote_manifests_byte_identical": True,
            },
            "selection": {
                "root": str(selection_root),
                "model_selection_sha256": EXPECTED_SELECTION_SHA256,
                "completion_marker_sha256": EXPECTED_SELECTION_MARKER_SHA256,
                "selected_model": "E",
                "selected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            },
            "runtime": {
                "environment": str(runtime_env),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "nnunetv2": importlib.metadata.version("nnunetv2"),
                "nibabel": importlib.metadata.version("nibabel"),
                "inference_script": str(inference_script),
                "inference_script_sha256": EXPECTED_INFERENCE_SHA256,
                "inference_trainer": str(inference_trainer),
                "inference_trainer_sha256": EXPECTED_INFERENCE_TRAINER_SHA256,
                "prepare_script": str(Path(__file__).resolve()),
                "prepare_script_sha256": sha256_file(Path(__file__).resolve()),
            },
            "input_preparation": input_evidence,
            "model": model_evidence,
            "shards": shard_evidence,
            "inference": {
                "fold": 0,
                "checkpoint_name": "checkpoint_final.pth",
                "tile_step_size": 0.5,
                "use_gaussian": True,
                "use_mirroring": True,
                "perform_everything_on_device": True,
                "preprocess_workers_per_shard": 4,
                "export_workers_per_shard": 4,
                "save_probabilities": False,
                "uses_met_aug": False,
                "uses_g1_g2_diffusion": False,
                "test_time_training": False,
            },
            "packaging": {
                "zip_allowed": False,
                "synapse_upload_allowed": False,
            },
        }
        contract["contract_sha256"] = canonical_sha256(contract)
        contract_path = run_root / "OFFICIAL_INFERENCE_CONTRACT.json"
        write_json(contract_path, contract)
        marker = (
            "status=pass\n"
            f"prepared_at_utc={utc_now()}\n"
            f"contract_identity={contract['contract_sha256']}\n"
            f"contract_file_sha256={sha256_file(contract_path)}\n"
            f"checkpoint_sha256={EXPECTED_CHECKPOINT_SHA256}\n"
            f"cases={args.expected_count}\n"
            f"nifti={source_nifti_count}\n"
        )
        (run_root / "PREPARE_COMPLETE.ok").write_text(marker, encoding="utf-8")
    except Exception:
        (run_root / "PREPARE_FAILED.ok").write_text(
            f"status=fail\nfailed_at_utc={utc_now()}\n",
            encoding="utf-8",
        )
        raise

    return {
        "status": "pass",
        "run_root": str(run_root),
        "contract_identity": contract["contract_sha256"],
        "contract_file_sha256": sha256_file(run_root / "OFFICIAL_INFERENCE_CONTRACT.json"),
        "source_manifest_sha256": source_digest,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "shards": shard_evidence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--selection-root", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--runtime-env", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-json", required=True, type=Path)
    parser.add_argument("--plans-json", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=179)
    return parser


def main() -> None:
    try:
        result = prepare(build_parser().parse_args())
    except PreparationError as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
