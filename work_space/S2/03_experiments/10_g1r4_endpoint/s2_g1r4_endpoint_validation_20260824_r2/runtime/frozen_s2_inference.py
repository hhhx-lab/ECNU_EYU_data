#!/usr/bin/env python3
"""Run one frozen S2 checkpoint through a shared inference-only trainer shim."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import nnunetv2.inference.predict_from_raw_data as predict_module

from custom_nnunet.nnUNetTrainerBraTS2026RC_inference import nnUNetTrainerBraTS2026RC


ALLOWED_TRAINERS = {
    "nnUNetTrainerBraTS2026RC",
    "nnUNetTrainerBraTS2026RCCompletionFineTune",
    "nnUNetTrainerBraTS2026RCFocalCompletionFineTune",
    "nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--export-workers", type=int, default=4)
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    model_root = args.model_root.resolve()
    audit_path = args.audit_json.resolve()
    require(input_root.is_dir() and model_root.is_dir(), "input or model root missing")
    require(not output_root.exists() and not audit_path.exists(), "exclusive inference target exists")
    checkpoint = model_root / f"fold_{args.fold}" / "checkpoint_final.pth"
    require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
    require(sha256_file(checkpoint) == args.expected_checkpoint_sha256, "checkpoint SHA drift")
    input_files = sorted(input_root.glob("*.nii.gz"))
    require(len(input_files) == args.expected_count * 4, "input channel count drift")
    by_case: dict[str, set[str]] = {}
    for path in input_files:
        stem = path.name.removesuffix(".nii.gz")
        case_id, channel = stem.rsplit("_", 1)
        by_case.setdefault(case_id, set()).add(channel)
    require(len(by_case) == args.expected_count, "input case count drift")
    require(all(channels == {"0000", "0001", "0002", "0003"} for channels in by_case.values()), "incomplete channel set")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly one visible CUDA device is required")

    def trainer_finder(trainer_name: str):
        if trainer_name not in ALLOWED_TRAINERS:
            raise ValueError(f"unexpected checkpoint trainer: {trainer_name}")
        return nnUNetTrainerBraTS2026RC

    predict_module.recursive_find_trainer_class_by_name = trainer_finder
    predictor = predict_module.nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    predictor.initialize_from_trained_model_folder(
        str(model_root), use_folds=(args.fold,), checkpoint_name="checkpoint_final.pth"
    )
    require(predictor.trainer_name in ALLOWED_TRAINERS, f"loaded trainer is not frozen: {predictor.trainer_name}")
    predictor.predict_from_files(
        str(input_root),
        str(output_root),
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=args.preprocess_workers,
        num_processes_segmentation_export=args.export_workers,
    )
    predictions = sorted(output_root.glob("*.nii.gz"))
    predicted_ids = {path.name.removesuffix(".nii.gz") for path in predictions}
    require(len(predictions) == args.expected_count and predicted_ids == set(by_case), "prediction coverage drift")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "model_name": args.model_name,
        "trainer_name_from_checkpoint": predictor.trainer_name,
        "network_builder_shim": "nnUNetTrainerBraTS2026RC",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "case_count": args.expected_count,
        "prediction_count": len(predictions),
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "save_probabilities": False,
        "device_name": torch.cuda.get_device_name(0),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

