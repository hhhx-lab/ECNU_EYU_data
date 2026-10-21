#!/usr/bin/env python3
"""Run an isolated p=0 E-continue training-path smoke on one CUDA GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
import traceback
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TRAINER = "nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune"
EXPECTED_DATASET = "Dataset264_BraTS2026_MET_Completion"
EXPECTED_E_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_TRAIN_SHA256 = "1cfa31a71c1c5014fb6ed457277f634ef0db4a95607270f66a7eafcbf9020b52"
EXPECTED_VAL_SHA256 = "7027d91362adf799901544070204f0821b5ce0608f4d5c85c4d878ee5cc7219a"
EXPECTED_PLANS_FILE_SHA256 = "c20ac311f0b3db0f0710e98b0b56e65e8bb38c13b95094b6d6f9966ac529ffa5"
EXPECTED_PLANS_CANONICAL_SHA256 = "d67b890d1d035a90216a13d120aea2ec213de9646c5cda454711073a8407a6e1"
EXPECTED_CACHE_AUDIT_IDENTITY = "17b7fc946528f68c5f9da7157cd1d80135edb6fcc1a6d1e9ecd2a450d17e056f"
EXPECTED_TRAIN_COUNT = 1035
EXPECTED_VAL_COUNT = 103
EXPECTED_CACHE_COUNT = 1138
SMOKE_REPORT_SCHEMA = 1
FORBIDDEN_GENERATIVE_ENVIRONMENT = (
    "S2_MET_AUG_COMPONENT_MANIFEST",
    "S2_MET_AUG_ROUTE_CONFIG",
    "S2_MET_AUG_VALID_MASK_MANIFEST",
    "S2_MET_AUG_ROUTE_GATE",
    "S2_MET_AUG_G1_CODE_DIR",
    "S2_MET_AUG_G1_CHECKPOINT_ROOT",
    "S2_MET_AUG_G1_CHECKPOINT_SELECTION",
    "S2_MET_AUG_G2_QC_GATE",
    "G1_CODE_DIR",
    "G1_CHECKPOINT_ROOT",
    "G1_SELECTION",
    "G2_PARENT_GATE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    parser.add_argument("--nnunet-preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="264")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--trainer", default=EXPECTED_TRAINER)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--min-gpu-memory-gib", type=float, default=30.0)
    parser.add_argument("--eta-safety-factor", type=float, default=1.25)
    parser.add_argument("--max-estimated-training-hours", type=float, default=45.0)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any], *, exclude: tuple[str, ...] = ()) -> str:
    filtered = {key: value for key, value in payload.items() if key not in exclude}
    encoded = json.dumps(filtered, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_existing(path: str | None, *, label: str, directory: bool) -> Path:
    if not path:
        raise ValueError(f"{label} is required")
    resolved = Path(path).expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"missing {label} {kind}: {resolved}")
    return resolved


def validate_split_dir(path: str | Path) -> dict[str, Any]:
    root = _resolved_existing(str(path), label="fixed split", directory=True)
    train_file = _resolved_existing(str(root / "train_fixed.txt"), label="training split", directory=False)
    val_file = _resolved_existing(str(root / "val_fixed.txt"), label="validation split", directory=False)
    train_ids = [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_ids = [line.strip() for line in val_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_sha256 = sha256_file(train_file)
    val_sha256 = sha256_file(val_file)
    if (len(train_ids), len(val_ids)) != (EXPECTED_TRAIN_COUNT, EXPECTED_VAL_COUNT):
        raise RuntimeError(
            "E-continue requires the fixed Dataset264 1035/103 split: "
            f"observed={len(train_ids)}/{len(val_ids)}"
        )
    if train_sha256 != EXPECTED_TRAIN_SHA256 or val_sha256 != EXPECTED_VAL_SHA256:
        raise RuntimeError("fixed Dataset264 split SHA256 drifted")
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise RuntimeError("fixed split contains duplicate case IDs")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("fixed training and validation splits overlap")
    return {
        "root": str(root),
        "train_file": str(train_file),
        "train_count": len(train_ids),
        "train_sha256": train_sha256,
        "val_file": str(val_file),
        "val_count": len(val_ids),
        "val_sha256": val_sha256,
    }


def reject_generative_environment(environment: Mapping[str, str] | None = None) -> None:
    values = os.environ if environment is None else environment
    enabled = str(values.get("S2_MET_AUG_ENABLE", "0")).strip()
    if enabled not in ("", "0"):
        raise RuntimeError("E-continue smoke requires S2_MET_AUG_ENABLE=0")
    present = [name for name in FORBIDDEN_GENERATIVE_ENVIRONMENT if str(values.get(name, "")).strip()]
    if present:
        raise RuntimeError(
            "E-continue smoke refuses Route/G1/G2/Diffusion asset variables: " + ",".join(present)
        )


def validate_cache_audit(
    audit_path: str | Path,
    *,
    nnunet_preprocessed: Path,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = _resolved_existing(str(audit_path), label="true-1mm cache audit", directory=False)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "pass":
        raise RuntimeError("true-1mm cache audit is not pass")
    if payload.get("audit_identity_sha256") != EXPECTED_CACHE_AUDIT_IDENTITY:
        raise RuntimeError("true-1mm cache audit identity drifted")
    counts = payload.get("counts", {})
    for name in ("pkl", "data_b2nd", "seg_b2nd", "gt_segmentations"):
        if counts.get(name) != EXPECTED_CACHE_COUNT:
            raise RuntimeError(f"true-1mm cache count drifted for {name}")
    audit_split = payload.get("split", {})
    expected_split = {
        "train": EXPECTED_TRAIN_COUNT,
        "val": EXPECTED_VAL_COUNT,
        "test_locked": 104,
        "train_fixed_sha256": split["train_sha256"],
        "val_fixed_sha256": split["val_sha256"],
    }
    for name, expected in expected_split.items():
        if audit_split.get(name) != expected:
            raise RuntimeError(f"true-1mm cache audit split drifted for {name}")
    if payload.get("e_checkpoint", {}).get("sha256") != EXPECTED_E_SHA256:
        raise RuntimeError("true-1mm cache audit is not bound to frozen E")
    plans = payload.get("plans", {})
    if plans.get("file_sha256") != EXPECTED_PLANS_FILE_SHA256:
        raise RuntimeError("true-1mm plans file SHA256 drifted in audit")
    if plans.get("canonical_sha256") != EXPECTED_PLANS_CANONICAL_SHA256:
        raise RuntimeError("true-1mm canonical plans SHA256 drifted")
    if plans.get("checkpoint_canonical_sha256") != EXPECTED_PLANS_CANONICAL_SHA256:
        raise RuntimeError("true-1mm plans no longer match frozen E")
    plans_path = nnunet_preprocessed / EXPECTED_DATASET / "nnUNetPlans.json"
    if sha256_file(_resolved_existing(str(plans_path), label="true-1mm plans", directory=False)) != EXPECTED_PLANS_FILE_SHA256:
        raise RuntimeError("true-1mm plans file content drifted")
    ready_marker = resolved.parent / "TRUE1MM_CACHE_READY.ok"
    _resolved_existing(str(ready_marker), label="true-1mm ready marker", directory=False)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "identity_sha256": payload["audit_identity_sha256"],
        "ready_marker": str(ready_marker),
        "ready_marker_sha256": sha256_file(ready_marker),
        "plans_file": str(plans_path),
        "plans_file_sha256": EXPECTED_PLANS_FILE_SHA256,
        "plans_canonical_sha256": EXPECTED_PLANS_CANONICAL_SHA256,
        "counts": counts,
    }


def _lock_environment(name: str, value: str) -> None:
    observed = os.environ.get(name)
    if observed is not None and observed != value:
        raise RuntimeError(f"environment drift for {name}: observed={observed!r}, expected={value!r}")
    os.environ[name] = value


def configure_environment(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    reject_generative_environment()
    nnunet_raw = _resolved_existing(args.nnunet_raw, label="nnUNet raw root", directory=True)
    nnunet_preprocessed = _resolved_existing(
        args.nnunet_preprocessed,
        label="nnUNet preprocessed root",
        directory=True,
    )
    _resolved_existing(str(nnunet_raw / EXPECTED_DATASET), label="Dataset264 raw dataset", directory=True)
    _resolved_existing(
        str(nnunet_preprocessed / EXPECTED_DATASET / "nnUNetPlans_3d_fullres"),
        label="Dataset264 true-1mm 3d_fullres cache",
        directory=True,
    )
    split = validate_split_dir(args.split_dir)
    pretrained = _resolved_existing(args.pretrained_weights, label="frozen E checkpoint", directory=False)
    if sha256_file(pretrained) != EXPECTED_E_SHA256:
        raise RuntimeError("E-continue smoke must warm-start from frozen E; SHA256 mismatch")
    cache_audit = validate_cache_audit(
        args.cache_audit,
        nnunet_preprocessed=nnunet_preprocessed,
        split=split,
    )
    if args.trainer != EXPECTED_TRAINER:
        raise RuntimeError(f"E-continue smoke is locked to trainer {EXPECTED_TRAINER}")
    if str(args.dataset) != "264" or args.fold != 0:
        raise RuntimeError("E-continue smoke is locked to Dataset264 fold 0")
    if args.configuration != "3d_fullres" or args.plans != "nnUNetPlans":
        raise RuntimeError("E-continue smoke is locked to nnUNetPlans/3d_fullres")

    smoke_results = (output_dir / "nnUNet_results").resolve()
    forced = {
        "BRATS_S2_REPO_DIR": str(REPOSITORY_ROOT),
        "BRATS_SPLIT_DIR": split["root"],
        "nnUNet_extTrainer": str((REPOSITORY_ROOT / "custom_nnunet").resolve()),
        "nnUNet_raw": str(nnunet_raw),
        "nnUNet_preprocessed": str(nnunet_preprocessed),
        "nnUNet_results": str(smoke_results),
        "S2_EXPERIMENT_MODE": "met_aug_route_a_control",
    }
    locked = {
        "S2_MET_AUG_ENABLE": "0",
        "S2_PAIRED_TRAINING_SEED": "20260724",
        "S2_COMPLETION_EPOCHS": "200",
        "S2_COMPLETION_INITIAL_LR": "0.001",
        "S2_COMPLETION_SAVE_EVERY": "25",
        "S2_FOCAL_GAMMA": "2.0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "nnUNet_compile": "0",
        "nnUNet_n_proc_DA": "0",
    }
    for name, value in forced.items():
        os.environ[name] = value
    for name, value in locked.items():
        _lock_environment(name, value)
    return {
        "nnunet_raw": str(nnunet_raw),
        "nnunet_preprocessed": str(nnunet_preprocessed),
        "smoke_results": str(smoke_results),
        "split": split,
        "pretrained_weights": {"path": str(pretrained), "sha256": EXPECTED_E_SHA256},
        "cache_audit": cache_audit,
        "forbidden_generative_environment": list(FORBIDDEN_GENERATIVE_ENVIRONMENT),
        "forced_isolation_environment": forced,
        "locked_training_environment": locked,
    }


def summarize_steps(
    steps: list[Mapping[str, Any]],
    *,
    iterations_per_epoch: int,
    eta_safety_factor: float,
) -> dict[str, Any]:
    if len(steps) < 4:
        raise ValueError("E-continue timing requires at least four real train steps")
    steady = steps[2:]
    totals = [float(row["total_seconds"]) for row in steady]
    median_total = statistics.median(totals)
    raw_hours = median_total * int(iterations_per_epoch) * 200 / 3600
    return {
        "steps": len(steps),
        "warmup_steps_excluded": 2,
        "steady_batch_load_median_seconds": statistics.median(
            float(row["batch_load_seconds"]) for row in steady
        ),
        "steady_train_step_median_seconds": statistics.median(
            float(row["train_step_seconds"]) for row in steady
        ),
        "steady_total_median_seconds": median_total,
        "steady_total_min_seconds": min(totals),
        "steady_total_max_seconds": max(totals),
        "iterations_per_epoch": int(iterations_per_epoch),
        "estimated_epoch_seconds": median_total * int(iterations_per_epoch),
        "estimated_200_epochs_hours": raw_hours,
        "eta_safety_factor": float(eta_safety_factor),
        "estimated_200_epochs_hours_conservative": raw_hours * float(eta_safety_factor),
        "estimate_warning": "Short H20 smoke estimate; validation and checkpoint I/O are excluded and covered by the safety factor.",
    }


def _cuda_snapshot(torch_module) -> dict[str, float]:
    torch_module.cuda.synchronize()
    gib = float(1024**3)
    return {
        "allocated_gib": torch_module.cuda.memory_allocated() / gib,
        "reserved_gib": torch_module.cuda.memory_reserved() / gib,
        "peak_allocated_gib": torch_module.cuda.max_memory_allocated() / gib,
        "peak_reserved_gib": torch_module.cuda.max_memory_reserved() / gib,
    }


def _shutdown_dataloaders(trainer: Any) -> None:
    for name in ("dataloader_train", "dataloader_val"):
        dataloader = getattr(trainer, name, None)
        finish = getattr(dataloader, "_finish", None)
        if callable(finish):
            finish()


def run_smoke(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    runtime = configure_environment(args, output_dir)

    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "E-continue smoke requires exactly one scheduler-visible CUDA GPU; "
            f"observed={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    total_memory_gib = float(properties.total_memory) / float(1024**3)
    if total_memory_gib < float(args.min_gpu_memory_gib):
        raise RuntimeError(
            f"GPU memory gate failed: observed={total_memory_gib:.3f} GiB, "
            f"required={args.min_gpu_memory_gib:.3f} GiB"
        )
    torch.cuda.reset_peak_memory_stats(device)
    memory_before_trainer = _cuda_snapshot(torch)

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from custom_nnunet.nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune import (
        nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune,
    )
    from nnunetv2.run.run_training import maybe_load_checkpoint

    trainer = None
    started_at = utc_now()
    try:
        preprocessed_dataset = Path(runtime["nnunet_preprocessed"]) / EXPECTED_DATASET
        plans = json.loads((preprocessed_dataset / f"{args.plans}.json").read_text(encoding="utf-8"))
        plans["continue_training"] = False
        dataset_json = json.loads((preprocessed_dataset / "dataset.json").read_text(encoding="utf-8"))
        trainer = nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune(
            plans=plans,
            configuration=args.configuration,
            fold=args.fold,
            dataset_json=dataset_json,
            device=device,
        )
        trainer.disable_checkpointing = True
        trainer_output = Path(trainer.output_folder).resolve()
        expected_result_root = Path(runtime["smoke_results"])
        if expected_result_root not in trainer_output.parents:
            raise RuntimeError(f"trainer escaped isolated smoke result root: {trainer_output}")
        maybe_load_checkpoint(
            trainer,
            False,
            False,
            runtime["pretrained_weights"]["path"],
        )
        memory_after_pretrained = _cuda_snapshot(torch)

        start_begin = perf_counter()
        trainer.on_train_start()
        torch.cuda.synchronize()
        on_train_start_seconds = perf_counter() - start_begin
        memory_after_train_start = _cuda_snapshot(torch)
        trainer.on_epoch_start()
        trainer.on_train_epoch_start()

        steps: list[dict[str, Any]] = []
        train_outputs: list[dict[str, Any]] = []
        for step_index in range(1, args.steps + 1):
            torch.cuda.synchronize()
            batch_begin = perf_counter()
            batch = next(trainer.dataloader_train)
            torch.cuda.synchronize()
            train_begin = perf_counter()
            output = trainer.train_step(batch)
            torch.cuda.synchronize()
            step_end = perf_counter()
            loss = float(np.asarray(output["loss"], dtype=np.float64).mean())
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at smoke step {step_index}: {loss}")
            train_outputs.append(output)
            step = {
                "step": step_index,
                "batch_load_seconds": train_begin - batch_begin,
                "train_step_seconds": step_end - train_begin,
                "total_seconds": step_end - batch_begin,
                "loss": loss,
                "cuda": _cuda_snapshot(torch),
            }
            steps.append(step)
            print(json.dumps(step, ensure_ascii=True, sort_keys=True), flush=True)

        trainer.on_train_epoch_end(train_outputs)
        timing = summarize_steps(
            steps,
            iterations_per_epoch=int(trainer.num_iterations_per_epoch),
            eta_safety_factor=args.eta_safety_factor,
        )
        checkpoint_files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*.pth"))
        validation_files = sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file() and "validation" in path.parts
        )
        failures: list[str] = []
        if checkpoint_files:
            failures.append("smoke_created_checkpoint")
        if validation_files:
            failures.append("smoke_executed_validation")
        if timing["estimated_200_epochs_hours_conservative"] > args.max_estimated_training_hours:
            failures.append(
                "estimated_training_too_slow:"
                f"{timing['estimated_200_epochs_hours_conservative']:.6f}>"
                f"{args.max_estimated_training_hours:.6f}"
            )
        if memory_after_train_start["peak_allocated_gib"] <= 0:
            failures.append("no_cuda_allocation_observed")
        report: dict[str, Any] = {
            "schema_version": SMOKE_REPORT_SCHEMA,
            "status": "pass" if not failures else "fail",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "dataset": str(args.dataset),
            "configuration": args.configuration,
            "fold": args.fold,
            "plans": args.plans,
            "trainer": args.trainer,
            "augmentation_probability": 0.0,
            "device": {
                "name": torch.cuda.get_device_name(device),
                "total_memory_gib": total_memory_gib,
                "minimum_required_gib": float(args.min_gpu_memory_gib),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            },
            "runtime": runtime,
            "on_train_start_seconds": on_train_start_seconds,
            "memory": {
                "before_trainer": memory_before_trainer,
                "after_pretrained": memory_after_pretrained,
                "after_on_train_start": memory_after_train_start,
                "final": _cuda_snapshot(torch),
            },
            "steps": steps,
            "timing": timing,
            "checkpoint_files": checkpoint_files,
            "validation_files": validation_files,
            "validation_executed": False,
            "checkpoint_saved": False,
            "generative_assets_loaded": False,
            "failures": failures,
            "smoke_script_sha256": sha256_file(__file__),
        }
        report["report_sha256"] = canonical_sha256(report, exclude=("report_sha256",))
        return report
    finally:
        if trainer is not None:
            _shutdown_dataloaders(trainer)
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.steps < 4:
        raise ValueError("E-continue smoke requires --steps >= 4")
    if args.min_gpu_memory_gib <= 0:
        raise ValueError("min-gpu-memory-gib must be positive")
    if args.eta_safety_factor < 1:
        raise ValueError("eta-safety-factor must be at least 1")
    if args.max_estimated_training_hours <= 0:
        raise ValueError("max-estimated-training-hours must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"E-continue smoke output is immutable and already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "e_continue_training_smoke_report.json"
    exit_code = 0
    try:
        report = run_smoke(args, output_dir)
        if report["status"] != "pass":
            exit_code = 1
    except Exception as exc:
        report = {
            "schema_version": SMOKE_REPORT_SCHEMA,
            "status": "fail",
            "finished_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "validation_executed": False,
            "checkpoint_saved": False,
            "generative_assets_loaded": False,
            "smoke_script_sha256": sha256_file(__file__),
        }
        report["report_sha256"] = canonical_sha256(report, exclude=("report_sha256",))
        exit_code = 1
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=True, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
