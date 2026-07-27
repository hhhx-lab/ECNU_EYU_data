#!/usr/bin/env python3
"""Run an immutable Route A training-path throughput and CUDA-memory smoke."""

from __future__ import annotations

import argparse
from collections import Counter
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
EXPECTED_TRAINER = "nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune"
EXPECTED_DATASET = "Dataset264_BraTS2026_MET_Completion"
EXPECTED_E_SHA256 = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
EXPECTED_TRAIN_COUNT = 1035
EXPECTED_VAL_COUNT = 103
SMOKE_REPORT_SCHEMA = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    parser.add_argument("--nnunet-preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--pretrained-weights", required=True)
    parser.add_argument("--component-manifest", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--valid-mask-manifest", required=True)
    parser.add_argument("--route-approval", required=True)
    parser.add_argument("--g1-code-dir", required=True)
    parser.add_argument("--g1-checkpoint-root", required=True)
    parser.add_argument("--g1-checkpoint-selection", required=True)
    parser.add_argument("--g2-parent-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="264")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--trainer", default=EXPECTED_TRAINER)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--min-committed-events", type=int, default=1)
    parser.add_argument("--min-gpu-memory-gib", type=float, default=30.0)
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
    if len(train_ids) != EXPECTED_TRAIN_COUNT or len(val_ids) != EXPECTED_VAL_COUNT:
        raise RuntimeError(
            "Route A smoke requires the fixed Dataset264 1035/103 split: "
            f"observed={len(train_ids)}/{len(val_ids)}"
        )
    if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
        raise RuntimeError("fixed split contains duplicate case IDs")
    if set(train_ids) & set(val_ids):
        raise RuntimeError("fixed training and validation splits overlap")
    return {
        "root": str(root),
        "train_file": str(train_file),
        "train_count": len(train_ids),
        "train_sha256": sha256_file(train_file),
        "val_file": str(val_file),
        "val_count": len(val_ids),
        "val_sha256": sha256_file(val_file),
    }


def _lock_environment(name: str, value: str) -> None:
    observed = os.environ.get(name)
    if observed is not None and observed != value:
        raise RuntimeError(f"environment drift for {name}: observed={observed!r}, expected={value!r}")
    os.environ[name] = value


def configure_environment(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    nnunet_raw = _resolved_existing(args.nnunet_raw, label="nnUNet raw root", directory=True)
    nnunet_preprocessed = _resolved_existing(
        args.nnunet_preprocessed,
        label="nnUNet preprocessed root",
        directory=True,
    )
    split = validate_split_dir(args.split_dir)
    files = {
        "pretrained_weights": _resolved_existing(args.pretrained_weights, label="E checkpoint", directory=False),
        "component_manifest": _resolved_existing(args.component_manifest, label="component manifest", directory=False),
        "route_config": _resolved_existing(args.route_config, label="Route A config", directory=False),
        "valid_mask_manifest": _resolved_existing(
            args.valid_mask_manifest,
            label="valid-mask manifest",
            directory=False,
        ),
        "route_approval": _resolved_existing(args.route_approval, label="Route A approval", directory=False),
        "g1_checkpoint_selection": _resolved_existing(
            args.g1_checkpoint_selection,
            label="G1 checkpoint selection",
            directory=False,
        ),
        "g2_parent_gate": _resolved_existing(args.g2_parent_gate, label="G2 parent gate", directory=False),
    }
    directories = {
        "g1_code_dir": _resolved_existing(args.g1_code_dir, label="G1 code", directory=True),
        "g1_checkpoint_root": _resolved_existing(
            args.g1_checkpoint_root,
            label="G1 checkpoint root",
            directory=True,
        ),
    }
    e_sha256 = sha256_file(files["pretrained_weights"])
    if e_sha256 != EXPECTED_E_SHA256:
        raise RuntimeError(
            "training smoke must warm-start from the frozen E checkpoint: "
            f"observed={e_sha256}, expected={EXPECTED_E_SHA256}"
        )
    if args.trainer != EXPECTED_TRAINER:
        raise RuntimeError(f"training smoke is locked to trainer {EXPECTED_TRAINER}")
    if str(args.dataset) != "264":
        raise RuntimeError("training smoke is locked to Dataset264 via --dataset 264")
    if args.configuration != "3d_fullres" or args.plans != "nnUNetPlans":
        raise RuntimeError(
            "training smoke is locked to configuration=3d_fullres and plans=nnUNetPlans"
        )
    if args.fold != 0:
        raise RuntimeError("training smoke is locked to fold 0")
    _resolved_existing(
        str(nnunet_raw / EXPECTED_DATASET),
        label="Dataset264 raw dataset",
        directory=True,
    )
    _resolved_existing(
        str(nnunet_preprocessed / EXPECTED_DATASET / "nnUNetPlans_3d_fullres"),
        label="Dataset264 3d_fullres preprocessed dataset",
        directory=True,
    )

    smoke_results = (output_dir / "nnUNet_results").resolve()
    audit_path = (output_dir / "met_aug_training_smoke_events.jsonl").resolve()
    forced = {
        "BRATS_S2_REPO_DIR": str(REPOSITORY_ROOT),
        "BRATS_SPLIT_DIR": split["root"],
        "nnUNet_extTrainer": str((REPOSITORY_ROOT / "custom_nnunet").resolve()),
        "nnUNet_raw": str(nnunet_raw),
        "nnUNet_preprocessed": str(nnunet_preprocessed),
        "nnUNet_results": str(smoke_results),
        "S2_MET_AUG_AUDIT_PATH": str(audit_path),
    }
    locked = {
        "S2_MET_AUG_ENABLE": "1",
        "S2_MET_AUG_COMPONENT_MANIFEST": str(files["component_manifest"]),
        "S2_MET_AUG_ROUTE_CONFIG": str(files["route_config"]),
        "S2_MET_AUG_VALID_MASK_MANIFEST": str(files["valid_mask_manifest"]),
        "S2_MET_AUG_ROUTE_GATE": str(files["route_approval"]),
        "S2_MET_AUG_G1_CODE_DIR": str(directories["g1_code_dir"]),
        "S2_MET_AUG_G1_CHECKPOINT_ROOT": str(directories["g1_checkpoint_root"]),
        "S2_MET_AUG_G1_CHECKPOINT_SELECTION": str(files["g1_checkpoint_selection"]),
        "S2_MET_AUG_G2_QC_GATE": str(files["g2_parent_gate"]),
        "S2_PAIRED_TRAINING_SEED": "20260724",
        "S2_COMPLETION_EPOCHS": "200",
        "S2_COMPLETION_INITIAL_LR": "0.001",
        "S2_COMPLETION_SAVE_EVERY": "25",
        "S2_FOCAL_GAMMA": "2.0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "nnUNet_compile": "0",
    }
    for name, value in forced.items():
        os.environ[name] = value
    for name, value in locked.items():
        _lock_environment(name, value)
    return {
        "nnunet_raw": str(nnunet_raw),
        "nnunet_preprocessed": str(nnunet_preprocessed),
        "smoke_results": str(smoke_results),
        "audit_path": str(audit_path),
        "split": split,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "directories": {name: str(path) for name, path in directories.items()},
        "forced_isolation_environment": forced,
        "locked_training_environment": locked,
    }


def read_audit_events(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("state"):
            raise RuntimeError(f"invalid MET-AUG audit row at line {line_number}")
        events.append(value)
    return events


def summarize_audit_events(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    states = Counter(str(event.get("state", "")) for event in events)
    reasons = Counter(str(event.get("reason")) for event in events if event.get("reason") is not None)
    return {
        "event_count": len(events),
        "committed_events": states.get("COMMITTED", 0),
        "state_counts": dict(sorted(states.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def summarize_steps(steps: list[Mapping[str, Any]], *, iterations_per_epoch: int) -> dict[str, Any]:
    if not steps:
        raise ValueError("cannot summarize an empty training smoke")
    steady = steps[1:] if len(steps) > 1 else steps

    def values(key: str, rows: list[Mapping[str, Any]]) -> list[float]:
        return [float(row[key]) for row in rows]

    steady_total = values("total_seconds", steady)
    median_total = statistics.median(steady_total)
    return {
        "steps": len(steps),
        "first_step_seconds": float(steps[0]["total_seconds"]),
        "steady_batch_load_median_seconds": statistics.median(values("batch_load_seconds", steady)),
        "steady_train_step_median_seconds": statistics.median(values("train_step_seconds", steady)),
        "steady_total_median_seconds": median_total,
        "steady_total_min_seconds": min(steady_total),
        "steady_total_max_seconds": max(steady_total),
        "iterations_per_epoch": int(iterations_per_epoch),
        "estimated_epoch_seconds": median_total * int(iterations_per_epoch),
        "estimated_200_epochs_hours": median_total * int(iterations_per_epoch) * 200 / 3600,
        "estimate_warning": "Short smoke estimate; queue time, validation, checkpointing, and event-mix uncertainty are excluded.",
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

    if not torch.cuda.is_available():
        raise RuntimeError("Route A training smoke requires a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Route A training smoke requires exactly one scheduler-visible GPU; "
            f"observed={torch.cuda.device_count()}"
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

    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    trainer = None
    started_at = utc_now()
    try:
        trainer = get_trainer_from_args(
            str(args.dataset),
            args.configuration,
            args.fold,
            args.trainer,
            args.plans,
            False,
            device=device,
        )
        trainer.disable_checkpointing = True
        expected_result_root = Path(runtime["smoke_results"])
        trainer_output = Path(trainer.output_folder).resolve()
        if expected_result_root not in trainer_output.parents:
            raise RuntimeError(f"trainer escaped the isolated smoke result root: {trainer_output}")
        maybe_load_checkpoint(
            trainer,
            False,
            False,
            runtime["files"]["pretrained_weights"]["path"],
        )
        memory_after_pretrained = _cuda_snapshot(torch)

        on_train_start_begin = perf_counter()
        trainer.on_train_start()
        torch.cuda.synchronize()
        on_train_start_seconds = perf_counter() - on_train_start_begin
        memory_after_train_start = _cuda_snapshot(torch)
        trainer.on_epoch_start()
        trainer.on_train_epoch_start()

        steps: list[dict[str, Any]] = []
        train_outputs: list[dict[str, Any]] = []
        audit_path = Path(runtime["audit_path"])
        for step_index in range(1, args.max_steps + 1):
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
            audit = summarize_audit_events(read_audit_events(audit_path))
            step = {
                "step": step_index,
                "batch_load_seconds": train_begin - batch_begin,
                "train_step_seconds": step_end - train_begin,
                "total_seconds": step_end - batch_begin,
                "loss": loss,
                "audit_event_count": audit["event_count"],
                "committed_events": audit["committed_events"],
                "cuda": _cuda_snapshot(torch),
            }
            steps.append(step)
            print(json.dumps(step, ensure_ascii=True, sort_keys=True), flush=True)
            if step_index >= args.min_steps and audit["committed_events"] >= args.min_committed_events:
                break

        trainer.on_train_epoch_end(train_outputs)
        audit_events = read_audit_events(audit_path)
        audit_summary = summarize_audit_events(audit_events)
        failures: list[str] = []
        if len(steps) < args.min_steps:
            failures.append(f"insufficient_steps:{len(steps)}<{args.min_steps}")
        if audit_summary["committed_events"] < args.min_committed_events:
            failures.append(
                "insufficient_committed_events:"
                f"{audit_summary['committed_events']}<{args.min_committed_events}"
            )
        checkpoint_files = sorted(
            str(path.relative_to(output_dir)) for path in output_dir.rglob("*.pth")
        )
        if checkpoint_files:
            failures.append("smoke_created_checkpoint")
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
            "audit": audit_summary,
            "steps": steps,
            "timing": summarize_steps(
                steps,
                iterations_per_epoch=int(trainer.num_iterations_per_epoch),
            ),
            "checkpoint_files": checkpoint_files,
            "validation_executed": False,
            "checkpoint_saved": False,
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
    if args.min_steps <= 0 or args.max_steps < args.min_steps:
        raise ValueError("require 0 < min-steps <= max-steps")
    if args.min_committed_events <= 0:
        raise ValueError("min-committed-events must be positive")
    if args.min_gpu_memory_gib <= 0:
        raise ValueError("min-gpu-memory-gib must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"training smoke output is immutable and already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "met_aug_training_smoke_report.json"
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
