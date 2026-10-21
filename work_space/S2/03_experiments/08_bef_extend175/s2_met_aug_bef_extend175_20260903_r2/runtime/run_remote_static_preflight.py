#!/usr/bin/env python3
"""Fail-closed preflight for the independent B/E/F 125-to-175 extension."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import torch

from custom_nnunet.met_aug_core import sha256_file
from custom_nnunet.nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune import (
    EXTENSION_START_LR,
    nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune,
)


TOKENS = ("B", "C40", "D", "E_SNFH", "F")
EXTENDED = ("B", "E_SNFH", "F")
EPOCHS = (25, 50, 75, 100, 125)
SOURCE_JOBS = ("3554452", "3554453", "3554454", "3554455", "3554456")
DATASET = "Dataset264_BraTS2026_MET_Completion"
SOURCE_TRAINER = "nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune"


def all_finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() or value.is_complex() else True
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def write_exclusive(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("preflight JSON round-trip failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    plan_path = root / "EXTEND175_EXECUTION_PLAN_20260903_R2.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    contract = plan.get("training_contract", {})
    if (
        plan.get("artifact_status") != "experimental_unvalidated"
        or plan.get("operator_approved") is not False
        or plan.get("formal_gate_status") != "not_run_not_passed"
        or contract.get("source_total_epoch") != 125
        or contract.get("target_total_epoch") != 175
        or contract.get("extension_epochs") != 50
        or contract.get("extension_start_lr") != EXTENSION_START_LR
    ):
        raise SystemExit("extension plan contract drifted")
    if nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune.__name__ != "nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune":
        raise SystemExit("extension trainer import drifted")
    implementation = plan.get("implementation_sha256", {})
    code_paths = {
        "launch_three_extension_jobs.py": root / "runtime/launch_three_extension_jobs.py",
        "nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune.py": root / "runtime/repository/custom_nnunet/nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune.py",
        "run_remote_static_preflight.py": root / "runtime/run_remote_static_preflight.py",
        "01_train_extend175_route.slurm": root / "slurm/01_train_extend175_route.slurm",
    }
    for name, path in code_paths.items():
        if sha256_file(path) != implementation.get(name):
            raise SystemExit(f"extension implementation SHA256 drifted: {name}")
    source_plan = source / "SCREEN125_EXECUTION_PLAN_20260831_R1.json"
    if sha256_file(source_plan) != plan["parent_screen125"]["plan_sha256"]:
        raise SystemExit("source Screen125 plan drifted")
    accounting = subprocess.run(
        ["sacct", "-n", "-P", "-X", "-j", ",".join(SOURCE_JOBS), "--format=JobIDRaw,State,ExitCode"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    observed_jobs = {}
    for line in accounting:
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] in SOURCE_JOBS:
            observed_jobs[fields[0]] = (fields[1].split("+", 1)[0], fields[2])
    if any(observed_jobs.get(job) != ("COMPLETED", "0:0") for job in SOURCE_JOBS):
        raise SystemExit(f"source scheduler accounting failed: {observed_jobs}")

    route_plan = {row["route_token"]: row for row in plan["routes"]}
    checkpoint_count = 0
    source_rows = []
    for token in TOKENS:
        validation_path = source / "evidence" / f"{token}_TRAINING_VALIDATION.json"
        complete_path = source / "evidence" / f"{token}_TRAINING_COMPLETE.ok"
        failed_path = source / "evidence" / f"{token}_TRAINING_FAILED.json"
        if not validation_path.is_file() or not complete_path.is_file() or failed_path.exists():
            raise SystemExit(f"source route evidence is not complete: {token}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("status") != "pass" or validation.get("completed_epochs") != 125:
            raise SystemExit(f"source validation failed: {token}")
        fold = source / "results" / token / DATASET / f"{SOURCE_TRAINER}__nnUNetPlans__3d_fullres" / "fold_0"
        observed = []
        for epoch in EPOCHS:
            path = fold / f"checkpoint_epoch_{epoch:04d}.pth"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if int(checkpoint.get("current_epoch", -1)) != epoch:
                raise SystemExit(f"checkpoint epoch mismatch: {path}")
            if checkpoint.get("trainer_name") != SOURCE_TRAINER or not all_finite(checkpoint):
                raise SystemExit(f"checkpoint identity or finiteness failed: {path}")
            ema = checkpoint.get("logging", {}).get("ema_fg_dice", [])
            score = float(ema[epoch - 1])
            if not math.isfinite(score):
                raise SystemExit(f"nonfinite candidate EMA: {path}")
            expected_row = next(row for row in validation["candidate_checkpoints"] if row["epoch"] == epoch)
            if expected_row["sha256"] != sha256_file(path) or abs(float(expected_row["ema_fg_dice"]) - score) > 1e-15:
                raise SystemExit(f"candidate validation mismatch: {path}")
            observed.append((epoch, score))
            checkpoint_count += 1
        best_epoch, best_score = sorted(observed, key=lambda row: (-row[1], row[0]))[0]
        if validation["best_checkpoint"]["epoch"] != best_epoch or abs(float(validation["best_checkpoint"]["ema_fg_dice"]) - best_score) > 1e-15:
            raise SystemExit(f"best-checkpoint rule mismatch: {token}")
        if token in EXTENDED:
            route = route_plan[token]
            source_cp = fold / "checkpoint_epoch_0125.pth"
            if sha256_file(source_cp) != route["source_checkpoint_sha256"]:
                raise SystemExit(f"extension source SHA drifted: {token}")
            lrs = [float(group["lr"]) for group in torch.load(source_cp, map_location="cpu", weights_only=False)["optimizer_state"]["param_groups"]]
            if any(abs(value - EXTENSION_START_LR) > 1e-15 for value in lrs):
                raise SystemExit(f"extension source LR drifted: {token}: {lrs}")
            config = source / "config" / f"SCREEN_ROUTE_{token}.json"
            if sha256_file(config) != route["screen_config_sha256"]:
                raise SystemExit(f"source screen config drifted: {token}")
            source_rows.append({"route_token": token, "checkpoint_sha256": route["source_checkpoint_sha256"], "optimizer_lrs": lrs})

    for token in EXTENDED:
        for suffix in ("TRAINING_LAUNCH.json", "TRAINING_STARTED.json", "TRAINING_FAILED.json", "TRAINING_VALIDATION.json", "TRAINING_COMPLETE.ok"):
            if (root / "evidence" / f"{token}_EXTENSION_{suffix}").exists():
                raise SystemExit(f"extension evidence already exists: {token}_{suffix}")
        if (root / "results" / token).exists():
            raise SystemExit(f"extension result root already exists: {token}")
    queue = subprocess.run(
        ["squeue", "-h", "-n", "s2ab175,s2ae175,s2af175", "-o", "%i"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if queue:
        raise SystemExit(f"extension job name already exists: {queue}")
    usage = shutil.disk_usage("/public")
    if usage.free < 500 * 1024**3:
        raise SystemExit(f"insufficient /public free space: {usage.free}")
    payload = {
        "artifact_status": "experimental_unvalidated",
        "checkpoint_count_deep_validated": checkpoint_count,
        "implementation_sha256": {name: sha256_file(path) for name, path in code_paths.items()},
        "formal_gate_status": "not_run_not_passed",
        "free_public_bytes": usage.free,
        "operator_approved": False,
        "plan_sha256": sha256_file(plan_path),
        "source_scheduler_accounting": {job: list(state) for job, state in observed_jobs.items()},
        "source_plan_sha256": sha256_file(source_plan),
        "source_routes": source_rows,
        "status": "pass",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    write_exclusive(root / "evidence/REMOTE_STATIC_PREFLIGHT.json", payload)


if __name__ == "__main__":
    main()
