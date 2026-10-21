#!/usr/bin/env python3
"""Fail-closed static preflight before any B-F screening sbatch submission."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil

from custom_nnunet.met_aug_core import ComponentManifest, RouteConfig, sha256_file
from custom_nnunet.met_aug_screening import load_screening_policy
from custom_nnunet.nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune import (
    nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    args = parser.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    plan = root / "SCREEN125_EXECUTION_PLAN_20260831_R1.json"
    base_config_path = parent / "calibration_rebuild_20260819_r10/FIX_V3_ROUTE_CONFIG_SCHEMA5.json"
    manifest_path = parent / "route_rebuild_20260812_r1/component_pool/component_manifest.json"
    counts_path = root / "config/TRAIN_NATIVE_LESION_COUNTS.json"
    required = [
        plan,
        base_config_path,
        manifest_path,
        counts_path,
        parent / "calibration_rebuild_20260819_r10/FIX_V3_EMERGENCY_DECISION.json",
        parent / "calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json",
        parent / "frozen/original_e/checkpoint_final.pth",
        parent / "frozen/fix_v2/QC_HOLDOUT_RUN_ATTEMPT_02_FAILED.json",
        parent / "inputs/splits/train_fixed.txt",
        parent / "inputs/splits/val_fixed.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"required frozen assets are missing: {missing}")
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    if (
        plan_payload.get("artifact_status") != "experimental_unvalidated"
        or plan_payload.get("operator_approved") is not False
        or plan_payload.get("formal_gate_status") != "not_run_not_passed"
        or plan_payload.get("training_contract", {}).get("epochs") != 125
        or plan_payload.get("training_contract", {}).get("checkpoint_epochs") != [25, 50, 75, 100, 125]
    ):
        raise SystemExit("screening plan contract drifted")
    train_ids = [line.strip() for line in (parent / "inputs/splits/train_fixed.txt").read_text().splitlines() if line.strip()]
    val_ids = [line.strip() for line in (parent / "inputs/splits/val_fixed.txt").read_text().splitlines() if line.strip()]
    if len(train_ids) != 1035 or len(val_ids) != 103 or set(train_ids) & set(val_ids):
        raise SystemExit("frozen 1035/103 split drifted")
    original_e = parent / "frozen/original_e/checkpoint_final.pth"
    if sha256_file(original_e) != "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267":
        raise SystemExit("original E checkpoint SHA256 drifted")
    manifest = ComponentManifest.load(manifest_path)
    base_config = RouteConfig.load(base_config_path, manifest)
    tokens = ("B", "C40", "D", "E_SNFH", "F")
    route_ids = []
    config_shas = {}
    for token in tokens:
        config_path = root / f"config/SCREEN_ROUTE_{token}.json"
        policy = load_screening_policy(
            config_path,
            base_config=base_config,
            manifest=manifest,
            target_counts_path=counts_path,
        )
        route_ids.append(policy.route_id)
        config_shas[token] = sha256_file(config_path)
        for suffix in ("TRAINING_STARTED.json", "TRAINING_FAILED.json", "TRAINING_VALIDATION.json", "TRAINING_COMPLETE.ok"):
            if (root / "evidence" / f"{token}_{suffix}").exists():
                raise SystemExit(f"route evidence already exists before launch: {token}_{suffix}")
        if (root / "results" / token).exists():
            raise SystemExit(f"route result root already exists before launch: {token}")
    expected_routes = {"MET-AUG-B", "MET-AUG-C40", "MET-AUG-D", "MET-AUG-E-SNFH", "MET-AUG-F"}
    if set(route_ids) != expected_routes:
        raise SystemExit(f"screening route set drifted: {route_ids}")
    if nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune.__name__ != "nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune":
        raise SystemExit("screening trainer import drifted")
    usage = shutil.disk_usage("/public")
    if usage.free < 500 * 1024**3:
        raise SystemExit(f"insufficient /public free space: {usage.free}")
    code_files = [
        root / "runtime/repository/custom_nnunet/met_aug_screening.py",
        root / "runtime/repository/custom_nnunet/nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune.py",
        root / "runtime/prepare_screening_configs.py",
        root / "runtime/run_remote_static_preflight.py",
        root / "runtime/launch_five_screening_jobs.py",
        root / "slurm/01_train_screen125_route.slurm",
    ]
    payload = {
        "artifact_status": "experimental_unvalidated",
        "base_route_config_sha256": sha256_file(base_config_path),
        "code_sha256": {path.name: sha256_file(path) for path in code_files},
        "config_sha256": config_shas,
        "formal_gate_status": "not_run_not_passed",
        "free_public_bytes": usage.free,
        "operator_approved": False,
        "original_e_sha256": sha256_file(original_e),
        "plan_sha256": sha256_file(plan),
        "route_ids": route_ids,
        "status": "pass",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_count": len(train_ids),
        "trainer": nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune.__name__,
        "validation_count": len(val_ids),
    }
    destination = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
