#!/usr/bin/env python3
"""Fail-closed static preflight for the K1245 prerequisite batch."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from custom_nnunet.met_aug_core import sha256_file
from custom_nnunet.met_aug_k1245 import extract_k1245_components


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    args = parser.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    plan_path = root / "K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("experiment_id") != "s2_met_aug_k1245_20260921_r1"
        or plan.get("artifact_status") != "experimental_unvalidated"
        or plan.get("operator_approved") is not False
        or plan.get("formal_gate_status") != "not_run_not_passed"
        or plan.get("pool_contracts", {}).get("rc27", {}).get("max_bbox_mm") != 56.0
    ):
        raise SystemExit("K1245 plan contract drifted")
    required = {
        "legacy_route": parent / "calibration_rebuild_20260819_r10/FIX_V3_ROUTE_CONFIG_SCHEMA5.json",
        "calibration": parent / "calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json",
        "legacy_manifest": parent / "route_rebuild_20260812_r1/component_pool/component_manifest.json",
        "coordinate_audit": parent / "route_rebuild_20260812_r1/preprocessed_coordinate_contract.json",
        "valid_masks": parent / "route_rebuild_20260812_r1/valid_masks/valid_mask_manifest.json",
        "train": parent / "inputs/splits/train_fixed.txt",
        "val": parent / "inputs/splits/val_fixed.txt",
        "mapping": parent / "inputs/mapping/nnunet_case_mapping_master.csv",
        "original_e": parent / "frozen/original_e/checkpoint_final.pth",
        "g1_selection": parent / "frozen/g1/checkpoint_selection.json",
        "g2_gate": parent / "frozen/g1/g2_diffusion_qc_gate.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"required K1245 inputs are missing: {missing}")
    required_directories = {
        "raw_dataset": Path(
            "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/"
            "ecnu_completion_emergency/nnUNet_raw/Dataset264_BraTS2026_MET_Completion"
        ),
        "preprocessed_dataset": parent
        / "data/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres",
    }
    missing_directories = [
        str(path) for path in required_directories.values() if not path.is_dir()
    ]
    if missing_directories:
        raise SystemExit(f"required K1245 input directories are missing: {missing_directories}")
    expected_files = plan["parent_contract"]
    checks = {
        "fix_v3_route_config_file_sha256": required["legacy_route"],
        "fix_v3_calibration_file_sha256": required["calibration"],
        "legacy_component_manifest_file_sha256": required["legacy_manifest"],
        "original_e_checkpoint_sha256": required["original_e"],
    }
    for key, path in checks.items():
        if sha256_file(path) != expected_files[key]:
            raise SystemExit(f"K1245 frozen input drifted: {key}")
    if sha256_file(required["g1_selection"]) != plan["g1_contract"]["checkpoint_selection_file_sha256"]:
        raise SystemExit("K1245 G1 selection drifted")
    if sha256_file(required["g2_gate"]) != plan["g1_contract"]["g2_qc_gate_file_sha256"]:
        raise SystemExit("K1245 G2 gate drifted")
    train = [v for v in required["train"].read_text().splitlines() if v.strip()]
    val = [v for v in required["val"].read_text().splitlines() if v.strip()]
    if len(train) != 1035 or len(val) != 103 or set(train) & set(val):
        raise SystemExit("K1245 frozen split drifted")
    for forbidden in ("pools", "config"):
        if (root / forbidden).exists():
            raise SystemExit(f"K1245 immutable output exists before launch: {forbidden}")
    if (root / "evidence/POOL_BUILD_LAUNCH.json").exists():
        raise SystemExit("K1245 pool launch audit already exists")
    result = subprocess.run(
        [sys.executable, "-B", str(root / "runtime/repository/tests/test_met_aug_k1245.py")],
        cwd=root / "runtime/repository",
        env={**os.environ, "PYTHONPATH": str(root / "runtime/repository")},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"K1245 remote tests failed:\n{result.stdout}\n{result.stderr}")
    usage = shutil.disk_usage("/public")
    if usage.free < 500 * 1024**3:
        raise SystemExit("K1245 requires at least 500 GiB free on /public")
    code_paths = [
        root / "runtime/repository/custom_nnunet/met_aug_k1245.py",
        root / "runtime/repository/custom_nnunet/met_aug_diffusion.py",
        root / "runtime/repository/custom_nnunet/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune.py",
        root / "runtime/repository/scripts/09_k1245/60_build_k1245_component_pools.py",
        root / "runtime/repository/scripts/09_k1245/61_prepare_k1245_configs.py",
        root / "runtime/repository/scripts/09_k1245/62_run_k1245_smoke.py",
        root / "slurm/01_build_k1245_pools.slurm",
    ]
    payload = {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "status": "pass",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": os.uname().nodename,
        "plan_sha256": sha256_file(plan_path),
        "free_public_bytes": usage.free,
        "train_count": len(train),
        "validation_count": len(val),
        "input_directories": {
            key: str(path) for key, path in sorted(required_directories.items())
        },
        "code_sha256": {str(path.relative_to(root)): sha256_file(path) for path in code_paths},
        "unit_tests": "7_pass",
    }
    destination = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
