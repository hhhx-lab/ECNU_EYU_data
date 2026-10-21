#!/usr/bin/env python3
"""Fail-closed remote preflight for the five-route selected103 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


EXPECTED_ROOT = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/"
    "experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1"
)
SCREEN_ROOT = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/"
    "s2_met_aug_btof_screen125_20260831_r1"
)
OLD_ENDPOINT_ROOT = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/"
    "experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2"
)
PYTHON = Path("/public/home/zqchen/.conda/envs/segmamba/bin/python")
EVAL_PYTHON = Path("/public/home/zqchen/.conda/envs/brats_eval/bin/python")
TRAINER = "nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune"
DATASET = "Dataset264_BraTS2026_MET_Completion"
MODEL_DIR = f"{TRAINER}__nnUNetPlans__3d_fullres"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_path(route: str, epoch: int) -> Path:
    return (
        SCREEN_ROOT
        / "results"
        / route
        / DATASET
        / MODEL_DIR
        / "fold_0"
        / f"checkpoint_epoch_{epoch:04d}.pth"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == EXPECTED_ROOT, f"unexpected root: {root}")
    passed = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    failed = root / "evidence/REMOTE_STATIC_PREFLIGHT_FAILED.json"
    require(not passed.exists() and not failed.exists(), "preflight evidence already exists")

    try:
        subprocess.run(
            ["sha256sum", "-c", "STATIC_SHA256SUMS.txt"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        plan_path = root / "SELECTED103_OFFICIAL_MET_EXECUTION_PLAN_20260903_R1.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        require(plan["experiment_id"] == root.name, "experiment ID drift")
        require(plan["artifact_status"] == "experimental_unvalidated", "artifact status drift")
        require(plan["operator_approved"] is False, "operator approval drift")
        require(plan["formal_gate_status"] == "not_run_not_passed", "formal gate drift")
        require(plan["trainer"] == TRAINER, "trainer contract drift")
        require(plan["submission_contract"]["route_order"] == ["B", "C40", "D", "E_SNFH", "F"], "route order drift")
        require(plan["bootstrap"] == {"paired_by_case": True, "replicates": 20000, "seed": 20260831}, "bootstrap drift")

        original_plan = SCREEN_ROOT / "SCREEN125_EXECUTION_PLAN_20260831_R1.json"
        require(
            sha256_file(original_plan) == plan["original_screen_plan_sha256"],
            "original Screen125 plan SHA drift",
        )
        manifest = root / "manifests/FIXED103_REAL_T2W_CASES.csv"
        require(sha256_file(manifest) == plan["case_contract"]["manifest_sha256"], "manifest SHA drift")
        with manifest.open(encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        require(len(manifest_rows) == 103, "manifest case count drift")
        require(len({row["source_case_id"] for row in manifest_rows}) == 103, "duplicate manifest cases")
        require(all(row["source_t2w_allowed"].lower() == "true" for row in manifest_rows), "real T2W policy drift")
        require(
            (OLD_ENDPOINT_ROOT / "evidence/FIXED103_PAIRED_VALIDATION.json").is_file(),
            "previous fixed103 validation evidence missing",
        )

        checkpoint_audits = {}
        for route in plan["submission_contract"]["route_order"]:
            expected = plan["routes"][route]
            epoch = int(expected["checkpoint_epoch"])
            source = checkpoint_path(route, epoch)
            require(source.is_file(), f"missing checkpoint: {source}")
            observed_sha = sha256_file(source)
            require(observed_sha == expected["checkpoint_sha256"], f"{route} checkpoint SHA drift")
            training_validation_path = SCREEN_ROOT / f"evidence/{route}_TRAINING_VALIDATION.json"
            training_complete = SCREEN_ROOT / f"evidence/{route}_TRAINING_COMPLETE.ok"
            require(training_validation_path.is_file() and training_complete.is_file(), f"{route} training evidence missing")
            training_validation = json.loads(training_validation_path.read_text(encoding="utf-8"))
            require(training_validation["status"] == "pass", f"{route} training validation failed")
            best = training_validation["best_checkpoint"]
            require(int(best["epoch"]) == epoch, f"{route} selected epoch drift")
            require(best["sha256"] == observed_sha, f"{route} selected SHA drift")
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            require(int(checkpoint.get("current_epoch", -1)) == epoch, f"{route} current_epoch drift")
            require(checkpoint.get("trainer_name") == TRAINER, f"{route} trainer drift")
            require(isinstance(checkpoint.get("init_args"), dict), f"{route} checkpoint init_args missing")
            snapshot = root / "model_snapshots" / route
            subprocess.run(
                [
                    str(PYTHON),
                    str(root / "runtime/assemble_model_snapshot.py"),
                    "--checkpoint",
                    str(source),
                    "--output-root",
                    str(snapshot),
                    "--expected-sha256",
                    observed_sha,
                    "--expected-trainer",
                    TRAINER,
                    "--model-name",
                    route,
                ],
                check=True,
            )
            checkpoint_audits[route] = {
                "epoch": epoch,
                "checkpoint_sha256": observed_sha,
                "source": str(source),
                "snapshot": str(snapshot),
            }

        cohort = root / "cohort/fixed103"
        subprocess.run(
            [
                str(PYTHON),
                str(root / "runtime/materialize_s2_inputs.py"),
                "--manifest",
                str(manifest),
                "--source-root",
                "/public/home/zqchen/data",
                "--route",
                "real",
                "--input-root",
                str(cohort / "input"),
                "--reference-root",
                str(cohort / "reference"),
                "--audit-json",
                str(cohort / "input_audit.json"),
                "--audit-csv",
                str(cohort / "input_audit.csv"),
                "--expected-count",
                "103",
            ],
            check=True,
        )
        input_audit = json.loads((cohort / "input_audit.json").read_text(encoding="utf-8"))
        require(input_audit["status"] == "pass" and input_audit["case_count"] == 103, "input audit failed")
        require(input_audit["source_t2w_used"] is True, "fixed103 must use authentic T2W")
        require(input_audit["geometry_mismatch_count"] == 0, "geometry mismatch detected")
        require(input_audit["geometry_repaired_count"] == 0, "unexpected geometry repair")

        environment_json = subprocess.check_output(
            [
                str(EVAL_PYTHON),
                "-c",
                (
                    "import importlib.metadata as m,json;"
                    "print(json.dumps({x:m.version(x) for x in "
                    "('BraTS-evaluation','panoptica','numpy')},sort_keys=True))"
                ),
            ],
            text=True,
        ).strip()
        environment = json.loads(environment_json)
        require(
            environment == {"BraTS-evaluation": "0.0.8", "numpy": "1.26.4", "panoptica": "2.1.0"},
            f"official MET environment drift: {environment}",
        )
        require((Path("/public/home/zqchen/.conda/envs/brats_eval/bin/brats-evaluate")).is_file(), "brats-evaluate missing")
        require((Path("/public/home/zqchen/.conda/envs/brats_eval/bin/brats-parse-metrics")).is_file(), "brats-parse-metrics missing")

        payload = {
            "schema_version": 1,
            "status": "pass",
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "execution_plan_sha256": sha256_file(plan_path),
            "original_screen_plan_sha256": sha256_file(original_plan),
            "manifest_sha256": sha256_file(manifest),
            "case_count": 103,
            "input_file_count": 412,
            "reference_file_count": 103,
            "checkpoint_audits": checkpoint_audits,
            "official_met_environment": environment,
            "deep_validated": 5 + 103,
            "locked_test_used": False,
        }
        write_exclusive(passed, payload)
        print(json.dumps(payload, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if not failed.exists():
            write_exclusive(failed, failure)
        raise


if __name__ == "__main__":
    main()
