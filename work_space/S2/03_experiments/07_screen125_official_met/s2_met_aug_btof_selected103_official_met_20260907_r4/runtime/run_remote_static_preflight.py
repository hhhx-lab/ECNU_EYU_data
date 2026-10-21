#!/usr/bin/env python3
"""Fail-closed preflight for the C40/D shared-memory recovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ROOT = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/"
    "experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4"
)
R3 = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/"
    "experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r3"
)
R2 = Path(
    "/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/"
    "experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2"
)
PLAN_SHA = "1767372636d82e5d02bbd0e9d8ce4b982b1777a60c36c27b655c3b3b5b429c3b"
JOBS = {"B": "3652702", "C40": "3652703", "D": "3652704", "E_SNFH": "3652705", "F": "3652706"}


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
        checksum_output = subprocess.check_output(
            ["sha256sum", "-c", "STATIC_SHA256SUMS.txt"], cwd=root, text=True, stderr=subprocess.STDOUT
        )
        plan = json.loads((root / "SELECTED103_OFFICIAL_MET_RECOVERY_PLAN_20260907_R4.json").read_text())
        require(plan["experiment_id"] == root.name, "experiment ID drift")
        require(plan["artifact_status"] == "experimental_unvalidated", "artifact status drift")
        require(plan["operator_approved"] is False, "operator approval drift")
        require(plan["formal_gate_status"] == "not_run_not_passed", "formal gate drift")
        require(plan["repair_contract"]["routes"] == ["C40", "D"], "recovery routes drift")
        require(plan["repair_contract"]["execution_mode"] == "sequential_in_process", "execution mode drift")
        require(plan["repair_contract"]["preprocess_workers"] == 0, "preprocess worker drift")
        require(plan["repair_contract"]["export_workers"] == 0, "export worker drift")
        require(plan["repair_contract"]["multiprocessing_queue_used"] is False, "queue contract drift")
        require(plan["repair_contract"]["single_gpu_allocation"] is True, "single-GPU contract drift")
        require(plan["repair_contract"]["exclusive_node"] is False, "exclusive-node contract drift")
        require(plan["bootstrap"] == {"paired_by_case": True, "replicates": 20000, "seed": 20260831}, "bootstrap drift")
        require(sha256_file(R2 / "SELECTED103_OFFICIAL_MET_EXECUTION_PLAN_20260907_R2.json") == PLAN_SHA, "r2 plan SHA drift")
        manifest = root / "manifests/FIXED103_REAL_T2W_CASES.csv"
        require(sha256_file(manifest) == plan["case_contract"]["manifest_sha256"], "manifest SHA drift")
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 103 and len({row["source_case_id"] for row in rows}) == 103, "manifest case drift")
        require(len(list((R2 / "cohort/fixed103/input").glob("*.nii.gz"))) == 412, "r2 input count drift")
        require(len(list((R2 / "cohort/fixed103/reference").glob("*.nii.gz"))) == 103, "r2 reference count drift")

        prior_failures = {}
        for route in ("C40", "D"):
            failed_json = R2 / f"evidence/{route}_EVALUATION_FAILED.json"
            inference_log = R2 / f"logs/{route}_inference_{JOBS[route]}.log"
            data = json.loads(failed_json.read_text())
            text = inference_log.read_text(encoding="utf-8", errors="replace")
            require(data["status"] == "failed" and data["job_id"] == JOBS[route], f"r2 {route} failure drift")
            require("No space left on device (28)" in text, f"r2 {route} root cause drift")
            require(len(list((R2 / f"cohort/fixed103/models/{route}/prediction").glob("*.nii.gz"))) == 37, f"r2 {route} partial count drift")
            prior_failures[route] = {"job_id": JOBS[route], "failure_sha256": sha256_file(failed_json), "log_sha256": sha256_file(inference_log)}

        r3_failure = json.loads((R3 / "evidence/C40_RECOVERY_FAILED.json").read_text())
        require(r3_failure["status"] == "failed" and r3_failure["job_id"] == "3652753", "r3 failure evidence drift")
        require(not (R3 / "evidence/C40_RECOVERY_STARTED.json").exists(), "r3 unexpectedly reached STARTED contract")

        for route in ("B", "E_SNFH", "F"):
            require(not (R2 / f"evidence/{route}_EVALUATION_FAILED.json").exists(), f"r2 {route} already failed")
            source_checkpoint = R2 / f"model_snapshots/{route}/fold_0/checkpoint_final.pth"
            require(source_checkpoint.is_file(), f"r2 {route} model snapshot missing")
            require(sha256_file(source_checkpoint) == plan["routes"][route]["checkpoint_sha256"], f"{route} checkpoint SHA drift")
        for route in ("C40", "D"):
            source_checkpoint = R2 / f"model_snapshots/{route}/fold_0/checkpoint_final.pth"
            require(sha256_file(source_checkpoint) == plan["routes"][route]["checkpoint_sha256"], f"{route} checkpoint SHA drift")
            require(not (root / f"cohort/fixed103/models/{route}").exists(), f"fresh {route} output already exists")

        inference_text = (root / "runtime/frozen_s2_inference.py").read_text()
        slurm_text = (root / "slurm/01_eval_recovery_route.slurm").read_text()
        require("predict_single_npy_array(" in inference_text and "reader_writer.read_images(" in inference_text, "in-process inference path missing")
        require("predict_from_files(" not in inference_text, "multiprocessing inference path remains")
        require("#SBATCH --exclusive" not in slurm_text, "exclusive-node directive remains")
        require("--execution-mode sequential_in_process" in slurm_text, "sequential invocation missing")
        require(not any((root / "evidence").glob("*_SUBMISSION_RESERVATION.json")), "submission reservation already exists")

        payload = {
            "schema_version": 1,
            "status": "pass",
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
            "execution_plan_sha256": sha256_file(root / "SELECTED103_OFFICIAL_MET_RECOVERY_PLAN_20260907_R4.json"),
            "r2_plan_sha256": PLAN_SHA,
            "manifest_sha256": sha256_file(manifest),
            "case_count": 103,
            "input_file_count": 412,
            "reference_file_count": 103,
            "prior_r2_failures": prior_failures,
            "prior_r3_failure": {"job_id": "3652753", "failure_sha256": sha256_file(R3 / "evidence/C40_RECOVERY_FAILED.json")},
            "repair": {"execution_mode": "sequential_in_process", "preprocess_workers": 0, "export_workers": 0, "multiprocessing_queue_used": False, "exclusive_node": False, "single_gpu_allocation": True, "serial_routes": True},
            "deep_validated": 111,
            "checksum_lines": len(checksum_output.splitlines()),
            "locked_test_used": False,
        }
        write_exclusive(passed, payload)
        print(json.dumps(payload, sort_keys=True))
    except Exception as exc:
        payload = {"schema_version":1,"status":"failed","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","failed_at_utc":datetime.now(timezone.utc).isoformat(),"hostname":os.uname().nodename,"error_type":type(exc).__name__,"error":str(exc)}
        if not failed.exists():
            write_exclusive(failed, payload)
        raise


if __name__ == "__main__":
    main()
