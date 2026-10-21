#!/usr/bin/env python3
"""Atomically submit serial C40/D recovery jobs and the five-route paired job."""

from __future__ import annotations

import argparse
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
UPSTREAM = {"B": "3652702", "E_SNFH": "3652705", "F": "3652706"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def submit(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {result.stdout!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"unexpected root: {root}")
    preflight = json.loads((root / "evidence/REMOTE_STATIC_PREFLIGHT.json").read_text())
    if preflight.get("status") != "pass":
        raise RuntimeError("remote static preflight did not pass")
    plan_path = root / "SELECTED103_OFFICIAL_MET_RECOVERY_PLAN_20260907_R4.json"
    plan_sha = sha256_file(plan_path)
    if (root / "evidence/RECOVERY_LAUNCH_SUMMARY.json").exists():
        raise RuntimeError("recovery launch summary already exists")
    launched = {}
    try:
        c_res = root / "evidence/C40_RECOVERY_SUBMISSION_RESERVATION.json"
        write_exclusive(c_res, {"schema_version":1,"status":"reserved","route":"C40","job_name":"s2c103m4","plan_sha256":plan_sha,"reserved_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed"})
        c_job = submit(["sbatch", "--parsable", "--job-name=s2c103m4", str(root / "slurm/01_eval_recovery_route.slurm"), "C40"])
        launched["C40"] = c_job
        write_exclusive(root / "evidence/C40_RECOVERY_LAUNCH.json", {"schema_version":1,"status":"submitted","route":"C40","job_name":"s2c103m4","job_id":c_job,"plan_sha256":plan_sha,"submitted_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed"})

        d_res = root / "evidence/D_RECOVERY_SUBMISSION_RESERVATION.json"
        write_exclusive(d_res, {"schema_version":1,"status":"reserved","route":"D","job_name":"s2d103m4","dependency":f"afterok:{c_job}","plan_sha256":plan_sha,"reserved_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed"})
        d_job = submit(["sbatch", "--parsable", "--job-name=s2d103m4", f"--dependency=afterok:{c_job}", str(root / "slurm/01_eval_recovery_route.slurm"), "D"])
        launched["D"] = d_job
        write_exclusive(root / "evidence/D_RECOVERY_LAUNCH.json", {"schema_version":1,"status":"submitted","route":"D","job_name":"s2d103m4","job_id":d_job,"dependency":f"afterok:{c_job}","plan_sha256":plan_sha,"submitted_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed"})

        dependency = "afterok:" + ":".join([UPSTREAM["B"], c_job, d_job, UPSTREAM["E_SNFH"], UPSTREAM["F"]])
        write_exclusive(root / "evidence/PAIRED_COMPARISON_SUBMISSION_RESERVATION.json", {"schema_version":1,"status":"reserved","job_name":"s2x103p4","dependency":dependency,"plan_sha256":plan_sha,"reserved_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed"})
        paired_job = submit(["sbatch", "--parsable", "--job-name=s2x103p4", f"--dependency={dependency}", str(root / "slurm/02_compare_recovered_routes.slurm")])
        write_exclusive(root / "evidence/PAIRED_COMPARISON_LAUNCH.json", {"schema_version":1,"status":"submitted","job_name":"s2x103p4","job_id":paired_job,"dependency":dependency,"plan_sha256":plan_sha,"submitted_at_utc":datetime.now(timezone.utc).isoformat(),"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed"})
        summary = {"schema_version":1,"status":"submitted","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","submitted_at_utc":datetime.now(timezone.utc).isoformat(),"recovery_jobs":launched,"paired_cpu_job":paired_job,"paired_dependency":dependency,"reused_r2_jobs":UPSTREAM,"serial_recovery":True,"execution_mode":"sequential_in_process","exclusive_node":False,"plan_sha256":plan_sha,"locked_test_used":False}
        write_exclusive(root / "evidence/RECOVERY_LAUNCH_SUMMARY.json", summary)
        print(json.dumps(summary, sort_keys=True))
    except Exception as exc:
        failure = root / "evidence/RECOVERY_LAUNCH_FAILED.json"
        if not failure.exists():
            write_exclusive(failure, {"schema_version":1,"status":"failed","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","failed_at_utc":datetime.now(timezone.utc).isoformat(),"error_type":type(exc).__name__,"error":str(exc),"already_submitted":launched})
        raise


if __name__ == "__main__":
    main()
