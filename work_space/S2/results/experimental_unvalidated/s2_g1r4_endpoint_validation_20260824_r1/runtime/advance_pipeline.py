#!/usr/bin/env python3
"""Advance at most one gated stage of the frozen endpoint-validation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = (
    {
        "key": "val27_r4_synthesis",
        "job_name": "s2r4v27s1",
        "slurm": "slurm/01_val27_synthesize_r4.slurm",
        "launch": "evidence/VAL27_SYNTHESIS_LAUNCH.json",
        "reservation": "evidence/VAL27_SYNTHESIS_SUBMISSION_RESERVATION.json",
        "failed": "evidence/VAL27_SYNTHESIS_FAILED.json",
        "validation": "evidence/VAL27_SYNTHESIS_VALIDATION.json",
        "complete": "evidence/VAL27_SYNTHESIS_COMPLETE.ok",
        "count": 27,
    },
    {
        "key": "val27_four_models",
        "job_name": "s2r4v27m1",
        "slurm": "slurm/02_val27_four_model_eval.slurm",
        "launch": "evidence/VAL27_FOUR_MODELS_LAUNCH.json",
        "reservation": "evidence/VAL27_FOUR_MODELS_SUBMISSION_RESERVATION.json",
        "failed": "evidence/VAL27_FOUR_MODELS_FAILED.json",
        "validation": "evidence/VAL27_FOUR_MODELS_VALIDATION.json",
        "complete": "evidence/VAL27_FOUR_MODELS_COMPLETE.ok",
        "count": 27,
    },
    {
        "key": "fixed103_real_vs_synthetic",
        "job_name": "s2r4103p1",
        "slurm": "slurm/03_fixed103_real_vs_synthetic.slurm",
        "launch": "evidence/FIXED103_PAIRED_LAUNCH.json",
        "reservation": "evidence/FIXED103_PAIRED_SUBMISSION_RESERVATION.json",
        "failed": "evidence/FIXED103_PAIRED_FAILED.json",
        "validation": "evidence/FIXED103_PAIRED_VALIDATION.json",
        "complete": "evidence/FIXED103_PAIRED_COMPLETE.ok",
        "count": 103,
    },
    {
        "key": "test26_locked_endpoint",
        "job_name": "s2r4t26e1",
        "slurm": "slurm/04_test26_locked_endpoint.slurm",
        "launch": "evidence/TEST26_LOCKED_ENDPOINT_LAUNCH.json",
        "reservation": "evidence/TEST26_LOCKED_ENDPOINT_SUBMISSION_RESERVATION.json",
        "failed": "evidence/TEST26_LOCKED_ENDPOINT_FAILED.json",
        "validation": "evidence/TEST26_LOCKED_ENDPOINT_VALIDATION.json",
        "complete": "evidence/TEST26_LOCKED_ENDPOINT_COMPLETE.ok",
        "count": 26,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def scheduler_state(job_id: str) -> tuple[str, str]:
    result = subprocess.run(
        ["sacct", "-X", "-j", job_id, "-n", "-P", "-o", "JobIDRaw,State,ExitCode"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"sacct failed: {result.stdout}")
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    exact = [row for row in rows if row[0] == str(job_id)]
    require(exact, f"job {job_id} absent from sacct")
    return exact[0][1].split()[0], exact[0][2]


def active_jobs(job_name: str) -> list[str]:
    result = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", "zqchen"), "-h", "-n", job_name, "-o", "%i|%T"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"squeue failed: {result.stdout}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def validate_complete(root: Path, stage: dict[str, Any]) -> dict[str, Any]:
    require((root / stage["complete"]).is_file(), f"complete marker missing: {stage['key']}")
    path = root / stage["validation"]
    require(path.is_file(), f"validation missing: {stage['key']}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("status") == "pass", f"validation status failed: {stage['key']}")
    require(payload.get("artifact_status") == "experimental_unvalidated", "artifact status drift")
    require(payload.get("operator_approved") is False, "operator approval drift")
    require(payload.get("formal_gate_status") == "not_run_not_passed", "formal gate drift")
    require(payload.get("stage") == stage["key"], f"stage validation binding drift: {stage['key']}")
    require(payload.get("case_count") == stage["count"], f"stage case count drift: {stage['key']}")
    return payload


def submit_stage(root: Path, stage: dict[str, Any]) -> dict[str, Any]:
    require(not (root / stage["launch"]).exists(), f"launch audit already exists: {stage['key']}")
    require(not (root / stage["reservation"]).exists(), f"submission reservation already exists: {stage['key']}")
    require(not (root / stage["failed"]).exists(), f"failure evidence exists: {stage['key']}")
    require(not active_jobs(stage["job_name"]), f"active job already uses name {stage['job_name']}")
    script = root / stage["slurm"]
    plan = root / "ENDPOINT_VALIDATION_EXECUTION_PLAN_20260824_R1.json"
    reservation = {
        "schema_version": 1,
        "status": "reserved",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "stage": stage["key"],
        "job_name": stage["job_name"],
        "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_script_sha256": sha256_file(script),
        "execution_plan_sha256": sha256_file(plan),
    }
    write_exclusive(root / stage["reservation"], reservation)
    result = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(result.returncode == 0, f"sbatch failed after reservation: {result.stdout}")
    job_id = result.stdout.strip().split(";", 1)[0]
    require(job_id.isdigit(), f"invalid sbatch job id: {result.stdout}")
    launch = {
        **reservation,
        "status": "submitted",
        "job_id": job_id,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission_reservation_sha256": sha256_file(root / stage["reservation"]),
    }
    write_exclusive(root / stage["launch"], launch)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    require(root.is_dir(), f"missing root: {root}")
    remote_preflight = json.loads((root / "evidence/REMOTE_STATIC_PREFLIGHT.json").read_text())
    require(remote_preflight.get("status") == "pass", "remote static preflight is not pass")

    for index, stage in enumerate(STAGES):
        if (root / stage["failed"]).exists():
            print(json.dumps({"status": "failed", "stage": stage["key"], "failure": str(root / stage["failed"])}, sort_keys=True))
            return
        if (root / stage["complete"]).exists():
            validate_complete(root, stage)
            continue
        launch_path = root / stage["launch"]
        if not launch_path.exists():
            if index > 0:
                validate_complete(root, STAGES[index - 1])
            launch = submit_stage(root, stage)
            print(json.dumps({"status": "submitted", "stage": stage["key"], "job_id": launch["job_id"]}, sort_keys=True))
            return
        launch = json.loads(launch_path.read_text())
        state, exit_code = scheduler_state(str(launch["job_id"]))
        if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
            print(json.dumps({"status": "active", "stage": stage["key"], "job_id": launch["job_id"], "state": state}, sort_keys=True))
            return
        if state == "COMPLETED" and exit_code == "0:0":
            validate_complete(root, stage)
            if index + 1 == len(STAGES):
                require((root / "ENDPOINT_VALIDATION_PIPELINE_COMPLETE.ok").is_file(), "pipeline complete marker missing")
                print(json.dumps({"status": "pipeline_complete", "stage": stage["key"], "job_id": launch["job_id"]}, sort_keys=True))
                return
            next_launch = submit_stage(root, STAGES[index + 1])
            print(json.dumps({"status": "submitted", "completed_stage": stage["key"], "stage": STAGES[index + 1]["key"], "job_id": next_launch["job_id"]}, sort_keys=True))
            return
        print(json.dumps({"status": "scheduler_failure", "stage": stage["key"], "job_id": launch["job_id"], "state": state, "exit_code": exit_code}, sort_keys=True))
        return
    require((root / "ENDPOINT_VALIDATION_PIPELINE_COMPLETE.ok").is_file(), "all stage markers exist but pipeline marker is missing")
    print(json.dumps({"status": "pipeline_complete"}, sort_keys=True))


if __name__ == "__main__":
    main()
