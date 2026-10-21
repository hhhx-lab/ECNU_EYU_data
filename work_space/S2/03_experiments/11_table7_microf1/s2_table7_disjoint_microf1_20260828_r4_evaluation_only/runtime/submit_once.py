#!/usr/bin/env python3
"""Submit the immutable evaluation-only r4 exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


JOB_NAME = "s2t7mf1r4"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-script-sha256", required=True)
    args = parser.parse_args()

    root = args.root.resolve(strict=True)
    evidence = root / "evidence"
    script = root / "slurm/01_evaluate_only_table7.slurm"
    preflight_path = evidence / "REMOTE_STATIC_PREFLIGHT.json"
    reservation_path = evidence / "TABLE7_EVALUATION_ONLY_SUBMISSION_RESERVED.json"
    launch_path = evidence / "TABLE7_EVALUATION_ONLY_LAUNCH.json"
    submission_failed = evidence / "TABLE7_EVALUATION_ONLY_SUBMISSION_FAILED.json"

    require(script.is_file(), "missing Slurm script")
    require(sha256_file(script) == args.expected_script_sha256, "Slurm script SHA drift")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "pass", "remote static preflight did not pass")
    require(preflight["evaluation_only"] is True, "preflight is not evaluation-only")
    for path in (reservation_path, launch_path, submission_failed):
        require(not path.exists(), f"submission evidence already exists: {path.name}")
    for path in (
        evidence / "TABLE7_EVALUATION_ONLY_STARTED.json",
        evidence / "TABLE7_EVALUATION_ONLY_FAILED.json",
        evidence / "TABLE7_DISJOINT_MICROF1_VALIDATION.json",
        evidence / "TABLE7_DISJOINT_MICROF1_COMPLETE.ok",
        root / "views",
        root / "official_reproduction",
        root / "disjoint_microf1",
    ):
        require(not path.exists(), f"r4 output already exists: {path}")

    queue = subprocess.run(
        ["squeue", "-h", "-n", JOB_NAME, "-o", "%i|%j|%T"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(not queue, f"existing {JOB_NAME} job: {queue}")

    now = datetime.now(timezone.utc).isoformat()
    reservation = {
        "schema_version": 1,
        "status": "reserved",
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "evaluation_only": True,
        "job_name": JOB_NAME,
        "reserved_at_utc": now,
        "reserved_on_host": socket.gethostname(),
        "script_sha256": sha256_file(script),
        "preflight_sha256": sha256_file(preflight_path),
    }
    write_json_exclusive(reservation_path, reservation)

    try:
        result = subprocess.run(
            ["sbatch", "--parsable", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.fullmatch(r"([0-9]+)(?:;[^\s]+)?", result.stdout.strip())
        require(match is not None, f"unexpected sbatch output: {result.stdout.strip()!r}")
        job_id = match.group(1)
        launch = {
            "schema_version": 1,
            "status": "submitted",
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "evaluation_only": True,
            "training": False,
            "inference": False,
            "job_id": job_id,
            "job_name": JOB_NAME,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "slurm/01_evaluate_only_table7.slurm",
            "script_sha256": sha256_file(script),
            "preflight_sha256": sha256_file(preflight_path),
            "plan_sha256": sha256_file(root / "TABLE7_DISJOINT_MICROF1_EXECUTION_PLAN_20260828_R4.json"),
            "r2_job_id": "3542339",
            "r2_inference_audit_sha256": preflight["r2_inference_audit_sha256"],
            "mapping_csv_sha256": preflight["mapping_csv_sha256"],
            "case_pair_sha256": preflight["case_pair_sha256"],
        }
        write_json_exclusive(launch_path, launch)
        print(json.dumps(launch, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "evaluation_only": True,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            write_json_exclusive(submission_failed, failure)
        finally:
            raise


if __name__ == "__main__":
    main()
