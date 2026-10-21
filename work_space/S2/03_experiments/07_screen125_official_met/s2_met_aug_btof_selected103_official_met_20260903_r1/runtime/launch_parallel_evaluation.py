#!/usr/bin/env python3
"""Exclusively submit five route jobs and their dependent paired comparison."""

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
    "experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1"
)
ROUTES = ("B", "C40", "D", "E_SNFH", "F")
JOB_NAMES = {
    "B": "s2b103m1",
    "C40": "s2c103m1",
    "D": "s2d103m1",
    "E_SNFH": "s2e103m1",
    "F": "s2f103m1",
}


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
    preflight_path = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "pass":
        raise RuntimeError("remote static preflight did not pass")
    plan_path = root / "SELECTED103_OFFICIAL_MET_EXECUTION_PLAN_20260903_R1.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["submission_contract"]["route_order"] != list(ROUTES):
        raise RuntimeError("route order drift")
    if (root / "evidence/FIVE_ROUTE_LAUNCH_SUMMARY.json").exists():
        raise RuntimeError("five-route launch summary already exists")

    launched: dict[str, dict] = {}
    try:
        for route in ROUTES:
            reservation = root / f"evidence/{route}_EVALUATION_SUBMISSION_RESERVATION.json"
            launch = root / f"evidence/{route}_EVALUATION_LAUNCH.json"
            if reservation.exists() or launch.exists():
                raise RuntimeError(f"{route} submission evidence already exists")
            reservation_payload = {
                "schema_version": 1,
                "status": "reserved",
                "artifact_status": "experimental_unvalidated",
                "operator_approved": False,
                "formal_gate_status": "not_run_not_passed",
                "route": route,
                "job_name": JOB_NAMES[route],
                "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": sha256_file(plan_path),
                "checkpoint_sha256": plan["routes"][route]["checkpoint_sha256"],
                "selected_epoch": plan["routes"][route]["checkpoint_epoch"],
            }
            write_exclusive(reservation, reservation_payload)
            job_id = submit(
                [
                    "sbatch",
                    "--parsable",
                    f"--job-name={JOB_NAMES[route]}",
                    str(root / "slurm/01_eval_selected_route.slurm"),
                    route,
                ]
            )
            launch_payload = {
                **reservation_payload,
                "status": "submitted",
                "job_id": job_id,
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "slurm_script_sha256": sha256_file(root / "slurm/01_eval_selected_route.slurm"),
            }
            write_exclusive(launch, launch_payload)
            launched[route] = launch_payload

        paired_reservation = root / "evidence/PAIRED_COMPARISON_SUBMISSION_RESERVATION.json"
        paired_launch = root / "evidence/PAIRED_COMPARISON_LAUNCH.json"
        if paired_reservation.exists() or paired_launch.exists():
            raise RuntimeError("paired submission evidence already exists")
        dependency = "afterok:" + ":".join(launched[route]["job_id"] for route in ROUTES)
        paired_reservation_payload = {
            "schema_version": 1,
            "status": "reserved",
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "job_name": "s2x103p1",
            "dependency": dependency,
            "upstream_job_ids": [launched[route]["job_id"] for route in ROUTES],
            "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": sha256_file(plan_path),
        }
        write_exclusive(paired_reservation, paired_reservation_payload)
        paired_job_id = submit(
            [
                "sbatch",
                "--parsable",
                "--job-name=s2x103p1",
                f"--dependency={dependency}",
                str(root / "slurm/02_compare_selected_routes.slurm"),
            ]
        )
        paired_launch_payload = {
            **paired_reservation_payload,
            "status": "submitted",
            "job_id": paired_job_id,
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_script_sha256": sha256_file(root / "slurm/02_compare_selected_routes.slurm"),
        }
        write_exclusive(paired_launch, paired_launch_payload)
        summary = {
            "schema_version": 1,
            "status": "submitted",
            "artifact_status": "experimental_unvalidated",
            "operator_approved": False,
            "formal_gate_status": "not_run_not_passed",
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "parallel_gpu_jobs": {route: launched[route]["job_id"] for route in ROUTES},
            "paired_cpu_job": paired_job_id,
            "paired_dependency": dependency,
            "plan_sha256": sha256_file(plan_path),
            "extension_jobs_are_separate": True,
            "locked_test_used": False,
        }
        write_exclusive(root / "evidence/FIVE_ROUTE_LAUNCH_SUMMARY.json", summary)
        print(json.dumps(summary, sort_keys=True))
    except Exception as exc:
        failure_path = root / "evidence/FIVE_ROUTE_LAUNCH_FAILED.json"
        if not failure_path.exists():
            write_exclusive(
                failure_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "artifact_status": "experimental_unvalidated",
                    "operator_approved": False,
                    "formal_gate_status": "not_run_not_passed",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "already_submitted": {route: item["job_id"] for route, item in launched.items()},
                },
            )
        raise


if __name__ == "__main__":
    main()
