#!/usr/bin/env python3
"""Exclusively submit the five independent screening jobs once."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    preflight_path = root / "evidence/REMOTE_STATIC_PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "pass":
        raise SystemExit("remote static preflight has not passed")
    plan = root / "SCREEN125_EXECUTION_PLAN_20260831_R1.json"
    script = root / "slurm/01_train_screen125_route.slurm"
    routes = (
        ("B", "s2ab125"),
        ("C40", "s2ac125"),
        ("D", "s2ad125"),
        ("E_SNFH", "s2ae125"),
        ("F", "s2af125"),
    )
    for token, job_name in routes:
        launch_path = root / "evidence" / f"{token}_TRAINING_LAUNCH.json"
        if launch_path.exists():
            raise SystemExit(f"launch audit already exists: {launch_path}")
        queue = subprocess.run(
            ["squeue", "-h", "-n", job_name, "-o", "%i"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if queue:
            raise SystemExit(f"job name already exists in queue: {job_name} -> {queue}")
    launched = []
    for token, job_name in routes:
        config = root / "config" / f"SCREEN_ROUTE_{token}.json"
        result = subprocess.run(
            ["sbatch", "--parsable", "--job-name", job_name, str(script), token],
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SystemExit(f"unexpected sbatch response for {token}: {result.stdout!r}")
        payload = {
            "artifact_status": "experimental_unvalidated",
            "config_sha256": sha256_file(config),
            "epochs": 125,
            "formal_gate_status": "not_run_not_passed",
            "job_id": job_id,
            "job_name": job_name,
            "operator_approved": False,
            "plan_sha256": sha256_file(plan),
            "preflight_sha256": sha256_file(preflight_path),
            "route_token": token,
            "save_every": 25,
            "script_sha256": sha256_file(script),
            "status": "submitted",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        launch_path = root / "evidence" / f"{token}_TRAINING_LAUNCH.json"
        exclusive_json(launch_path, payload)
        launched.append(payload)
    summary = {
        "artifact_status": "experimental_unvalidated",
        "formal_gate_status": "not_run_not_passed",
        "jobs": launched,
        "operator_approved": False,
        "status": "submitted",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    exclusive_json(root / "evidence/FIVE_ROUTE_LAUNCH_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
