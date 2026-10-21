#!/usr/bin/env python3
"""Exclusively submit the B/E/F Screen175 extension jobs once."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: dict) -> None:
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
    if preflight.get("status") != "pass" or preflight.get("checkpoint_count_deep_validated") != 25:
        raise SystemExit("strict 25-checkpoint preflight has not passed")
    plan = root / "EXTEND175_EXECUTION_PLAN_20260903_R1.json"
    script = root / "slurm/01_train_extend175_route.slurm"
    routes = (("B", "s2ab175"), ("E_SNFH", "s2ae175"), ("F", "s2af175"))
    for token, job_name in routes:
        launch = root / "evidence" / f"{token}_EXTENSION_TRAINING_LAUNCH.json"
        if launch.exists():
            raise SystemExit(f"extension launch audit already exists: {launch}")
        queue = subprocess.run(["squeue", "-h", "-n", job_name, "-o", "%i"], check=True, capture_output=True, text=True).stdout.strip()
        if queue:
            raise SystemExit(f"extension job name already exists: {job_name} -> {queue}")
    launched = []
    for token, job_name in routes:
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
            "extension_epochs": 50,
            "formal_gate_status": "not_run_not_passed",
            "job_id": job_id,
            "job_name": job_name,
            "operator_approved": False,
            "plan_sha256": sha256_file(plan),
            "preflight_sha256": sha256_file(preflight_path),
            "route_token": token,
            "script_sha256": sha256_file(script),
            "source_total_epoch": 125,
            "status": "submitted",
            "target_total_epoch": 175,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "trainer_sha256": sha256_file(root / "runtime/repository/custom_nnunet/nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune.py"),
        }
        launch = root / "evidence" / f"{token}_EXTENSION_TRAINING_LAUNCH.json"
        write_exclusive(launch, payload)
        launched.append(payload)
    write_exclusive(
        root / "evidence/THREE_ROUTE_EXTENSION_LAUNCH_SUMMARY.json",
        {
            "artifact_status": "experimental_unvalidated",
            "formal_gate_status": "not_run_not_passed",
            "jobs": launched,
            "operator_approved": False,
            "status": "submitted",
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )
    print(json.dumps(launched, sort_keys=True))


if __name__ == "__main__":
    main()
