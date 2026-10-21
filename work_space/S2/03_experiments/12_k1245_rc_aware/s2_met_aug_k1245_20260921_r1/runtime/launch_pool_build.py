#!/usr/bin/env python3
"""Reserve, submit held, audit, and release the one K1245 pool job."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess


def exclusive(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = root / "evidence"
    preflight = evidence / "REMOTE_STATIC_PREFLIGHT.json"
    if json.loads(preflight.read_text()).get("status") != "pass":
        raise SystemExit("K1245 remote static preflight has not passed")
    reservation = evidence / "POOL_BUILD_RESERVATION.json"
    launch = evidence / "POOL_BUILD_LAUNCH.json"
    released = evidence / "POOL_BUILD_RELEASED.json"
    if any(path.exists() for path in (reservation, launch, released)):
        raise SystemExit("K1245 pool build already has reservation/launch evidence")
    queue = subprocess.run(["squeue", "-h", "-n", "s2kpoolr1", "-o", "%i"], check=True, capture_output=True, text=True).stdout.strip()
    if queue:
        raise SystemExit(f"K1245 pool job name already exists: {queue}")
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    exclusive(reservation, {"status": "reserved", "timestamp_utc": now, "job_name": "s2kpoolr1"})
    script = root / "slurm/01_build_k1245_pools.slurm"
    result = subprocess.run(["sbatch", "--hold", "--parsable", str(script)], check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise SystemExit(f"unexpected sbatch response: {result.stdout!r}")
    exclusive(launch, {
        "artifact_status": "experimental_unvalidated",
        "operator_approved": False,
        "formal_gate_status": "not_run_not_passed",
        "status": "submitted_held",
        "timestamp_utc": now,
        "job_id": job_id,
        "job_name": "s2kpoolr1",
        "plan_sha256": sha(root / "K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.json"),
        "preflight_sha256": sha(preflight),
        "script_sha256": sha(script),
    })
    subprocess.run(["scontrol", "release", job_id], check=True)
    exclusive(released, {"status": "released", "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"), "job_id": job_id})
    print(json.dumps({"job_id": job_id, "job_name": "s2kpoolr1", "status": "released"}, sort_keys=True))


if __name__ == "__main__":
    main()
