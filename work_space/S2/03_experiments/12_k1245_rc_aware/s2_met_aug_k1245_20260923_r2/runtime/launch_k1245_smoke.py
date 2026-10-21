#!/usr/bin/env python3
"""One-time held submission and atomic audit for the K1245 smoke batch."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess


JOB_NAME = "s2k1245smkr2"
META = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
}


def exclusive(path: Path, payload: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def sha(path: Path) -> str:
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
    now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    evidence = root / "evidence"
    preflight = evidence / "REMOTE_STATIC_PREFLIGHT.json"
    pool_complete = evidence / "POOL_BUILD_COMPLETE.ok"
    smoke_preflight = evidence / "K1245_SMOKE_STATIC_PREFLIGHT.json"
    launch = evidence / "K1245_SMOKE_LAUNCH.json"
    released = evidence / "K1245_SMOKE_RELEASED.json"
    for path in (preflight, pool_complete):
        if not path.is_file():
            raise SystemExit(f"missing prerequisite: {path}")
    if json.loads(preflight.read_text()).get("status") != "pass":
        raise SystemExit("static preflight is not pass")
    if smoke_preflight.exists():
        raise SystemExit("smoke static preflight already exists")
    if any(path.exists() for path in (launch, released)):
        raise SystemExit("smoke launch audit already exists")
    for name in ("K1245_SMOKE_STARTED.json", "K1245_SMOKE_FAILED.json", "K1245_SMOKE_VALIDATION.json", "K1245_SMOKE_COMPLETE.ok"):
        if (evidence / name).exists():
            raise SystemExit(f"smoke evidence already exists: {name}")
    queue = subprocess.run(["squeue", "-h", "-n", JOB_NAME, "-o", "%i"], check=True, capture_output=True, text=True).stdout.strip()
    if queue:
        raise SystemExit(f"smoke job name already exists: {queue}")
    config_hashes = {}
    manifest_hashes = {}
    for route, pool in (("K2", "rc27"), ("K4", "small8")):
        config_path = root / f"config/K1245_ROUTE_{route}.json"
        manifest_path = root / f"pools/{pool}/component_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"missing {route} smoke input")
        config = json.loads(config_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if config.get("route_id") != route or config.get("manifest_sha256") != manifest.get("manifest_sha256"):
            raise SystemExit(f"{route} config/manifest binding drifted")
        if int(manifest.get("component_count", 0)) < 24:
            raise SystemExit(f"{route} smoke pool has fewer than 24 components")
        config_hashes[route] = sha(config_path)
        manifest_hashes[route] = sha(manifest_path)
    exclusive(smoke_preflight, {
        **META,
        "status": "pass",
        "timestamp_utc": now,
        "job_name": JOB_NAME,
        "routes": ["K2", "K4"],
        "pool_complete_sha256": sha(pool_complete),
        "remote_static_preflight_sha256": sha(preflight),
        "config_sha256": config_hashes,
        "manifest_file_sha256": manifest_hashes,
        "single_gpu": True,
        "manual_review_required": True,
    })
    script = root / "slurm/02_run_k1245_smoke.slurm"
    if not script.is_file():
        raise SystemExit("smoke slurm script missing")
    exclusive(evidence / "K1245_SMOKE_RESERVATION.json", {
        **META, "status":"reserved",
        "job_name":JOB_NAME, "timestamp_utc":now,
    })
    try:
        result = subprocess.run(["sbatch", "--hold", "--parsable", str(script)], check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SystemExit(f"unexpected sbatch response: {result.stdout!r}")
        exclusive(launch, {
            **META, "status":"submitted_held",
            "timestamp_utc":now, "job_id":job_id, "job_name":JOB_NAME,
            "plan_sha256":sha(next(root.glob("K1245_RC_AWARE_EXECUTION_PLAN_*.json"))),
            "preflight_sha256":sha(preflight), "script_sha256":sha(script),
            "smoke_preflight_sha256": sha(smoke_preflight),
            "routes":["K2","K4"], "single_gpu":True,
        })
        subprocess.run(["scontrol", "release", job_id], check=True)
        exclusive(released, {"status":"released", "job_id":job_id, "job_name":JOB_NAME,
                             "timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z")})
    except Exception:
        raise
    print(json.dumps({"job_id":job_id, "job_name":JOB_NAME, "status":"released"}, sort_keys=True))


if __name__ == "__main__":
    main()
