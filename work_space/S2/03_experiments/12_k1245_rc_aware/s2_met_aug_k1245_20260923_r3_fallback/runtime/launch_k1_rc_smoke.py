#!/usr/bin/env python3
"""One-time static preflight and held/audited submission of K1 RC smoke."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
R2_ROOT = Path("/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r2")
JOB_NAME = "s2k1rcsmkr3b"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive(path: Path, payload: dict, mode: int = 0o400) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    plan = ROOT / "K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3B.json"
    config_plan = ROOT / "K1_FALLBACK_CONFIG_PLAN_20260923_R3A.json"
    fallback = ROOT / "K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json"
    gate_path = ROOT / "evidence/F_FALLBACK_GATE_VALIDATION.json"
    config_path = ROOT / "config/K1245_ROUTE_K1_FALLBACK.json"
    config_evidence = ROOT / "evidence/K1_FALLBACK_CONFIG_PREPARATION.json"
    slurm = ROOT / "slurm/02_run_k1_rc_smoke_r3b.slurm"
    runtime = ROOT / "runtime/run_k1_rc_smoke.py"
    repo = ROOT / "runtime/repository"
    evidence = ROOT / "evidence"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit = ROOT / "operator" / f"launch_smoke_{stamp}"
    audit.mkdir(parents=True, exist_ok=False)
    expected_plan = "91179ab3edab237462148c33ef6f89a9214932fd6143ed900311e460b0a88da9"
    expected_config_plan = "0a674b1e1d9bb6cf9f7355005f137f0a919f86588687bd6d2a99dd1ec979ece6"
    expected_fallback = "b064244600893378f2693c57e9143c0edcd5584e8a338fa3e979a549b90b239b"
    if sha(plan) != expected_plan or sha(config_plan) != expected_config_plan or sha(fallback) != expected_fallback:
        raise SystemExit("frozen plan SHA mismatch")
    gate = json.loads(gate_path.read_text())
    prepared = json.loads(config_evidence.read_text())
    config = json.loads(config_path.read_text())
    if gate.get("status") != "pass" or gate.get("F_STRICT_PREREG_GATE") is not False:
        raise SystemExit("fallback gate must pass while strict remains false")
    if prepared.get("scientific_gate_pass") is not False or prepared.get("training_authorization") is not False:
        raise SystemExit("K1 is not authorized for training")
    if config.get("route_id") != "K1" or config.get("F_STRICT_PREREG_GATE") is not False:
        raise SystemExit("K1 fallback config contract mismatch")
    if config.get("config_sha256") != prepared.get("config_sha256"):
        raise SystemExit("K1 config hash is not bound to preparation evidence")
    if not (R2_ROOT / "pools/rc27/component_manifest.json").is_file():
        raise SystemExit("RC27 manifest missing")
    manifest_payload = json.loads((R2_ROOT / "pools/rc27/component_manifest.json").read_text())
    if manifest_payload.get("manifest_sha256") != plan_payload_manifest_identity(plan):
        raise SystemExit("RC27 canonical manifest identity mismatch")
    if not slurm.is_file() or not runtime.is_file():
        raise SystemExit("K1 RC smoke runtime is incomplete")
    if sha(runtime) != json.loads(plan.read_text())["smoke_runtime_sha256"]:
        raise SystemExit("K1 smoke runtime SHA mismatch")
    if (evidence / "K1_SMOKE_R3B_LAUNCH.json").exists() or (evidence / "K1_SMOKE_R3B_RESERVATION.json").exists():
        raise SystemExit("K1 RC smoke already has launch evidence")
    if any((evidence / name).exists() for name in ("K1_SMOKE_R3B_STARTED.json", "K1_SMOKE_R3B_FAILED.json", "K1_SMOKE_R3B_VALIDATION.json")):
        raise SystemExit("K1 RC smoke terminal/started evidence already exists")
    queue = subprocess.run(["squeue", "-h", "-n", JOB_NAME, "-o", "%i"], check=True, capture_output=True, text=True).stdout.strip()
    if queue:
        raise SystemExit(f"K1 RC smoke job name already in queue: {queue}")
    for source in (plan, config_plan, gate_path, config_path, slurm, runtime):
        if not source.is_file():
            raise SystemExit(f"required smoke input missing: {source}")
    preflight = {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "status": "pass",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "job_name": JOB_NAME, "route_id": "K1", "device": "one A100", "exclusive": False,
        "fallback_plan_sha256": sha(fallback), "smoke_plan_sha256": sha(plan),
        "config_plan_sha256": sha(config_plan), "fallback_gate_sha256": sha(gate_path),
        "config_sha256": sha(config_path), "runtime_sha256": sha(runtime), "slurm_sha256": sha(slurm),
        "manifest_file_sha256": sha(R2_ROOT / "pools/rc27/component_manifest.json"),
        "manifest_identity_sha256": manifest_payload["manifest_sha256"],
        "F_STRICT_PREREG_GATE": False, "training_authorization": False,
        "manual_review_required": 24,
    }
    exclusive(evidence / "K1_SMOKE_R3B_STATIC_PREFLIGHT.json", preflight)
    exclusive(evidence / "K1_SMOKE_R3B_RESERVATION.json", {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "status": "reserved",
        "job_name": JOB_NAME, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "preflight_sha256": sha(evidence / "K1_SMOKE_R3B_STATIC_PREFLIGHT.json"),
    })
    result = subprocess.run(["sbatch", "--hold", "--parsable", str(slurm)], check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise SystemExit("unexpected sbatch response")
    exclusive(evidence / "K1_SMOKE_R3B_LAUNCH.json", {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "status": "submitted_held",
        "job_id": job_id, "job_name": JOB_NAME,
        "plan_sha256": sha(plan), "config_sha256": sha(config_path),
        "preflight_sha256": sha(evidence / "K1_SMOKE_R3B_STATIC_PREFLIGHT.json"),
        "slurm_sha256": sha(slurm), "runtime_sha256": sha(runtime),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    subprocess.run(["scontrol", "release", job_id], check=True)
    exclusive(evidence / "K1_SMOKE_R3B_RELEASED.json", {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "status": "released", "job_id": job_id,
        "job_name": JOB_NAME, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    exclusive(audit / "LOCAL_LAUNCH_RECEIPT.json", {
        "artifact_status": "experimental_unvalidated", "operator_approved": False,
        "formal_gate_status": "not_run_not_passed", "status": "released", "job_id": job_id,
        "job_name": JOB_NAME, "plan_sha256": sha(plan), "preflight_sha256": sha(evidence / "K1_SMOKE_R3B_STATIC_PREFLIGHT.json"),
        "runtime_sha256": sha(runtime), "slurm_sha256": sha(slurm),
    })
    print(json.dumps({"job_id": job_id, "job_name": JOB_NAME, "status": "released", "strict_gate": False}, sort_keys=True))


def plan_payload_manifest_identity(plan_path: Path) -> str:
    payload = json.loads(plan_path.read_text())
    return payload["smoke"]["pool_manifest_identity_sha256"]


if __name__ == "__main__":
    main()
