"""Check preserved submitted files and stop receipts locally; no remote access."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "d9d353513e5ce6e95e905c843f6174db17c3e651f8a060fff07876b2edbcf387"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    plan_path = ROOT / "CPU_EXECUTION_PLAN.json"
    check(digest(plan_path) == PLAN_SHA, "submitted plan drift")
    plan = json.loads(plan_path.read_text())
    bundle = ROOT / "operator/deploy_cpu_once_20261013T130349Z/transfer_bundle"
    check(digest(bundle / plan_path.name) == PLAN_SHA, "bundle plan drift")
    for name, expected in plan["static_sha256"].items():
        check(digest(ROOT / name) == expected, "local static drift: " + name)
        check(digest(bundle / name) == expected, "bundle static drift: " + name)
        if name.endswith(".json"):
            json.loads((ROOT / name).read_text())
    python_paths = list((ROOT / "runtime").glob("*.py"))
    python_paths += list((ROOT / "tests").glob("*.py"))
    python_paths += list((ROOT / "operator").glob("*.py"))
    for path in python_paths:
        compile(path.read_bytes(), str(path), "exec")
    stop_root = ROOT / "operator/stop_known_unsafe_cpu_20261013T130539Z"
    stop = json.loads((stop_root / "STOP_RECEIPT.json").read_text())
    receipt = json.loads((stop_root / "LOCAL_OPERATION_RECEIPT.json").read_text())
    check(stop["job_id"] == "4051333" and stop["job_name"] == plan["job_name"], "stop identity drift")
    check(stop["status"] == "cancel_requested_known_runtime_edge_bug", "unexpected stop status")
    check(receipt["vpn_closed"] is True, "owned VPN close unconfirmed")
    result = dict(
        artifact_status="experimental_unvalidated", operator_approved=False,
        formal_gate_status="not_run_not_passed", status="pass",
        local_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=PLAN_SHA,
        static_files_checked=len(plan["static_sha256"]), frozen_bundle_matches=True,
        python_syntax_checked=len(python_paths), vpn_closed=True,
        stop_status=stop["status"], remote_terminal_verified=False,
        empty_reference_runtime_tested=False, diagnosis_complete=False,
        inspection="local_only_no_remote_access",
    )
    target = ROOT / "operator/FROZEN_LOCAL_VERIFICATION_20261013.json"
    with target.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
