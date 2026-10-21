"""Local post-submit integrity verification; never connects or changes runtime."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "operator/deploy_checked_once_20261014T041648Z"
PLAN_SHA = "e3169bf735a39d05c066ca34c5568285b48ba060c79007331a850c903c936beb"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    plan = json.loads((ROOT / "CPU_EXECUTION_PLAN.json").read_text())
    require(sha(ROOT / "CPU_EXECUTION_PLAN.json") == PLAN_SHA, "r2 plan drift")
    require(sha(AUDIT / "transfer_bundle/CPU_EXECUTION_PLAN.json") == PLAN_SHA, "transferred plan drift")
    for name, digest in plan["static_sha256"].items():
        require(sha(ROOT/name) == digest and sha(AUDIT/"transfer_bundle"/name) == digest,
                "r2 frozen static drift: " + name)
    old_root = ROOT.parent / "s2_k1_small_lesion_p0_20261013_r1"
    old_plan_path = old_root / "CPU_EXECUTION_PLAN.json"
    old_plan = json.loads(old_plan_path.read_text())
    require(sha(old_plan_path) == plan["parent_plan_sha256"], "r1 plan drift")
    for name, digest in old_plan["static_sha256"].items():
        require(sha(old_root/name) == digest, "r1 static drift: " + name)
    for key in ("case_ids", "input_reference_sha256", "prediction_roots", "prediction_sha256",
                "dependency_sha256", "software_sha256", "plans_path", "dataset_path",
                "small_reference_contract", "matched_dice_threshold", "matching_threshold",
                "cpu_count", "memory_gb", "gpu_count", "walltime"):
        require(plan[key] == old_plan[key], "unintended data/metric/resource contract change: " + key)
    boundary = json.loads((AUDIT / "OFFICIAL_BOUNDARY_TESTS.json").read_text())
    pre = json.loads((AUDIT / "CPU_STATIC_PREFLIGHT.json").read_text())
    launch = json.loads((AUDIT / "LAUNCH_RECEIPT.json").read_text())
    receipt = json.loads((AUDIT / "LOCAL_OPERATION_RECEIPT.json").read_text())
    require(pre["plan_sha256"] == PLAN_SHA and pre["status"] == "pass", "preflight failed")
    require(pre["official_boundary_report_sha256"] == sha(AUDIT / "OFFICIAL_BOUNDARY_TESTS.json"), "boundary SHA drift")
    require(boundary["status"] == "pass" and boundary["unittest_count"] == 11, "tests failed")
    require(len(boundary["synthetic_checks"]) == 14 and len(boundary["real_fixed103_checks"]) == 4, "test coverage missing")
    require(launch["status"] == "released" and launch["job_id"] == "4068477" and
            launch["job_name"] == plan["job_name"], "launch mismatch")
    require(receipt["vpn_closed"] is True, "VPN closure unconfirmed")
    for path in list((ROOT / "runtime").glob("*.py")) + list((ROOT / "tests").glob("*.py")) + list((ROOT / "operator").glob("*.py")):
        compile(path.read_bytes(), str(path), "exec")
    result = dict(artifact_status="experimental_unvalidated", operator_approved=False,
                  formal_gate_status="not_run_not_passed", status="pass_engineering_submission_only",
                  local_utc=datetime.now(timezone.utc).isoformat(), job_id=launch["job_id"],
                  job_name=plan["job_name"], plan_sha256=PLAN_SHA,
                  new_static_files_checked=len(plan["static_sha256"]), old_r1_static_unchanged=True,
                  data_metric_resource_contracts_unchanged=True, actual_official_unittests=11,
                  synthetic_comparisons=14, real_fixed103_subset_checks=4,
                  boundary_report_sha256=sha(AUDIT/"OFFICIAL_BOUNDARY_TESTS.json"),
                  launch_receipt_sha256=sha(AUDIT/"LAUNCH_RECEIPT.json"),
                  remote_release_utc=launch["remote_utc"], vpn_closed=True,
                  new_job_progress_queried=False, full103_diagnosis_complete=False,
                  training_authorized=False, inspection="local_only_post_submission_integrity")
    with (ROOT / "operator/LOCAL_EXECUTION_ACCEPTANCE_20261014.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
