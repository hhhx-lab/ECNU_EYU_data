"""Freeze corrected runtime and explicit user authorization, preserving r1."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT.parents[2]
META = dict(artifact_status="experimental_unvalidated", operator_approved=False,
            formal_gate_status="not_run_not_passed")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_path = ROOT / "sources/r1_preserved/CPU_EXECUTION_PLAN.json"
    plan = json.loads(old_path.read_text())
    contract = S2 / "05_documents/02_plans/20261009_k1_small_lesion_nnunet_refinement/revisions/20261014/P0_CPU_EMPTY_REGION_RECOVERY_R2.md"
    new_contract = ROOT / "sources/P0_CPU_EMPTY_REGION_RECOVERY_R2.md"
    if new_contract.exists() or (ROOT / "CPU_EXECUTION_PLAN.json").exists():
        raise RuntimeError("candidate already frozen; no overwrite")
    shutil.copyfile(contract, new_contract)
    static = list(plan["static_sha256"]) + [str(p.relative_to(ROOT)) for p in (ROOT / "sources/r1_preserved").iterdir()]
    static += ["sources/P0_CPU_EMPTY_REGION_RECOVERY_R2.md", "runtime/boundary_tests.py"]
    plan.update(created_local_utc=datetime.now(timezone.utc).isoformat(),
                remote_root="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/"+ROOT.name,
                job_name="s2k1p0cpu2", parent_plan_sha256=sha(old_path), parent_job_id="4051333",
                authorization_source="direct_user_20261014_fix_empty_regions_real_tests_independent_attempt",
                mandatory_gate="official_boundary_tests_and_real_fixed103_subset_before_held_submission",
                zero_region_policy="zero refs: FP all pred components; zero preds: all ref FN; both empty:0",
                static_sha256={name: sha(ROOT/name) for name in static})
    with (ROOT / "CPU_EXECUTION_PLAN.json").open("x") as handle:
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for path in (ROOT / "runtime").glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    print(json.dumps(dict(**META, status="frozen", static_files=len(static),
                         plan_sha256=sha(ROOT/"CPU_EXECUTION_PLAN.json"))))


if __name__ == "__main__":
    main()
