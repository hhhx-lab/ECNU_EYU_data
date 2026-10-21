"""Freeze an independent CPU diagnosis contract from the validated inventory."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2 = ROOT.parents[2]
SNAPSHOT = ROOT / "operator/readonly_inventory_20261013T125306Z"
META = dict(artifact_status="experimental_unvalidated", operator_approved=False,
            formal_gate_status="not_run_not_passed")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    inventory = json.loads((SNAPSHOT / "REMOTE_INVENTORY.json").read_text())
    validation = json.loads((SNAPSHOT / "INVENTORY_VALIDATION.json").read_text())
    assert validation["status"] == "pass"
    assert validation["original_inventory_sha256"] == sha(SNAPSHOT / "REMOTE_INVENTORY.json")
    assert not (ROOT / "CPU_EXECUTION_PLAN.json").exists()
    kroot = S2 / "03_experiments/12_k1245_rc_aware"
    k1 = kroot / "s2_met_aug_k1_selected103_official_met_20261008_r1"
    k4 = kroot / "s2_met_aug_k24_selected103_official_met_20261002_r1"
    copies = {
        "manifests/FIXED103_REAL_T2W_CASES.csv": k1 / "manifests/FIXED103_REAL_T2W_CASES.csv",
        "manifests/INPUT_REFERENCE_SHA256.json": k1 / "manifests/INPUT_REFERENCE_SHA256.json",
        "sources/INVENTORY_VALIDATION.json": SNAPSHOT / "INVENTORY_VALIDATION.json",
    }
    for method, source in (("K1",k1),("K4",k4)):
        root = source / "cohort/fixed103/models" / method
        for name in ("panoptica_evaluation_summary.json", "leaderboard_metrics.csv"):
            copies["sources/"+method+"/"+name] = root / "evaluation" / name
        copies["sources/"+method+"/PREDICTION_SHA256.json"] = root / "PREDICTION_SHA256.json"
    for relative, source in copies.items():
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)
        assert sha(target) == sha(source)
    with (ROOT / "manifests/FIXED103_REAL_T2W_CASES.csv").open() as handle:
        cases = [r["source_case_id"] for r in csv.DictReader(handle)]
    software = {r["path"]:r["sha256"] for r in inventory["software"].values()}
    evaluator = inventory["official_evaluator"]
    eval_root = Path(evaluator["root"])
    for name, text in evaluator["python_sources"].items():
        software[str(eval_root / name)] = hashlib.sha256(text.encode()).hexdigest()
    for name, text in evaluator["brats_sources"].items():
        software[str(eval_root.parent / "brats_evaluation" / name)] = hashlib.sha256(text.encode()).hexdigest()
    contract = S2 / "05_documents/02_plans/20261009_k1_small_lesion_nnunet_refinement/revisions/20261013/P0_CPU_CONTRACT_20261013.md"
    copies["sources/P0_CPU_CONTRACT_20261013.md"] = contract
    shutil.copyfile(contract, ROOT / "sources/P0_CPU_CONTRACT_20261013.md")
    static_paths = list(copies) + ["runtime/cpu_roundtrip.py", "runtime/official_identity.py", "slurm/cpu_diagnosis.slurm"]
    deps = {r["path"]:r["sha256"] for r in inventory["frozen_files"]}
    plan = dict(**META, created_local_utc=datetime.now(timezone.utc).isoformat(),
        remote_root="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/"+ROOT.name,
        cohort_root="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103",
        job_name="s2k1p0cpu1", phase="P0-B/C_partial_fixed103_and_K1_K4_identity",
        plans_path=next(p for p in deps if p.endswith("/nnUNetPlans.json")),
        dataset_path=next(p for p in deps if p.endswith("/dataset.json")),
        eval_python="/public/home/zqchen/.conda/envs/brats_eval/bin/python",
        case_ids=cases, input_reference_sha256=json.loads((ROOT/"manifests/INPUT_REFERENCE_SHA256.json").read_text())["files"],
        prediction_roots={m:r["root"]+"/prediction" for m,r in inventory["prediction_assets"].items()},
        prediction_sha256={m:json.loads((ROOT/"sources"/m/"PREDICTION_SHA256.json").read_text())["files"] for m in ("K1","K4")},
        dependency_sha256=deps, software_sha256=software,
        static_sha256={name:sha(ROOT/name) for name in static_paths},
        small_reference_contract="original_full_volume_region_component_physical_volume_mm3_lt27",
        region_labels=dict(wt=[1,2,3],tc=[1,3],et=[3],rc=[4]),
        matched_dice_threshold=.2, matching_threshold=.000001,
        instance_transport_classification="pending_independent_mask_diagnosis",
        cpu_count=4, memory_gb=64, gpu_count=0, walltime="1-00:00:00", training=False,
        exploratory_only=True, inventory_sha256=sha(SNAPSHOT/"REMOTE_INVENTORY.json"))
    with (ROOT / "CPU_EXECUTION_PLAN.json").open("x") as handle:
        json.dump(plan,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status="prepared",plan_sha256=sha(ROOT/"CPU_EXECUTION_PLAN.json"),files=len(static_paths),software_files=len(software))))


if __name__ == "__main__":
    main()
