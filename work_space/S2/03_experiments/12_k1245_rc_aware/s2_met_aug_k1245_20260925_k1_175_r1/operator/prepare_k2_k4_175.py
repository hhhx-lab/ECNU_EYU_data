"""Materialize independent K2 and K4 175-epoch submission contracts once."""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path

K1 = Path(__file__).resolve().parents[1]
BASE = K1.parent
K2 = BASE / "s2_met_aug_k1245_20260929_k2_175_r1"
K4 = BASE / "s2_met_aug_k1245_20260929_k4_175_r4"
K4_R5 = BASE / "s2_met_aug_k1245_20260929_k4_175_r5"
K1_R2 = BASE / "s2_met_aug_k1245_20260929_k1_175_r2"
MKL_RECOVERY = {
    "K1": (K1_R2, BASE / "s2_met_aug_k1245_20260929_k1_175_r3",
           "K1245_K1_175_EXECUTION_PLAN_20260929_R2.json", "K1245_K1_175_EXECUTION_PLAN_20260929_R3.json",
           "01_train_k1_175_r2.slurm", "01_train_k1_175_r3.slurm", "s2k1train175r2", "s2k1train175r3", "3827458"),
    "K2": (K2, BASE / "s2_met_aug_k1245_20260929_k2_175_r2",
           "K1245_K2_175_EXECUTION_PLAN_20260929_R1.json", "K1245_K2_175_EXECUTION_PLAN_20260929_R2.json",
           "01_train_k2_175_r1.slurm", "01_train_k2_175_r2.slurm", "s2k2train175r1", "s2k2train175r2", "3827469"),
    "K4": (K4_R5, BASE / "s2_met_aug_k1245_20260929_k4_175_r6",
           "K1245_K4_175_EXECUTION_PLAN_20260929_R5.json", "K1245_K4_175_EXECUTION_PLAN_20260929_R6.json",
           "01_train_k4_175_r5.slurm", "01_train_k4_175_r6.slurm", "s2k4train175r5", "s2k4train175r6", "3827503"),
}
R3 = BASE / "s2_met_aug_k1245_20260925_r3_175"
TRAINER = "nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175.py"
META = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
}
E_SHA = "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
CAL_SHA = "68a5d8a6c501804b7a47d626e43a58ff71b781154c0cc955a9c5a5d5b786d37f"
RC_MANIFEST = "a9b74b410838f8fcba50d66d52c7751bef2b07ddc94f2f774eabaed8c98701ac"


def put(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)


def put_json(path: Path, payload: dict) -> None:
    put(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def prepare_k2() -> None:
    plan_name = "K1245_K2_175_EXECUTION_PLAN_20260929_R1.json"
    root = K2.name
    slurm = (K1 / "slurm/01_train_k1_175_r1.slurm").read_text()
    slurm = slurm.replace(K1.name, root)
    slurm = slurm.replace("K1245_K1_175_EXECUTION_PLAN_20260925_R1.json", plan_name)
    slurm = slurm.replace("s2k1train175r1", "s2k2train175r1")
    slurm = slurm.replace("K1_175", "K2_175")
    slurm = slurm.replace("K1245_ROUTE_K1", "K1245_ROUTE_K2")
    slurm = slurm.replace("/results/K1", "/results/K2")
    slurm = slurm.replace("_K1\"", "_K2\"")
    slurm = slurm.replace("met_aug_k1245_20260925_k1_175_r1_K2", "met_aug_k1245_20260929_k2_175_r1_K2")
    slurm = slurm.replace("'K1'", "'K2'")
    slurm = slurm.replace('"route":"K1"', '"route":"K2"')
    slurm = slurm.replace('"K1"', '"K2"')
    assert K1.name not in slurm and "s2k1train175r1" not in slurm
    assert "--gres=gpu:1" in slurm and "S2_K1245_EPOCHS=175" in slurm
    assert '"route":"K2"' in slurm and "S2_MET_AUG_FIX_V3_CALIBRATION" in slurm

    plan = {
        **META,
        "experiment_id": root,
        "submission_attempt": "k2_training_r1",
        "parent_plan": "K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.md",
        "route_id": "K2",
        "route": "E-SNFH* + D* + X1 + X2 + B1 + B2",
        "epochs": 175,
        "checkpoint_epochs": [25, 50, 75, 100, 125, 150, 175],
        "train_cases": 1035,
        "validation_cases": 103,
        "seed": 20260724,
        "device": "one A100 GPU",
        "ddp": False,
        "torch_compile": False,
        "augmentation_workers": 0,
        "pretrained_checkpoint_sha256": E_SHA,
        "calibration_sha256": CAL_SHA,
        "rc27_manifest_identity_sha256": RC_MANIFEST,
        "f_strict_prereg_gate": False,
        "d_empirical_source": "20260923_r2/pools/K1245_POOL_AND_EMPIRICAL_VALIDATION.json",
        "d_config_source_sha256": "ab4e1001e4e2b57f64acaaf8340558a3694add6fb9f78df5c2d36bbbc7aec815",
        "d_rank_volume_strata": 7,
        "d_nn_distance_samples": 6887,
        "second_tumour_probability": 0.20,
        "rc_smoke_source": "K1 R3J job 3740743, shared RC24 engineering smoke, automatic 24/24",
        "x2_schedule": "original preregistered linear decay from epoch 83 to zero at epoch 125; remains zero to epoch 175",
        "selection": {
            "metric": "fixed103 development validation ema_fg_dice",
            "tie_break": "earliest epoch",
            "fixed103_for_selection_and_report": True,
        },
        "exploratory_only": True,
        "k5_authorization": False,
        "notes": [
            "The RC24 smoke is a shared technical gate, not a K2-specific D-route smoke or scientific pass.",
            "K2 also requires independent static validation of the frozen D empirical config before launch.",
            "The user authorized K1/K2/K4 training on 2026-09-29; no old manual CSV is rewritten.",
        ],
    }
    auth = {
        **META,
        "authorization_type": "user_explicit_engineering_authorization",
        "route_id": "K2",
        "status": "pass",
        "training_authorization": True,
        "epochs": 175,
        "F_STRICT_PREREG_GATE": False,
        "smoke_job_id": "3740743",
        "smoke_automatic_status": "shared_RC24_pass_24_of_24",
        "manual_review_status": "user_declared_shared_RC24_review_pass; no_K2_specific_montage_claim",
        "d_empirical_contract": "subject_to_remote_static_preflight",
        "k5_authorization": False,
        "note": "Engineering submission only; not a formal or scientific gate and not K2-specific visual evidence.",
    }
    put_json(K2 / plan_name, plan)
    put_json(K2 / "evidence/K2_175_TRAINING_AUTHORIZATION.json", auth)
    put(K2 / "slurm/01_train_k2_175_r1.slurm", slurm)
    (K2 / "runtime").mkdir()
    shutil.copyfile(K1 / "runtime" / TRAINER, K2 / "runtime" / TRAINER)


def prepare_k1_r2() -> None:
    old_plan = K1 / "K1245_K1_175_EXECUTION_PLAN_20260925_R1.json"
    plan_name = "K1245_K1_175_EXECUTION_PLAN_20260929_R2.json"
    plan = copy.deepcopy(json.loads(old_plan.read_text()))
    plan.update({
        "experiment_id": K1_R2.name,
        "submission_attempt": "k1_training_r2",
        "revision_reason": "R1 remote staging copy timed out before sbatch; new isolated attempt uses read-only frozen pool symlink",
        "prior_submission_attempt": K1.name,
    })
    auth = copy.deepcopy(json.loads((K1 / "evidence/K1_175_TRAINING_AUTHORIZATION.json").read_text()))
    auth["carry_forward_to"] = K1_R2.name
    auth["note"] += " User reaffirmed K1 authorization on 2026-09-29."
    slurm = (K1 / "slurm/01_train_k1_175_r1.slurm").read_text()
    slurm = slurm.replace(K1.name, K1_R2.name)
    slurm = slurm.replace(old_plan.name, plan_name)
    slurm = slurm.replace("s2k1train175r1", "s2k1train175r2")
    slurm = slurm.replace("met_aug_k1245_20260925_k1_175_r1_K1", "met_aug_k1245_20260929_k1_175_r2_K1")
    assert K1.name not in slurm and "s2k1train175r1" not in slurm
    put_json(K1_R2 / plan_name, plan)
    put_json(K1_R2 / "evidence/K1_175_TRAINING_AUTHORIZATION.json", auth)
    put(K1_R2 / "slurm/01_train_k1_175_r2.slurm", slurm)
    (K1_R2 / "runtime").mkdir()
    shutil.copyfile(K1 / "runtime" / TRAINER, K1_R2 / "runtime" / TRAINER)


def prepare_k4() -> None:
    old_plan = R3 / "K1245_RC_AWARE_175_EXECUTION_PLAN_20260925_R3.json"
    old_slurm = R3 / "slurm/01_train_k4_175_r3.slurm"
    plan_name = "K1245_K4_175_EXECUTION_PLAN_20260929_R4.json"
    plan = copy.deepcopy(json.loads(old_plan.read_text()))
    plan.update({
        "experiment_id": K4.name,
        "submission_attempt": "r4_after_r3_qos_rejection",
        "revision_reason": "R3 held sbatch was rejected by QOS; a new isolated attempt retains the calibration fix",
        "prior_submission_attempt": R3.name,
        "route_id": "K4",
    })
    auth = copy.deepcopy(json.loads((R3 / "evidence/K4_175_TRAINING_AUTHORIZATION.json").read_text()))
    auth["carry_forward_to"] = K4.name
    auth["note"] += " User again explicitly authorized K4 on 2026-09-29."
    slurm = old_slurm.read_text().replace(R3.name, K4.name)
    slurm = slurm.replace("s2k4train175r3", "s2k4train175r4")
    slurm = slurm.replace(old_plan.name, plan_name)
    slurm = slurm.replace("met_aug_k1245_20260925_r3_175_K4", "met_aug_k1245_20260929_k4_175_r4_K4")
    assert R3.name not in slurm and "s2k4train175r3" not in slurm
    assert "--gres=gpu:1" in slurm and "S2_K1245_EPOCHS=175" in slurm
    assert '"route":"K4"' in slurm and "S2_MET_AUG_FIX_V3_CALIBRATION" in slurm
    put_json(K4 / plan_name, plan)
    put_json(K4 / "evidence/K4_175_TRAINING_AUTHORIZATION.json", auth)
    put(K4 / "slurm/01_train_k4_175_r4.slurm", slurm)
    (K4 / "runtime").mkdir()
    shutil.copyfile(K1 / "runtime" / TRAINER, K4 / "runtime" / TRAINER)


def prepare_k4_r5() -> None:
    old_plan = K4 / "K1245_K4_175_EXECUTION_PLAN_20260929_R4.json"
    plan_name = "K1245_K4_175_EXECUTION_PLAN_20260929_R5.json"
    plan = copy.deepcopy(json.loads(old_plan.read_text()))
    plan.update({
        "experiment_id": K4_R5.name,
        "submission_attempt": "r5_complete_frozen_small8_pool",
        "revision_reason": "R4 static load found R5 smoke manifest without referenced records; R5 training binds complete R2 small8 pool with identical manifest identity",
        "prior_submission_attempt": K4.name,
        "small8_pool_source": "s2_met_aug_k1245_20260923_r2/pools/small8",
        "small8_manifest_identity_sha256": "956e6384d47216e41b79905ad9c3c776dbc47422f445dadd9bd8e9c3b7a03244",
    })
    auth = copy.deepcopy(json.loads((K4 / "evidence/K4_175_TRAINING_AUTHORIZATION.json").read_text()))
    auth["carry_forward_to"] = K4_R5.name
    auth["note"] += " R4 static failure retained; R2 complete small8 pool has the same frozen manifest identity."
    slurm = (K4 / "slurm/01_train_k4_175_r4.slurm").read_text()
    slurm = slurm.replace(K4.name, K4_R5.name)
    slurm = slurm.replace("s2k4train175r4", "s2k4train175r5")
    slurm = slurm.replace(old_plan.name, plan_name)
    slurm = slurm.replace("met_aug_k1245_20260929_k4_175_r4_K4", "met_aug_k1245_20260929_k4_175_r5_K4")
    assert K4.name not in slurm and "s2k4train175r4" not in slurm
    put_json(K4_R5 / plan_name, plan)
    put_json(K4_R5 / "evidence/K4_175_TRAINING_AUTHORIZATION.json", auth)
    put(K4_R5 / "slurm/01_train_k4_175_r5.slurm", slurm)
    (K4_R5 / "runtime").mkdir()
    shutil.copyfile(K1 / "runtime" / TRAINER, K4_R5 / "runtime" / TRAINER)


def prepare_mkl_recovery(route: str) -> Path:
    source, target, old_plan, new_plan, old_slurm, new_slurm, old_job, new_job, failed_job = MKL_RECOVERY[route]
    if target.exists():
        raise SystemExit(f"{target} already exists; refusing to overwrite")
    plan = copy.deepcopy(json.loads((source / old_plan).read_text()))
    plan.update({
        "experiment_id": target.name,
        "submission_attempt": target.name.split("_")[-1] + "_mkl_gnu_recovery",
        "revision_reason": "Prior accepted job failed before STARTED: MKL_THREADING_LAYER=INTEL conflicted with libgomp; fresh attempt explicitly sets GNU",
        "prior_failed_job_id": failed_job,
        "prior_submission_attempt": source.name,
        "mkl_threading_layer": "GNU",
    })
    auth_name = f"{route}_175_TRAINING_AUTHORIZATION.json"
    auth = copy.deepcopy(json.loads((source / "evidence" / auth_name).read_text()))
    auth["carry_forward_to"] = target.name
    auth["note"] += " Prior job failed in MKL import before STARTED; user authorization is carried only as engineering execution approval."
    slurm = (source / "slurm" / old_slurm).read_text()
    slurm = slurm.replace(source.name, target.name)
    slurm = slurm.replace(source.name.removeprefix("s2_"), target.name.removeprefix("s2_"))
    slurm = slurm.replace(old_plan, new_plan).replace(old_job, new_job)
    anchor = "export PATH=\"/public/home/zqchen/.conda/envs/segmamba/bin:$PATH\""
    assert slurm.count(anchor) == 1
    slurm = slurm.replace(anchor, anchor + "\nexport MKL_THREADING_LAYER=GNU")
    assert source.name not in slurm and old_job not in slurm and "MKL_THREADING_LAYER=GNU" in slurm
    put_json(target / new_plan, plan)
    put_json(target / "evidence" / auth_name, auth)
    put(target / "slurm" / new_slurm, slurm)
    (target / "runtime").mkdir()
    shutil.copyfile(source / "runtime" / TRAINER, target / "runtime" / TRAINER)
    return target


if __name__ == "__main__":
    if sys.argv[1:] == ["k1-r2"]:
        if K1_R2.exists():
            raise SystemExit("K1 R2 already exists; refusing to overwrite")
        prepare_k1_r2()
        print(json.dumps({"K1": str(K1_R2)}, ensure_ascii=False))
    elif sys.argv[1:] == ["k4-r5"]:
        if K4_R5.exists():
            raise SystemExit("K4 R5 already exists; refusing to overwrite")
        prepare_k4_r5()
        print(json.dumps({"K4": str(K4_R5)}, ensure_ascii=False))
    elif len(sys.argv[1:]) == 2 and sys.argv[1] == "mkl-recovery" and sys.argv[2] in MKL_RECOVERY:
        route = sys.argv[2]
        print(json.dumps({route: str(prepare_mkl_recovery(route))}, ensure_ascii=False))
    elif not sys.argv[1:]:
        if K2.exists() or K4.exists():
            raise SystemExit("a new attempt already exists; refusing to overwrite")
        prepare_k2()
        prepare_k4()
        print(json.dumps({"K2": str(K2), "K4": str(K4)}, ensure_ascii=False))
    else:
        raise SystemExit("usage: prepare_k2_k4_175.py [k1-r2|k4-r5|mkl-recovery K1|K2|K4]")
