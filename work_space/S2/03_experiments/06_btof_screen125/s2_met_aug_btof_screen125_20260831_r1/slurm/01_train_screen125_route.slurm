#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1/logs/%x_%j.err

set -euo pipefail
umask 077

ROUTE_TOKEN="${1:?route token is required}"
case "${ROUTE_TOKEN}" in
    B|C40|D|E_SNFH|F) ;;
    *) echo "unsupported route token: ${ROUTE_TOKEN}" >&2; exit 2 ;;
esac

ROOT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1"
PARENT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1"
REPO="${ROOT}/runtime/repository"
DISCOVERY="${ROOT}/runtime/trainer_discovery"
PYTHON="/public/home/zqchen/.conda/envs/segmamba/bin/python"
SOURCE="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency"
DATASET="Dataset264_BraTS2026_MET_Completion"
TRAINER="nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune"
ROUTE_ROOT="${PARENT}/route_rebuild_20260812_r1"
CAL_ROOT="${PARENT}/calibration_rebuild_20260819_r10"
PLAN="${ROOT}/SCREEN125_EXECUTION_PLAN_20260831_R1.json"
SCREEN_CONFIG="${ROOT}/config/SCREEN_ROUTE_${ROUTE_TOKEN}.json"
RESULTS_ROOT="${ROOT}/results/${ROUTE_TOKEN}"
RESULT_FOLD="${RESULTS_ROOT}/${DATASET}/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
STARTED="${ROOT}/evidence/${ROUTE_TOKEN}_TRAINING_STARTED.json"
FAILED="${ROOT}/evidence/${ROUTE_TOKEN}_TRAINING_FAILED.json"
VALIDATION="${ROOT}/evidence/${ROUTE_TOKEN}_TRAINING_VALIDATION.json"
COMPLETE="${ROOT}/evidence/${ROUTE_TOKEN}_TRAINING_COMPLETE.ok"

write_failure() {
    local rc="$?"
    trap - ERR
    if [[ ! -e "${FAILED}" && ! -e "${COMPLETE}" ]]; then
        "${PYTHON}" - "${FAILED}" "${rc}" "${SLURM_JOB_ID:-unknown}" "${ROUTE_TOKEN}" <<'PY'
import datetime as dt
import json
import os
import sys

path, rc, job_id, route = sys.argv[1:]
payload = {
    "artifact_status": "experimental_unvalidated",
    "formal_gate_status": "not_run_not_passed",
    "job_id": job_id,
    "operator_approved": False,
    "return_code": int(rc),
    "route_token": route,
    "status": "failed",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
PY
    fi
    exit "${rc}"
}
trap write_failure ERR

[[ -f "${ROOT}/evidence/REMOTE_STATIC_PREFLIGHT.json" ]] || { echo "remote static preflight missing" >&2; exit 2; }
[[ -f "${SCREEN_CONFIG}" && -f "${PLAN}" ]] || { echo "screening config or plan missing" >&2; exit 2; }
[[ ! -e "${STARTED}" && ! -e "${FAILED}" && ! -e "${VALIDATION}" && ! -e "${COMPLETE}" ]] || {
    echo "route evidence already exists; refusing duplicate execution" >&2
    exit 2
}
[[ ! -e "${RESULT_FOLD}" ]] || { echo "exclusive result fold already exists" >&2; exit 2; }

module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${DISCOVERY}:${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_extTrainer="${DISCOVERY}"
export nnUNet_raw="${SOURCE}/nnUNet_raw"
export nnUNet_preprocessed="${PARENT}/data/nnUNet_preprocessed"
export nnUNet_results="${RESULTS_ROOT}"
export BRATS_S2_REPO_DIR="${REPO}"
export BRATS_SPLIT_DIR="${PARENT}/inputs/splits"
export S2_EXPERIMENT_MODE="met_aug_btof_screen125_20260831_r1"
export S2_CONTINUE="0"
export S2_SKIP_COMPLETED="0"
export S2_MET_AUG_ENABLE="1"
export S2_MET_AUG_COMPONENT_MANIFEST="${ROUTE_ROOT}/component_pool/component_manifest.json"
export S2_MET_AUG_ROUTE_CONFIG="${CAL_ROOT}/FIX_V3_ROUTE_CONFIG_SCHEMA5.json"
export S2_MET_AUG_VALID_MASK_MANIFEST="${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json"
export S2_MET_AUG_EMERGENCY_DECISION="${CAL_ROOT}/FIX_V3_EMERGENCY_DECISION.json"
export S2_MET_AUG_FIX_V3_CALIBRATION="${CAL_ROOT}/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
export S2_MET_AUG_ORIGINAL_E_CHECKPOINT="${PARENT}/frozen/original_e/checkpoint_final.pth"
export S2_MET_AUG_FIX_V2_FAILURE_AUDIT="${PARENT}/frozen/fix_v2/QC_HOLDOUT_RUN_ATTEMPT_02_FAILED.json"
export S2_MET_AUG_G1_CODE_DIR="${PARENT}/runtime/g1_archive/BraTS_2023_2024_solutions-main_3/Segmentation_Tasks/GliGAN"
export S2_MET_AUG_G1_CHECKPOINT_ROOT="${PARENT}/frozen/g1"
export S2_MET_AUG_G1_CHECKPOINT_SELECTION="${PARENT}/frozen/g1/checkpoint_selection.json"
export S2_MET_AUG_G2_QC_GATE="${PARENT}/frozen/g1/g2_diffusion_qc_gate.json"
export S2_MET_AUG_AUDIT_PATH="${RESULT_FOLD}/met_aug_events.jsonl"
export S2_MET_AUG_SCREEN_CONFIG="${SCREEN_CONFIG}"
export S2_MET_AUG_SCREEN_PLAN="${PLAN}"
export S2_MET_AUG_TARGET_LESION_COUNTS="${ROOT}/config/TRAIN_NATIVE_LESION_COUNTS.json"
export S2_PAIRED_TRAINING_SEED="20260724"
export S2_COMPLETION_EPOCHS="125"
export S2_COMPLETION_INITIAL_LR="0.001"
export S2_COMPLETION_SAVE_EVERY="25"
export S2_SCREEN_EPOCHS="125"
export S2_SCREEN_INITIAL_LR="0.001"
export S2_SCREEN_SAVE_EVERY="25"
export S2_FOCAL_GAMMA="2.0"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export nnUNet_compile="0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON}" - "${ROOT}" "${SCREEN_CONFIG}" "${STARTED}" "${SLURM_JOB_ID:-unknown}" "${ROUTE_TOKEN}" "${DISCOVERY}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

root, config_path, started_path = map(Path, sys.argv[1:4])
job_id, route_token, discovery = sys.argv[4], sys.argv[5], Path(sys.argv[6])
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("screening job requires exactly one scheduler-visible CUDA GPU")
device_name = torch.cuda.get_device_name(0)
if "A100" not in device_name:
    raise SystemExit(f"screening job requires A100, got {device_name}")
trainer_name = "nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune"
trainer_class = recursive_find_python_class(
    str(discovery), trainer_name, current_module=None,
    base_folder=str(discovery), verbose=True, cleanup_imports_from_base_folder=False,
)
if trainer_class is None or trainer_class.__name__ != trainer_name:
    raise SystemExit("screening trainer discovery failed")
config = json.loads(config_path.read_text(encoding="utf-8"))
payload = {
    "artifact_status": "experimental_unvalidated",
    "checkpoint_epochs": [25, 50, 75, 100, 125],
    "device_name": device_name,
    "epochs": 125,
    "formal_gate_status": "not_run_not_passed",
    "initial_lr": 0.001,
    "job_id": job_id,
    "operator_approved": False,
    "pretrained_sha256": "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267",
    "resume": False,
    "route_id": config["route_id"],
    "route_token": route_token,
    "save_every": 25,
    "seed": 20260724,
    "status": "started",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "trainer_module": trainer_class.__module__,
}
fd = os.open(started_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
PY

nnUNetv2_train 264 3d_fullres 0 \
    -tr "${TRAINER}" \
    -num_gpus 1 \
    -pretrained_weights "${PARENT}/frozen/original_e/checkpoint_final.pth"

"${PYTHON}" - "${RESULT_FOLD}" "${VALIDATION}" "${COMPLETE}" "${SLURM_JOB_ID:-unknown}" "${ROUTE_TOKEN}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

fold, validation_path, complete_path = map(Path, sys.argv[1:4])
job_id, route_token = sys.argv[4:]
checkpoint_rows = []
for epoch in (25, 50, 75, 100, 125):
    path = fold / f"checkpoint_epoch_{epoch:04d}.pth"
    if not path.is_file():
        raise SystemExit(f"missing retained checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("current_epoch", -1)) != epoch:
        raise SystemExit(f"checkpoint epoch mismatch: {path}")
    logging = checkpoint.get("logging", {})
    ema = logging.get("ema_fg_dice", [])
    if len(ema) < epoch:
        raise SystemExit(f"checkpoint lacks EMA history: {path}")
    checkpoint_rows.append({
        "epoch": epoch,
        "ema_fg_dice": float(ema[epoch - 1]),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
best = sorted(checkpoint_rows, key=lambda row: (-row["ema_fg_dice"], row["epoch"]))[0]
payload = {
    "artifact_status": "experimental_unvalidated",
    "best_checkpoint": best,
    "candidate_checkpoints": checkpoint_rows,
    "checkpoint_selection_metric": "development_validation_ema_fg_dice",
    "checkpoint_selection_tie_break": "earliest_epoch",
    "completed_epochs": 125,
    "formal_gate_status": "not_run_not_passed",
    "job_id": job_id,
    "operator_approved": False,
    "route_token": route_token,
    "status": "pass",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
fd = os.open(validation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
fd = os.open(complete_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="ascii") as handle:
    handle.write("complete\n")
PY
