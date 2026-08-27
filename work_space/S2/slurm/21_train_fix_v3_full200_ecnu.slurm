#!/bin/bash
#SBATCH --job-name=s2f3full200
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/full200_training_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/full200_training_%j.err

set -euo pipefail
umask 077

ROOT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1"
REPO="${ROOT}/runtime/repository"
PYTHON="/public/home/zqchen/.conda/envs/segmamba/bin/python"
SOURCE="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency"
DATASET="Dataset264_BraTS2026_MET_Completion"
TRAINER="nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune"
ROUTE_ROOT="${ROOT}/route_rebuild_20260812_r1"
CAL_ROOT="${ROOT}/calibration_rebuild_20260812_r1"
RESULT_FOLD="${ROOT}/data/nnUNet_results/${DATASET}/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
FAILED="${ROOT}/evidence/FULL200_TRAINING_FAILED.json"
STARTED="${ROOT}/evidence/FULL200_TRAINING_STARTED.json"
VALIDATION="${ROOT}/evidence/FULL200_TRAINING_VALIDATION.json"
COMPLETE="${ROOT}/evidence/FULL200_TRAINING_COMPLETE.ok"
G1_CODE="${ROOT}/runtime/g1_archive/BraTS_2023_2024_solutions-main_3/Segmentation_Tasks/GliGAN"

write_failure() {
    local rc="$?"
    trap - ERR
    if [[ ! -e "${FAILED}" && ! -e "${COMPLETE}" ]]; then
        "${PYTHON}" - "${FAILED}" "${rc}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import json
import os
import sys

path, rc, job_id = sys.argv[1:]
payload = {
    "artifact_status": "experimental_unvalidated",
    "job_id": job_id,
    "return_code": int(rc),
    "status": "failed",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
PY
    fi
    exit "${rc}"
}
trap write_failure ERR

[[ -f "${ROOT}/evidence/FULL200_SMOKE_COMPLETE.ok" ]] || { echo "A100 smoke is incomplete" >&2; exit 2; }
[[ -f "${ROOT}/evidence/FULL200_SMOKE_VALIDATION.json" ]] || { echo "A100 smoke validation is missing" >&2; exit 2; }
[[ -f "${CAL_ROOT}/CALIBRATION_REBUILD_COMPLETE.ok" ]] || { echo "Calibration rebuild is incomplete" >&2; exit 2; }
[[ ! -e "${FAILED}" && ! -e "${STARTED}" && ! -e "${VALIDATION}" && ! -e "${COMPLETE}" ]] || {
    echo "Training evidence already exists; refusing duplicate submission" >&2
    exit 2
}
[[ ! -e "${RESULT_FOLD}" ]] || { echo "Exclusive training fold already exists: ${RESULT_FOLD}" >&2; exit 2; }

module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:${PATH}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_extTrainer="${REPO}/custom_nnunet"
export nnUNet_raw="${SOURCE}/nnUNet_raw"
export nnUNet_preprocessed="${ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${ROOT}/data/nnUNet_results"
export BRATS_S2_REPO_DIR="${REPO}"
export BRATS_SPLIT_DIR="${ROOT}/inputs/splits"
export S2_EXPERIMENT_MODE="met_aug_route_a_fix_v3_emergency"
export S2_CONTINUE="0"
export S2_SKIP_COMPLETED="0"
export S2_MET_AUG_ENABLE="1"
export S2_MET_AUG_COMPONENT_MANIFEST="${ROUTE_ROOT}/component_pool/component_manifest.json"
export S2_MET_AUG_ROUTE_CONFIG="${CAL_ROOT}/FIX_V3_ROUTE_CONFIG_SCHEMA5.json"
export S2_MET_AUG_VALID_MASK_MANIFEST="${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json"
export S2_MET_AUG_EMERGENCY_DECISION="${CAL_ROOT}/FIX_V3_EMERGENCY_DECISION.json"
export S2_MET_AUG_FIX_V3_CALIBRATION="${CAL_ROOT}/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
export S2_MET_AUG_ORIGINAL_E_CHECKPOINT="${ROOT}/frozen/original_e/checkpoint_final.pth"
export S2_MET_AUG_FIX_V2_FAILURE_AUDIT="${ROOT}/frozen/fix_v2/QC_HOLDOUT_RUN_ATTEMPT_02_FAILED.json"
export S2_MET_AUG_G1_CODE_DIR="${G1_CODE}"
export S2_MET_AUG_G1_CHECKPOINT_ROOT="${ROOT}/frozen/g1"
export S2_MET_AUG_G1_CHECKPOINT_SELECTION="${ROOT}/frozen/g1/checkpoint_selection.json"
export S2_MET_AUG_G2_QC_GATE="${ROOT}/frozen/g1/g2_diffusion_qc_gate.json"
export S2_MET_AUG_AUDIT_PATH="${RESULT_FOLD}/met_aug_events.jsonl"
export S2_PAIRED_TRAINING_SEED="20260724"
export S2_COMPLETION_EPOCHS="200"
export S2_COMPLETION_INITIAL_LR="0.001"
export S2_COMPLETION_SAVE_EVERY="25"
export S2_FOCAL_GAMMA="2.0"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export nnUNet_compile="0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON}" - "${ROOT}" "${RESULT_FOLD}" "${STARTED}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
result_fold = Path(sys.argv[2])
started_path = Path(sys.argv[3])
job_id = sys.argv[4]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("full-200 requires exactly one scheduler-visible CUDA GPU")
device_name = torch.cuda.get_device_name(0)
if "A100" not in device_name:
    raise SystemExit(f"full-200 requires A100, got {device_name}")
smoke = json.load(open(root / "evidence/FULL200_SMOKE_VALIDATION.json", encoding="utf-8"))
if smoke.get("status") != "pass" or int(smoke.get("committed_events", 0)) < 1:
    raise SystemExit("full-200 smoke contract did not pass")
if result_fold.exists():
    raise SystemExit("exclusive full-200 result fold already exists")
train_ids = [line.strip() for line in (root / "inputs/splits/train_fixed.txt").read_text().splitlines() if line.strip()]
val_ids = [line.strip() for line in (root / "inputs/splits/val_fixed.txt").read_text().splitlines() if line.strip()]
if len(train_ids) != 1035 or len(val_ids) != 103 or set(train_ids) & set(val_ids):
    raise SystemExit("fixed 1035/103 split contract drifted")
splits = json.load(open(root / "data/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion/splits_final.json", encoding="utf-8"))
if len(splits) != 1 or splits[0].get("train") != train_ids or splits[0].get("val") != val_ids:
    raise SystemExit("nnUNet splits_final.json does not match fixed 1035/103 lists")
checkpoint_path = root / "frozen/original_e/checkpoint_final.pth"
if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267":
    raise SystemExit("frozen original E SHA256 drifted")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if int(checkpoint.get("current_epoch", -1)) != 200:
    raise SystemExit("frozen original E current_epoch drifted")
for name, tensor in checkpoint.get("network_weights", {}).items():
    if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
        raise SystemExit(f"non-finite pretrained tensor: {name}")
payload = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "status": "started",
    "job_id": job_id,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "device_name": device_name,
    "mode": "fresh_second_stage_0_to_200",
    "resume": False,
    "epochs": 200,
    "seed": 20260724,
    "initial_lr": 0.001,
    "save_every": 25,
    "p_select": 0.20,
    "fixed_train_count": 1035,
    "fixed_validation_count": 103,
    "pretrained_sha256": "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267",
}
fd = os.open(started_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
PY

nnUNetv2_train \
    264 3d_fullres 0 \
    -tr "${TRAINER}" \
    -num_gpus 1 \
    -pretrained_weights "${ROOT}/frozen/original_e/checkpoint_final.pth"

"${PYTHON}" - "${RESULT_FOLD}" "${VALIDATION}" "${COMPLETE}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

fold = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
marker_path = Path(sys.argv[3])
job_id = sys.argv[4]
checkpoint_path = fold / "checkpoint_final.pth"
summary_path = fold / "validation/summary.json"
if not checkpoint_path.is_file() or not summary_path.is_file():
    raise SystemExit("full-200 checkpoint or fixed-103 summary is missing")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if int(checkpoint.get("current_epoch", -1)) != 200:
    raise SystemExit("full-200 checkpoint current_epoch is not 200")
if checkpoint.get("trainer_name") != "nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune":
    raise SystemExit("full-200 trainer identity drifted")
for collection in ("network_weights", "optimizer_state", "grad_scaler_state"):
    value = checkpoint.get(collection)
    if value is None:
        raise SystemExit(f"full-200 checkpoint lacks {collection}")
def walk(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from walk(child)
    elif torch.is_tensor(value):
        yield value
for collection in ("network_weights", "optimizer_state", "grad_scaler_state"):
    for tensor in walk(checkpoint[collection]):
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            raise SystemExit(f"non-finite tensor in {collection}")
predictions = sorted((fold / "validation").glob("*.nii.gz"))
if len(predictions) != 103:
    raise SystemExit(f"fixed validation prediction count is {len(predictions)}, expected 103")
audit_path = fold / "met_aug_events.jsonl"
if not audit_path.is_file():
    raise SystemExit("MET-AUG training audit is missing")
payload = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "status": "pass",
    "job_id": job_id,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "training_complete": True,
    "completed_epochs": 200,
    "checkpoint_current_epoch": 200,
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "fixed_validation_prediction_count": 103,
    "validation_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "met_aug_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    "formal_gate_status": "not_run_not_passed",
}
fd = os.open(validation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write("experimental_unvalidated fresh Fix-v3 full-200 training pass\n")
PY

trap - ERR
echo "S2_FIX_V3_FULL200_TRAINING_PASS"
