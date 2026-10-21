#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r2/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r2/logs/%x_%j.err

set -euo pipefail
umask 077

ROUTE_TOKEN="${1:?route token is required}"
case "${ROUTE_TOKEN}" in
    B|E_SNFH|F) ;;
    *) echo "unsupported extension route token: ${ROUTE_TOKEN}" >&2; exit 2 ;;
esac

ROOT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r2"
SOURCE_ROOT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1"
PARENT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1"
REPO="${ROOT}/runtime/repository"
DISCOVERY="${ROOT}/runtime/trainer_discovery"
PYTHON="/public/home/zqchen/.conda/envs/segmamba/bin/python"
SOURCE="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency"
DATASET="Dataset264_BraTS2026_MET_Completion"
SOURCE_TRAINER="nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune"
TRAINER="nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune"
ROUTE_ROOT="${PARENT}/route_rebuild_20260812_r1"
CAL_ROOT="${PARENT}/calibration_rebuild_20260819_r10"
SOURCE_PLAN="${SOURCE_ROOT}/SCREEN125_EXECUTION_PLAN_20260831_R1.json"
EXTENSION_PLAN="${ROOT}/EXTEND175_EXECUTION_PLAN_20260903_R2.json"
SCREEN_CONFIG="${SOURCE_ROOT}/config/SCREEN_ROUTE_${ROUTE_TOKEN}.json"
SOURCE_FOLD="${SOURCE_ROOT}/results/${ROUTE_TOKEN}/${DATASET}/${SOURCE_TRAINER}__nnUNetPlans__3d_fullres/fold_0"
SOURCE_CHECKPOINT="${SOURCE_FOLD}/checkpoint_epoch_0125.pth"
RESULTS_ROOT="${ROOT}/results/${ROUTE_TOKEN}"
RESULT_FOLD="${RESULTS_ROOT}/${DATASET}/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
STARTED="${ROOT}/evidence/${ROUTE_TOKEN}_EXTENSION_TRAINING_STARTED.json"
FAILED="${ROOT}/evidence/${ROUTE_TOKEN}_EXTENSION_TRAINING_FAILED.json"
VALIDATION="${ROOT}/evidence/${ROUTE_TOKEN}_EXTENSION_TRAINING_VALIDATION.json"
COMPLETE="${ROOT}/evidence/${ROUTE_TOKEN}_EXTENSION_TRAINING_COMPLETE.ok"

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

[[ -f "${ROOT}/evidence/REMOTE_STATIC_PREFLIGHT.json" ]] || { echo "extension preflight missing" >&2; exit 2; }
[[ -f "${SOURCE_CHECKPOINT}" && -f "${SCREEN_CONFIG}" && -f "${EXTENSION_PLAN}" ]] || { echo "extension source input missing" >&2; exit 2; }
[[ ! -e "${STARTED}" && ! -e "${FAILED}" && ! -e "${VALIDATION}" && ! -e "${COMPLETE}" ]] || { echo "extension evidence already exists" >&2; exit 2; }
[[ ! -e "${RESULT_FOLD}" ]] || { echo "exclusive extension fold already exists" >&2; exit 2; }

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
export S2_EXPERIMENT_MODE="met_aug_bef_extend175_20260903_r2"
export S2_CONTINUE="1"
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
export S2_MET_AUG_SCREEN_PLAN="${SOURCE_PLAN}"
export S2_MET_AUG_TARGET_LESION_COUNTS="${SOURCE_ROOT}/config/TRAIN_NATIVE_LESION_COUNTS.json"
export S2_MET_AUG_EXTENSION_PLAN="${EXTENSION_PLAN}"
export S2_MET_AUG_EXTENSION_SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}"
export S2_PAIRED_TRAINING_SEED="20260724"
export S2_FOCAL_GAMMA="2.0"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export nnUNet_compile="0"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

case "${ROUTE_TOKEN}" in
    B) EXPECTED_SOURCE_SHA="efc710f5d55537a88c7a801281c935b81c2331989d7a777c524ba243f0ca38c6" ;;
    E_SNFH) EXPECTED_SOURCE_SHA="1a930dda957da8a367239f386f8332bff2eec3a67b44efbb36cf0484a04d41b6" ;;
    F) EXPECTED_SOURCE_SHA="0198f206832e46509c9b8d00066f3b206bd7fe813426e872a5e756bd9c5021cf" ;;
esac
export S2_MET_AUG_EXTENSION_SOURCE_SHA256="${EXPECTED_SOURCE_SHA}"

"${PYTHON}" - "${SOURCE_CHECKPOINT}" "${RESULT_FOLD}" "${EXPECTED_SOURCE_SHA}" <<'PY'
import hashlib
import os
import shutil
import sys
from pathlib import Path

source, fold = Path(sys.argv[1]), Path(sys.argv[2])
expected = sys.argv[3]
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
if sha256_file(source) != expected:
    raise SystemExit("source checkpoint SHA256 drifted before seeding")
fold.mkdir(parents=True, exist_ok=False)
for name in ("checkpoint_latest.pth", "checkpoint_epoch_0125.pth"):
    destination = fold / name
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as output, source.open("rb") as input_handle:
        shutil.copyfileobj(input_handle, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if sha256_file(destination) != expected:
        raise SystemExit(f"seeded checkpoint SHA256 mismatch: {destination}")
PY

"${PYTHON}" - "${ROOT}" "${STARTED}" "${SLURM_JOB_ID:-unknown}" "${ROUTE_TOKEN}" "${SOURCE_CHECKPOINT}" "${EXPECTED_SOURCE_SHA}" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

import torch

root, started, source = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[5])
job_id, route, expected_sha = sys.argv[3], sys.argv[4], sys.argv[6]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("extension requires exactly one scheduler-visible CUDA GPU")
device_name = torch.cuda.get_device_name(0)
if "A100" not in device_name:
    raise SystemExit(f"extension requires A100, got {device_name}")
payload = {
    "artifact_status": "experimental_unvalidated",
    "candidate_total_epochs": [125, 150, 175],
    "device_name": device_name,
    "extension_epochs": 50,
    "extension_start_lr": 1.2965252773542109e-05,
    "formal_gate_status": "not_run_not_passed",
    "job_id": job_id,
    "operator_approved": False,
    "resume": True,
    "resume_optimizer_state": True,
    "route_token": route,
    "source_checkpoint": str(source),
    "source_checkpoint_sha256": expected_sha,
    "source_total_epoch": 125,
    "status": "started",
    "target_total_epoch": 175,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
fd = os.open(started, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
PY

nnUNetv2_train 264 3d_fullres 0 -tr "${TRAINER}" -num_gpus 1 -c

"${PYTHON}" - "${RESULT_FOLD}" "${VALIDATION}" "${COMPLETE}" "${SLURM_JOB_ID:-unknown}" "${ROUTE_TOKEN}" "${EXPECTED_SOURCE_SHA}" <<'PY'
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

fold, validation_path, complete_path = map(Path, sys.argv[1:4])
job_id, route, source_sha = sys.argv[4:]
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
rows = []
for epoch in (125, 150, 175):
    path = fold / f"checkpoint_epoch_{epoch:04d}.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("current_epoch", -1)) != epoch:
        raise SystemExit(f"checkpoint epoch mismatch: {path}")
    ema = checkpoint.get("logging", {}).get("ema_fg_dice", [])
    score = float(ema[epoch - 1])
    if not math.isfinite(score):
        raise SystemExit(f"nonfinite candidate EMA: {path}")
    digest = sha256_file(path)
    if epoch == 125 and digest != source_sha:
        raise SystemExit("copied epoch-125 checkpoint drifted")
    rows.append({"total_epoch": epoch, "ema_fg_dice": score, "path": str(path), "sha256": digest})
best = sorted(rows, key=lambda row: (-row["ema_fg_dice"], row["total_epoch"]))[0]
payload = {
    "artifact_status": "experimental_unvalidated",
    "best_checkpoint": best,
    "candidate_checkpoints": rows,
    "checkpoint_selection_metric": "development_validation_ema_fg_dice",
    "checkpoint_selection_tie_break": "earliest_total_epoch",
    "completed_total_epochs": 175,
    "extension_epochs": 50,
    "formal_gate_status": "not_run_not_passed",
    "job_id": job_id,
    "operator_approved": False,
    "route_token": route,
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
