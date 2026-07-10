#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export nnUNet_raw="${nnUNet_raw:-${REPO_DIR}/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_DIR}/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_DIR}/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${REPO_DIR}/data/splits}"
export BRATS_S2_REPO_DIR="${BRATS_S2_REPO_DIR:-${REPO_DIR}}"
export S2_DATASET_ID="${S2_DATASET_ID:-260}"
export S2_FOLD="${S2_FOLD:-0}"

S2_DATASET_NAME="${S2_DATASET_NAME:-Dataset260_BraTS2026_MET_RealOnly}"
S2_TRAINER="${S2_TRAINER:-nnUNetTrainerBraTS2026RC}"
S2_CONFIGURATION="${S2_CONFIGURATION:-3d_fullres}"
S2_CONTINUE="${S2_CONTINUE:-auto}"
S2_SKIP_COMPLETED="${S2_SKIP_COMPLETED:-1}"

if [[ ! "${S2_FOLD}" =~ ^[0-4]$ ]]; then
    echo "S2_FOLD must be one integer from 0 to 4, got: ${S2_FOLD}" >&2
    exit 2
fi
if [[ "${S2_CONTINUE}" != "auto" && "${S2_CONTINUE}" != "0" && "${S2_CONTINUE}" != "1" ]]; then
    echo "S2_CONTINUE must be auto, 0, or 1, got: ${S2_CONTINUE}" >&2
    exit 2
fi
if [[ "${S2_SKIP_COMPLETED}" != "0" && "${S2_SKIP_COMPLETED}" != "1" ]]; then
    echo "S2_SKIP_COMPLETED must be 0 or 1, got: ${S2_SKIP_COMPLETED}" >&2
    exit 2
fi

export nnUNet_extTrainer="${REPO_DIR}/custom_nnunet"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python - <<'PY'
import importlib.util
import os
import shutil
from pathlib import Path

repo_dir = Path(os.environ["BRATS_S2_REPO_DIR"])
src = repo_dir / "custom_nnunet" / "nnUNetTrainerBraTS2026RC.py"
spec = importlib.util.find_spec("nnunetv2")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Cannot find nnunetv2 in the active Python environment.")
pkg_root = Path(list(spec.submodule_search_locations)[0])
dst = pkg_root / "training" / "nnUNetTrainer" / src.name
if not dst.parent.exists():
    raise SystemExit(f"Cannot find nnU-Net trainer directory: {dst.parent}")
if not dst.exists() or dst.read_bytes() != src.read_bytes():
    shutil.copy2(src, dst)
print(f"Custom trainer ready: {dst}")
PY

TRAIN_FILE="${BRATS_SPLIT_DIR}/train_fold${S2_FOLD}.txt"
VAL_FILE="${BRATS_SPLIT_DIR}/val_fold${S2_FOLD}.txt"
RESULT_FOLD_DIR="${nnUNet_results}/${S2_DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__${S2_CONFIGURATION}/fold_${S2_FOLD}"
CHECKPOINT_LATEST="${RESULT_FOLD_DIR}/checkpoint_latest.pth"
CHECKPOINT_FINAL="${RESULT_FOLD_DIR}/checkpoint_final.pth"
VALIDATION_DIR="${RESULT_FOLD_DIR}/validation"
VALIDATION_SUMMARY="${VALIDATION_DIR}/summary.json"

if [[ ! -s "${TRAIN_FILE}" || ! -s "${VAL_FILE}" ]]; then
    echo "Missing fold-specific split files:" >&2
    echo "  ${TRAIN_FILE}" >&2
    echo "  ${VAL_FILE}" >&2
    echo "Run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi

echo "Starting BraTS2026 RC training..."
echo "Fold       : ${S2_FOLD}"
TRAIN_COUNT=$(wc -l < "${TRAIN_FILE}" | tr -d ' ')
VAL_COUNT=$(wc -l < "${VAL_FILE}" | tr -d ' ')
echo "Train split: ${TRAIN_FILE} (${TRAIN_COUNT})"
echo "Val split  : ${VAL_FILE} (${VAL_COUNT})"
echo "Results    : ${RESULT_FOLD_DIR}"

VALIDATION_PREDICTIONS=0
if [[ -d "${VALIDATION_DIR}" ]]; then
    VALIDATION_PREDICTIONS=$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
fi

TRAIN_CMD=(
    nnUNetv2_train
    "${S2_DATASET_ID}" "${S2_CONFIGURATION}" "${S2_FOLD}"
    -tr "${S2_TRAINER}"
    -num_gpus 1
)

if [[ -f "${CHECKPOINT_FINAL}" ]]; then
    if [[ "${S2_SKIP_COMPLETED}" == "1" && -f "${VALIDATION_SUMMARY}" && "${VALIDATION_PREDICTIONS}" == "${VAL_COUNT}" ]]; then
        echo "Fold ${S2_FOLD} is complete: final checkpoint and ${VALIDATION_PREDICTIONS}/${VAL_COUNT} validation predictions exist. Skipping."
        exit 0
    fi
    echo "Final checkpoint exists but validation output is missing or incomplete (${VALIDATION_PREDICTIONS}/${VAL_COUNT}); running validation only."
    TRAIN_CMD+=(--val)
elif [[ "${S2_CONTINUE}" == "1" ]]; then
    if [[ ! -f "${CHECKPOINT_LATEST}" ]]; then
        echo "S2_CONTINUE=1 but checkpoint_latest.pth is missing: ${CHECKPOINT_LATEST}" >&2
        exit 1
    fi
    TRAIN_CMD+=(--c)
elif [[ "${S2_CONTINUE}" == "auto" && -f "${CHECKPOINT_LATEST}" ]]; then
    echo "Found checkpoint_latest.pth; resuming fold ${S2_FOLD}."
    TRAIN_CMD+=(--c)
else
    echo "No resumable checkpoint found; starting fold ${S2_FOLD} from scratch."
fi

printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

if [[ ! -f "${CHECKPOINT_FINAL}" ]]; then
    echo "Fold ${S2_FOLD} command exited without checkpoint_final.pth: ${CHECKPOINT_FINAL}" >&2
    exit 1
fi
if [[ ! -f "${VALIDATION_SUMMARY}" ]]; then
    echo "Fold ${S2_FOLD} command exited without validation summary: ${VALIDATION_SUMMARY}" >&2
    exit 1
fi
VALIDATION_PREDICTIONS=$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
if [[ "${VALIDATION_PREDICTIONS}" != "${VAL_COUNT}" ]]; then
    echo "Fold ${S2_FOLD} validation output is incomplete: ${VALIDATION_PREDICTIONS}/${VAL_COUNT}" >&2
    exit 1
fi
echo "Fold ${S2_FOLD} complete: final checkpoint and ${VALIDATION_PREDICTIONS} validation predictions verified."
