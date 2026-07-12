#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2_EXPERIMENT_MODE="${S2_EXPERIMENT_MODE:-current}"

case "${S2_EXPERIMENT_MODE}" in
    current)
        DEFAULT_S2_DATASET_ID=263
        DEFAULT_S2_DATASET_NAME=Dataset263_BraTS2026_MET_RealOnly_Current
        DEFAULT_S2_TRAIN_COUNT=823
        DEFAULT_S2_VAL_COUNT=103
        ;;
    legacy)
        DEFAULT_S2_DATASET_ID=260
        DEFAULT_S2_DATASET_NAME=Dataset260_BraTS2026_MET_RealOnly
        DEFAULT_S2_TRAIN_COUNT=828
        DEFAULT_S2_VAL_COUNT=207
        ;;
    *)
        echo "S2_EXPERIMENT_MODE must be current or legacy, got: ${S2_EXPERIMENT_MODE}" >&2
        exit 2
        ;;
esac

export nnUNet_raw="${nnUNet_raw:-${REPO_DIR}/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_DIR}/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_DIR}/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${REPO_DIR}/data/splits/${S2_EXPERIMENT_MODE}}"
export BRATS_S2_REPO_DIR="${BRATS_S2_REPO_DIR:-${REPO_DIR}}"
export S2_DATASET_ID="${S2_DATASET_ID:-${DEFAULT_S2_DATASET_ID}}"
export S2_DATASET_NAME="${S2_DATASET_NAME:-${DEFAULT_S2_DATASET_NAME}}"
export NNUNET_DATASET_DIR="${NNUNET_DATASET_DIR:-${nnUNet_raw}/${S2_DATASET_NAME}}"
S2_EXPECTED_TRAIN_COUNT="${S2_EXPECTED_TRAIN_COUNT:-${DEFAULT_S2_TRAIN_COUNT}}"
S2_EXPECTED_VAL_COUNT="${S2_EXPECTED_VAL_COUNT:-${DEFAULT_S2_VAL_COUNT}}"

S2_TRAINER="${S2_TRAINER:-nnUNetTrainerBraTS2026RC}"
S2_CONFIGURATION="${S2_CONFIGURATION:-3d_fullres}"
S2_PREPROCESSED_DATA_IDENTIFIER="${S2_PREPROCESSED_DATA_IDENTIFIER:-nnUNetPlans_3d_fullres}"
S2_CONTINUE="${S2_CONTINUE:-auto}"
S2_SKIP_COMPLETED="${S2_SKIP_COMPLETED:-1}"

if [[ -n "${S2_FOLD:-}" && "${S2_FOLD}" != "0" ]]; then
    echo "S2 cross-validation is disabled; S2_FOLD must be unset or 0." >&2
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
if [[ "${S2_DATASET_NAME}" != Dataset${S2_DATASET_ID}_* ]]; then
    echo "S2_DATASET_NAME must start with Dataset${S2_DATASET_ID}_, got: ${S2_DATASET_NAME}" >&2
    exit 2
fi
if [[ "${S2_DATASET_ID}" != "${DEFAULT_S2_DATASET_ID}" || "${S2_DATASET_NAME}" != "${DEFAULT_S2_DATASET_NAME}" ]]; then
    echo "${S2_EXPERIMENT_MODE} mode is locked to dataset ${DEFAULT_S2_DATASET_ID}/${DEFAULT_S2_DATASET_NAME}." >&2
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

TRAIN_FILE="${BRATS_SPLIT_DIR}/train_fixed.txt"
VAL_FILE="${BRATS_SPLIT_DIR}/val_fixed.txt"
RESULT_FOLD_DIR="${nnUNet_results}/${S2_DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__${S2_CONFIGURATION}/fold_0"
CHECKPOINT_LATEST="${RESULT_FOLD_DIR}/checkpoint_latest.pth"
CHECKPOINT_FINAL="${RESULT_FOLD_DIR}/checkpoint_final.pth"
VALIDATION_DIR="${RESULT_FOLD_DIR}/validation"
VALIDATION_SUMMARY="${VALIDATION_DIR}/summary.json"

if [[ ! -s "${TRAIN_FILE}" || ! -s "${VAL_FILE}" ]]; then
    echo "Missing fixed split files:" >&2
    echo "  ${TRAIN_FILE}" >&2
    echo "  ${VAL_FILE}" >&2
    echo "Run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi

python "${REPO_DIR}/scripts/05_validate_fixed_split_cache.py" \
    --train-file "${TRAIN_FILE}" \
    --val-file "${VAL_FILE}" \
    --dataset-dir "${NNUNET_DATASET_DIR}" \
    --preprocessed-dir "${nnUNet_preprocessed}/${S2_DATASET_NAME}/${S2_PREPROCESSED_DATA_IDENTIFIER}" \
    --output-json "${BRATS_SPLIT_DIR}/fixed_split_cache_audit.json"

echo "Starting BraTS2026 RC training..."
echo "Mode       : ${S2_EXPERIMENT_MODE}"
echo "Split      : fixed train/validation (nnU-Net internal key: fold_0)"
TRAIN_COUNT=$(wc -l < "${TRAIN_FILE}" | tr -d ' ')
VAL_COUNT=$(wc -l < "${VAL_FILE}" | tr -d ' ')
echo "Train split: ${TRAIN_FILE} (${TRAIN_COUNT})"
echo "Val split  : ${VAL_FILE} (${VAL_COUNT})"
echo "Results    : ${RESULT_FOLD_DIR}"
if [[ "${TRAIN_COUNT}" != "${S2_EXPECTED_TRAIN_COUNT}" || "${VAL_COUNT}" != "${S2_EXPECTED_VAL_COUNT}" ]]; then
    echo "Fixed split count mismatch for ${S2_EXPERIMENT_MODE}: expected ${S2_EXPECTED_TRAIN_COUNT}/${S2_EXPECTED_VAL_COUNT}, got ${TRAIN_COUNT}/${VAL_COUNT}" >&2
    exit 1
fi

VALIDATION_PREDICTIONS=0
if [[ -d "${VALIDATION_DIR}" ]]; then
    VALIDATION_PREDICTIONS=$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
fi

TRAIN_CMD=(
    nnUNetv2_train
    "${S2_DATASET_ID}" "${S2_CONFIGURATION}" 0
    -tr "${S2_TRAINER}"
    -num_gpus 1
)

if [[ -f "${CHECKPOINT_FINAL}" ]]; then
    if [[ "${S2_SKIP_COMPLETED}" == "1" && -f "${VALIDATION_SUMMARY}" && "${VALIDATION_PREDICTIONS}" == "${VAL_COUNT}" ]]; then
        echo "Fixed-split model is complete: final checkpoint and ${VALIDATION_PREDICTIONS}/${VAL_COUNT} validation predictions exist. Skipping."
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
    echo "Found checkpoint_latest.pth; resuming fixed-split training."
    TRAIN_CMD+=(--c)
else
    echo "No resumable checkpoint found; starting fixed-split training from scratch."
fi

printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

if [[ ! -f "${CHECKPOINT_FINAL}" ]]; then
    echo "Fixed-split command exited without checkpoint_final.pth: ${CHECKPOINT_FINAL}" >&2
    exit 1
fi
if [[ ! -f "${VALIDATION_SUMMARY}" ]]; then
    echo "Fixed-split command exited without validation summary: ${VALIDATION_SUMMARY}" >&2
    exit 1
fi
VALIDATION_PREDICTIONS=$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
if [[ "${VALIDATION_PREDICTIONS}" != "${VAL_COUNT}" ]]; then
    echo "Fixed validation output is incomplete: ${VALIDATION_PREDICTIONS}/${VAL_COUNT}" >&2
    exit 1
fi
echo "Fixed-split training complete: final checkpoint and ${VALIDATION_PREDICTIONS} validation predictions verified."
