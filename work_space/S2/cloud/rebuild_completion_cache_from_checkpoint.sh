#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/brats2026/ECNU_EYU_data}"
S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
S2_ROOT="${S2_ROOT:-/cloud/cloud-ssd1/brats2026/s2}"
S2_ENV="${S2_ENV:-/cloud/cloud-ssd1/brats2026/envs/s2_nnunet}"
DATASET_NAME="Dataset264_BraTS2026_MET_Completion"
DATASET_PREPROCESSED="${S2_ROOT}/nnUNet_preprocessed/${DATASET_NAME}"
CONFIG_DIR="${DATASET_PREPROCESSED}/nnUNetPlans_3d_fullres"
INCOMPATIBLE_BACKUP="${DATASET_PREPROCESSED}/nnUNetPlans_3d_fullres_incompatible_dataset260"
BASELINE_CHECKPOINT="${S2_ROOT}/baseline/checkpoint_final.pth"
SPLIT_DIR="${S2_ROOT}/splits/completion_warmstart"
READY_MARKER="${S2_ROOT}/WARMSTART_PREPROCESS_COMPLETE.ok"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-6}"

[[ -s "${S2_ROOT}/TRANSFER_COMPLETE.ok" ]] || {
    echo "Dataset264 transfer is incomplete" >&2
    exit 1
}
[[ -x "${S2_ENV}/bin/python" && -x "${S2_ENV}/bin/nnUNetv2_preprocess" ]] || {
    echo "S2 environment is incomplete: ${S2_ENV}" >&2
    exit 1
}
[[ -s "${BASELINE_CHECKPOINT}" ]] || {
    echo "Missing Dataset263 checkpoint: ${BASELINE_CHECKPOINT}" >&2
    exit 1
}
[[ "${PREPROCESS_WORKERS}" =~ ^[1-8]$ ]] || {
    echo "PREPROCESS_WORKERS must be from 1 to 8, got: ${PREPROCESS_WORKERS}" >&2
    exit 2
}

export PATH="${S2_ENV}/bin:${PATH}"
export nnUNet_raw="${S2_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${S2_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${S2_ROOT}/nnUNet_results"

if [[ ! -d "${INCOMPATIBLE_BACKUP}" ]]; then
    [[ -d "${CONFIG_DIR}" ]] || {
        echo "Missing original Dataset264 cache: ${CONFIG_DIR}" >&2
        exit 1
    }
    cp -p "${DATASET_PREPROCESSED}/nnUNetPlans.json" \
        "${DATASET_PREPROCESSED}/nnUNetPlans.incompatible_dataset260.json"
    if [[ -s "${DATASET_PREPROCESSED}/completion_plans_audit.json" ]]; then
        cp -p "${DATASET_PREPROCESSED}/completion_plans_audit.json" \
            "${DATASET_PREPROCESSED}/completion_plans_audit.incompatible_dataset260.json"
    fi
    mv "${CONFIG_DIR}" "${INCOMPATIBLE_BACKUP}"
    echo "S2_INCOMPATIBLE_CACHE_PRESERVED path=${INCOMPATIBLE_BACKUP}"
fi

python "${S2_REPO}/scripts/09_clone_baseline_plans.py" \
    --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
    --source-dataset-json "${nnUNet_raw}/${DATASET_NAME}/dataset.json" \
    --target-preprocessed-dir "${DATASET_PREPROCESSED}" \
    --target-dataset-name "${DATASET_NAME}" \
    --expected-num-training 1138

python - <<'PY'
import copy
import json
import os
from pathlib import Path

import torch

root = Path(os.environ["nnUNet_preprocessed"]) / "Dataset264_BraTS2026_MET_Completion"
checkpoint = torch.load(
    Path(os.environ["nnUNet_results"]).parent / "baseline" / "checkpoint_final.pth",
    map_location="cpu",
    weights_only=False,
)
expected = copy.deepcopy(checkpoint["init_args"]["plans"])
expected["dataset_name"] = "Dataset264_BraTS2026_MET_Completion"
actual = json.loads((root / "nnUNetPlans.json").read_text(encoding="utf-8"))
if actual != expected:
    raise SystemExit("Dataset264 plans do not exactly match the warm-start checkpoint")
config = actual["configurations"]["3d_fullres"]
print(
    "S2_WARMSTART_PLAN_PASS",
    f"spacing={config['spacing']}",
    f"patch_size={config['patch_size']}",
)
PY

nnUNetv2_preprocess \
    -d 264 \
    -plans_name nnUNetPlans \
    -c 3d_fullres \
    -np "${PREPROCESS_WORKERS}" \
    --no_pbar

python "${S2_REPO}/scripts/05_validate_fixed_split_cache.py" \
    --train-file "${SPLIT_DIR}/train_fixed.txt" \
    --val-file "${SPLIT_DIR}/val_fixed.txt" \
    --dataset-dir "${nnUNet_raw}/${DATASET_NAME}" \
    --preprocessed-dir "${CONFIG_DIR}" \
    --output-json "${SPLIT_DIR}/fixed_split_cache_warmstart_audit.json"

printf '%s\n' 'S2_WARMSTART_PREPROCESS_PASS' > "${READY_MARKER}"
echo "S2_WARMSTART_PREPROCESS_PASS cache=${CONFIG_DIR}"
