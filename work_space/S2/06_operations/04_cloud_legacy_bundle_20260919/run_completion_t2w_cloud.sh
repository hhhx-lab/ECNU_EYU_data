#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/brats2026/ECNU_EYU_data}"
S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
S2_ROOT="${S2_ROOT:-/cloud/cloud-ssd1/brats2026/s2}"
S2_ENV="${S2_ENV:-/cloud/cloud-ssd1/brats2026/envs/s2_nnunet}"
DATASET_NAME="Dataset264_BraTS2026_MET_Completion"
DIFFUSION_RUN_ROOT="${DIFFUSION_RUN_ROOT:-/root/brats2026/runs/g1_diffusion_v3}"

[[ -s "${S2_ROOT}/TRANSFER_COMPLETE.ok" ]] || {
    echo "Dataset264 cloud transfer is incomplete: ${S2_ROOT}/TRANSFER_COMPLETE.ok" >&2
    exit 1
}
[[ -s "${S2_ROOT}/WARMSTART_PREPROCESS_COMPLETE.ok" ]] || {
    echo "Checkpoint-compatible Dataset264 preprocessing is incomplete" >&2
    exit 1
}
PREFIX_COUNT=$(
    awk '{print substr($0, 1, 13)}' \
        "${S2_ROOT}/splits/completion_warmstart/train_fixed.txt" \
        "${S2_ROOT}/splits/completion_warmstart/val_fixed.txt" \
        | sort -u | wc -l | tr -d '[:space:]'
)
EXPECTED_MARKERS=$((PREFIX_COUNT * 2))
ACTUAL_MARKERS=$(
    find "${S2_ROOT}/transfer_markers" -maxdepth 1 -type f \
        \( -name 'preprocessed_BraTSMET_????.ok' -o -name 'raw_BraTSMET_????.ok' \) \
        | wc -l | tr -d '[:space:]'
)
if [[ "${ACTUAL_MARKERS}" != "${EXPECTED_MARKERS}" ]]; then
    echo "Dataset264 transfer marker mismatch: expected=${EXPECTED_MARKERS} actual=${ACTUAL_MARKERS}" >&2
    exit 1
fi
[[ -x "${S2_ENV}/bin/python" && -x "${S2_ENV}/bin/nnUNetv2_train" ]] || {
    echo "S2 environment is incomplete: ${S2_ENV}" >&2
    exit 1
}
[[ -s "${S2_ROOT}/baseline/checkpoint_final.pth" ]] || {
    echo "Missing Dataset263 warm-start checkpoint" >&2
    exit 1
}
[[ -s "${DIFFUSION_RUN_ROOT}/pids/t2w.pid" ]] || {
    echo "Missing t2w Diffusion PID file" >&2
    exit 1
}
DIFFUSION_PID=$(cat "${DIFFUSION_RUN_ROOT}/pids/t2w.pid")
kill -0 "${DIFFUSION_PID}" 2>/dev/null || {
    echo "t2w Diffusion is not running: pid=${DIFFUSION_PID}" >&2
    exit 1
}

export PATH="${S2_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export nnUNet_raw="${S2_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${S2_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${S2_ROOT}/nnUNet_results"
export BRATS_SPLIT_DIR="${S2_ROOT}/splits/completion_warmstart"
export NNUNET_DATASET_DIR="${nnUNet_raw}/${DATASET_NAME}"
export S2_EXPERIMENT_MODE=completion_warmstart
export S2_PRETRAINED_WEIGHTS="${S2_ROOT}/baseline/checkpoint_final.pth"
export S2_COMPLETION_EPOCHS="${S2_COMPLETION_EPOCHS:-200}"
export S2_COMPLETION_INITIAL_LR="${S2_COMPLETION_INITIAL_LR:-0.001}"
export S2_COMPLETION_SAVE_EVERY="${S2_COMPLETION_SAVE_EVERY:-25}"
export S2_CONTINUE="${S2_CONTINUE:-auto}"
export S2_SKIP_COMPLETED="${S2_SKIP_COMPLETED:-1}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python - <<'PY'
import copy
import importlib.metadata
import json
import os
from pathlib import Path

import torch

dataset_name = "Dataset264_BraTS2026_MET_Completion"
checkpoint = torch.load(
    Path(os.environ["S2_PRETRAINED_WEIGHTS"]),
    map_location="cpu",
    weights_only=False,
)
expected = copy.deepcopy(checkpoint["init_args"]["plans"])
expected["dataset_name"] = dataset_name
plans_path = Path(os.environ["nnUNet_preprocessed"]) / dataset_name / "nnUNetPlans.json"
actual = json.loads(plans_path.read_text(encoding="utf-8"))
if actual != expected:
    raise SystemExit("Dataset264 plans differ from the Dataset263 warm-start checkpoint")
print("warmstart plans: exact match")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("nnunetv2:", importlib.metadata.version("nnunetv2"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
props = torch.cuda.get_device_properties(0)
print("gpu:", props.name, f"{props.total_memory / 1024**3:.1f} GiB")
if props.total_memory < 70 * 1024**3:
    raise SystemExit("Concurrent S2/t2w run requires an 80GB-class GPU")
PY

echo "S2_CLOUD_TRAIN_START diffusion_pid=${DIFFUSION_PID} data_root=${S2_ROOT}"
cd "${S2_REPO}"
bash train.sh

RESULT_FOLD="${nnUNet_results}/${DATASET_NAME}/nnUNetTrainerBraTS2026RCCompletionFineTune__nnUNetPlans__3d_fullres/fold_0"
[[ -s "${RESULT_FOLD}/checkpoint_final.pth" ]] || {
    echo "Missing S2 final checkpoint: ${RESULT_FOLD}" >&2
    exit 1
}
[[ -s "${RESULT_FOLD}/validation/summary.json" ]] || {
    echo "Missing S2 validation summary: ${RESULT_FOLD}" >&2
    exit 1
}
echo "S2_CLOUD_TRAIN_PASS result=${RESULT_FOLD}"
