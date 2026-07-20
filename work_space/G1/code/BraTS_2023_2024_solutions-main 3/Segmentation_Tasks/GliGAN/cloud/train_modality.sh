#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

MODALITY="${1:-${MODALITY:-}}"
case "${MODALITY}" in
  t1c|t1n|t2w|t2f) ;;
  *) echo "Usage: $0 {t1c|t1n|t2w|t2f}" >&2; exit 2 ;;
esac

PROJ="${PROJ:-/root/brats2026/ECNU_EYU_data}"
RUN_ROOT="${RUN_ROOT:-/root/brats2026/runs/g1_diffusion_v3}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/root/brats2026/envs/g1_diffusion_v3}"
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
CSV_PATH="${RUN_ROOT}/splits/current/lesions.csv"
LOGDIR="${LOGDIR:-brats2026_diffusion_v3_edm_zscore}"
NUM_STEPS="${NUM_STEPS:-150000}"
N_STEPS="${N_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOADER_WORKERS="${LOADER_WORKERS:-0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-6}"
USE_COMPILE="${USE_COMPILE:-1}"
OPEN_FILES_LIMIT="${OPEN_FILES_LIMIT:-65536}"
WEIGHTS_DIR="${CHECKPOINT_ROOT}/${LOGDIR}/${MODALITY}/weights"

if ! [[ "${OPEN_FILES_LIMIT}" =~ ^[0-9]+$ ]] || (( OPEN_FILES_LIMIT < 1024 )); then
  echo "OPEN_FILES_LIMIT must be an integer >= 1024" >&2
  exit 2
fi
ulimit -n "${OPEN_FILES_LIMIT}"

test -x "${PYTHON_BIN}"
test -s "${RUN_ROOT}/PREPARED.ok"
test -s "${CSV_PATH}"
mkdir -p "${WEIGHTS_DIR}"
cd "${CODE_DIR}"

LATEST_STEP=""
LATEST_STEP=$(find "${WEIGHTS_DIR}" -maxdepth 1 -type f -name 'diffusion_*.pt' \
  | sed -nE 's|.*/diffusion_([0-9]+)\.pt$|\1|p' | sort -n | tail -1)

TRAIN_CMD=(
  "${PYTHON_BIN}" src/train/tumour_main_diffusion.py
  --dataset BRATS_2024
  --modality "${MODALITY}"
  --logdir "${LOGDIR}"
  --checkpoint_root "${CHECKPOINT_ROOT}"
  --csv_path "${CSV_PATH}"
  --split train
  --batch_size "${BATCH_SIZE}"
  --loader_workers "${LOADER_WORKERS}"
  --generator_type Unet_NnU
  --network_channels 48,96,192,384
  --network_strides 2,2,2
  --crop_size 64
  --small_lesion_weight 3.0
  --small_lesion_threshold 27.0
  --small_lesion_clamp 1.0
  --patient_balance_mode sqrt
  --num_steps "${NUM_STEPS}"
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
  --keep_last_checkpoints "${KEEP_LAST_CHECKPOINTS}"
  --n_steps "${N_STEPS}"
  --noise_schedule edm
  --normalization zscore
  --sigma_data 1.0
)
if [[ "${USE_COMPILE}" == "1" ]]; then
  TRAIN_CMD+=(--use_compile)
fi
if [[ -n "${LATEST_STEP}" ]]; then
  if (( LATEST_STEP >= NUM_STEPS )); then
    echo "TRAIN_ALREADY_COMPLETE modality=${MODALITY} step=${LATEST_STEP}"
    exit 0
  fi
  TRAIN_CMD+=(--resume_iter "${LATEST_STEP}")
  echo "AUTO_RESUME modality=${MODALITY} step=${LATEST_STEP}"
fi

echo "TRAIN_MODALITY_START modality=${MODALITY} target=${NUM_STEPS} gpu=${CUDA_VISIBLE_DEVICES:-0}"
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
exec "${TRAIN_CMD[@]}"
