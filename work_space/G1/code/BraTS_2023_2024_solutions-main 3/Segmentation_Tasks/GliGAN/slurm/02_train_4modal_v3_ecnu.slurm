#!/bin/bash
#SBATCH -J g1d3_train
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH --array=0-3%4
#SBATCH --requeue
#SBATCH -o /public/home/%u/g1_diffusion_v3_production_20260715/logs/train_%A_%a.out
#SBATCH -e /public/home/%u/g1_diffusion_v3_production_20260715/logs/train_%A_%a.err

set -Eeuo pipefail

PROD_ROOT=${PROD_ROOT:-/public/home/${USER}/g1_diffusion_v3_production_20260715}
CODE_DIR=${CODE_DIR:-"${PROD_ROOT}/code/Segmentation_Tasks/GliGAN"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROD_ROOT}/checkpoints"}
SPLIT_DIR=${SPLIT_DIR:-"${PROD_ROOT}/splits/current"}
CSV_PATH=${CSV_PATH:-"${SPLIT_DIR}/lesions.csv"}
LOGDIR=${LOGDIR:-brats2026_diffusion_v3_edm_zscore}
PYTHON_BIN=${PYTHON_BIN:-/public/home/${USER}/.conda/envs/segmamba/bin/python}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_STEPS=${NUM_STEPS:-100000}
LOADER_WORKERS=${LOADER_WORKERS:-0}
# ECNU's system GCC 4.8 lacks stdatomic.h, so Triton/torch.compile is disabled
# by default. Set USE_COMPILE=1 only after loading a verified newer compiler.
USE_COMPILE=${USE_COMPILE:-0}
AUTO_RESUME=${AUTO_RESUME:-1}

MODALITIES=(t1c t1n t2w t2f)
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( TASK_ID < 0 || TASK_ID >= ${#MODALITIES[@]} )); then
  echo "Invalid array task: ${TASK_ID}" >&2
  exit 2
fi
MODALITY=${MODALITIES[$TASK_ID]}

test -s "${PROD_ROOT}/PREPARED.ok"
test -s "${CSV_PATH}"
mkdir -p "${PROD_ROOT}/logs" "${CHECKPOINT_ROOT}"

echo "job_id=${SLURM_JOB_ID}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${TASK_ID}"
echo "modality=${MODALITY}"
echo "node=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "csv_path=${CSV_PATH}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
nvidia-smi
"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
assert any(token in name for token in ("A100", "H100", "H200")), name
print("gpu_contract=PASS", name, torch.__version__, torch.version.cuda)
PY

WEIGHTS_DIR="${CHECKPOINT_ROOT}/${LOGDIR}/${MODALITY}/weights"
RESUME_ARGS=()
if [[ -d "${WEIGHTS_DIR}" ]]; then
  LATEST_STEP=$(find "${WEIGHTS_DIR}" -maxdepth 1 -type f -name 'diffusion_*.pt' \
    -printf '%f\n' | sed -nE 's/^diffusion_([0-9]+)\.pt$/\1/p' | sort -n | tail -1)
  if [[ -n "${LATEST_STEP:-}" ]]; then
    if (( LATEST_STEP >= NUM_STEPS )); then
      echo "TRAIN_ALREADY_COMPLETE modality=${MODALITY} step=${LATEST_STEP}"
      exit 0
    fi
    if [[ "${AUTO_RESUME}" != "1" ]]; then
      echo "Existing checkpoint found but AUTO_RESUME=${AUTO_RESUME}: ${LATEST_STEP}" >&2
      exit 3
    fi
    RESUME_ARGS=(--resume_iter "${LATEST_STEP}")
    echo "auto_resume_step=${LATEST_STEP}"
  fi
fi

COMPILE_ARGS=()
if [[ "${USE_COMPILE}" == "1" ]]; then
  COMPILE_ARGS=(--use_compile)
fi

cd "${CODE_DIR}"
"${PYTHON_BIN}" src/train/tumour_main_diffusion.py \
  --dataset BRATS_2024 \
  --modality "${MODALITY}" \
  --logdir "${LOGDIR}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --csv_path "${CSV_PATH}" \
  --split train \
  --batch_size "${BATCH_SIZE}" \
  --loader_workers "${LOADER_WORKERS}" \
  --generator_type Unet_NnU \
  --network_channels 48,96,192,384 \
  --network_strides 2,2,2 \
  --crop_size 64 \
  --small_lesion_weight 3.0 \
  --small_lesion_threshold 27.0 \
  --small_lesion_clamp 1.0 \
  --patient_balance_mode sqrt \
  --num_steps "${NUM_STEPS}" \
  --noise_schedule edm \
  --normalization zscore \
  --sigma_data 1.0 \
  "${COMPILE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

test -s "${WEIGHTS_DIR}/diffusion_${NUM_STEPS}.pt"
echo "TRAIN_MODALITY_PASS modality=${MODALITY} checkpoint=${WEIGHTS_DIR}/diffusion_${NUM_STEPS}.pt"
