#!/bin/bash
#SBATCH --job-name=s4_sam2unet
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/s4_sam2unet_%j.out
#SBATCH --error=logs/s4_sam2unet_%j.err

set -euo pipefail

# Submit from work_space/S4 after running the G2 fixed-split preparation job.
# RUN_MODE defaults to debug deliberately. A full run requires an explicit
# SAM2UNET_EPOCHS value after the debug benchmark has been reviewed.
# Slurm may stage the script under /opt/gridview/slurm on compute nodes, so
# derive the project from the submit directory instead of BASH_SOURCE.
S4_ROOT="${S4_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${S4_ROOT}" ]]; then
  echo "S4_ROOT is required when SLURM_SUBMIT_DIR is unavailable." >&2
  exit 2
fi
PROJ="${PROJ:-$(cd "${S4_ROOT}/../.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-${S4_ROOT}/code}"
FIXED_SPLIT_ROOT="${FIXED_SPLIT_ROOT:-${PROJ}/work_space/G2/results/materialized_s4_realonly/BraTS2026_MET_RealOnly_S4_case_folders}"
SAVE_ROOT="${SAVE_ROOT:-${S4_ROOT}/results}"
RUN_MODE="${RUN_MODE:-debug}"

CONDA_SH="${CONDA_SH:-/public/software/apps/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats_swin}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${RUN_MODE}" != "debug" && "${RUN_MODE}" != "full" ]]; then
  echo "RUN_MODE must be debug or full, got: ${RUN_MODE}" >&2
  exit 2
fi
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing conda activation script: ${CONDA_SH}" >&2
  exit 1
fi
if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Missing S4 code directory: ${PROJECT_DIR}" >&2
  exit 1
fi
for split_name in train val test; do
  if [[ ! -d "${FIXED_SPLIT_ROOT}/${split_name}" ]]; then
    echo "Missing G2 fixed split directory: ${FIXED_SPLIT_ROOT}/${split_name}" >&2
    echo "Run slurm/00_prepare_g2_realonly_fixed_split.slurm first." >&2
    exit 1
  fi
done

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

if [[ "${RUN_MODE}" == "debug" ]]; then
  SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/debug_post}"
  export SAM2UNET_DEBUG_CASE_LIMIT="${SAM2UNET_DEBUG_CASE_LIMIT:-20}"
  export SAM2UNET_EPOCHS="${SAM2UNET_EPOCHS:-2}"
  export SAM2UNET_WARMUP_EPOCHS="${SAM2UNET_WARMUP_EPOCHS:-1}"
  export SAM2UNET_CHECKPOINT_INTERVAL="${SAM2UNET_CHECKPOINT_INTERVAL:-1}"
else
  if [[ -z "${SAM2UNET_EPOCHS:-}" ]]; then
    echo "RUN_MODE=full requires an explicit SAM2UNET_EPOCHS after the debug timing review." >&2
    exit 2
  fi
  SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/full_train_${SAM2UNET_EPOCHS}ep}"
  unset SAM2UNET_DEBUG_CASE_LIMIT
  export SAM2UNET_WARMUP_EPOCHS="${SAM2UNET_WARMUP_EPOCHS:-30}"
  export SAM2UNET_CHECKPOINT_INTERVAL="${SAM2UNET_CHECKPOINT_INTERVAL:-10}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export SAM2UNET_FIXED_SPLIT_ROOT="${FIXED_SPLIT_ROOT}"
export SAM2UNET_SAVE_DIR="${SAVE_DIR}"
export SAM2UNET_BATCH_SIZE="${SAM2UNET_BATCH_SIZE:-1}"
export SAM2UNET_ACCUMULATION_STEPS="${SAM2UNET_ACCUMULATION_STEPS:-1}"
export SAM2UNET_NUM_WORKERS="${SAM2UNET_NUM_WORKERS:-4}"
export SAM2UNET_PREFETCH_FACTOR="${SAM2UNET_PREFETCH_FACTOR:-2}"
export SAM2UNET_CROP_SIZE="${SAM2UNET_CROP_SIZE:-96,96,96}"
export SAM2UNET_SW_BATCH_SIZE="${SAM2UNET_SW_BATCH_SIZE:-1}"

mkdir -p "${SAVE_DIR}/logs"
cd "${PROJECT_DIR}"

if [[ ! -f train_sam2unet_post.py || ! -f sam2unet_model.py ]]; then
  echo "Missing S4 training code in ${PROJECT_DIR}" >&2
  exit 1
fi

echo "===== S4 SAM2-UNet job ====="
echo "Job ID           : ${SLURM_JOB_ID:-local}"
echo "Node             : ${SLURMD_NODENAME:-unknown}"
echo "Project root     : ${PROJ}"
echo "Code directory   : ${PROJECT_DIR}"
echo "G2 fixed split   : ${FIXED_SPLIT_ROOT}"
echo "Save directory   : ${SAVE_DIR}"
echo "Run mode         : ${RUN_MODE}"
echo "Epochs           : ${SAM2UNET_EPOCHS}"
echo "Conda environment: ${CONDA_ENV}"
echo "============================"

"${PYTHON_BIN}" - <<'PY'
import einops
import matplotlib
import nibabel
import numpy
import torch
import tqdm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the allocated GPU job")
print("GPU:", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" -u train_sam2unet_post.py 2>&1 | tee "${SAVE_DIR}/logs/train-${SLURM_JOB_ID:-local}.log"

echo "===== DONE: $(date) ====="
echo "Best combined model: ${SAVE_DIR}/best_combined_model.pth"
