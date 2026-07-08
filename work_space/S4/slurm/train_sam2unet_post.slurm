#!/bin/bash
#SBATCH -J sam2unet_post
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

# ============================================================
# SAM2-UNet post-training Slurm job
#
# Edit the #SBATCH lines above for your cluster:
#   - NYU/other clusters may use a partition such as gpu, a100, v100, or a lab-specific name.
#   - Some clusters require an account line, for example:
#       #SBATCH --account=<your_account>
#   - Keep --gres=gpu:1 if your cluster uses generic Slurm GPU resources.
#
# Common overrides at submit time:
#   sbatch --export=ALL,RUN_MODE=debug,PROJECT_DIR=/path/to/2025sam,VENV_DIR=/path/to/venv slurm/train_sam2unet_post.slurm
#   sbatch --export=ALL,RUN_MODE=full,PROJECT_DIR=/path/to/2025sam,CONDA_ENV=myenv slurm/train_sam2unet_post.slurm
# ============================================================

# -------- User-editable paths --------
# By default, use the directory from which sbatch was submitted.
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export TRAIN_DIR="${TRAIN_DIR:-${PROJECT_DIR}/TrainingData}"
export SAVE_ROOT="${SAVE_ROOT:-${PROJECT_DIR}/output}"
export RUN_MODE="${RUN_MODE:-full}"  # full or debug

# Use exactly one of these environment choices:
#   VENV_DIR=/path/to/venv       for python -m venv / virtualenv
#   CONDA_ENV=myenv              for conda/mamba environments
#   PYTHON_BIN=/path/to/python   if you want to skip activation and call Python directly
export VENV_DIR="${VENV_DIR:-}"
export CONDA_ENV="${CONDA_ENV:-}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# Optional module names. Leave empty when your cluster does not use modules
# or when the environment is already available.
export PYTHON_MODULE="${PYTHON_MODULE:-}"
export CUDA_MODULE="${CUDA_MODULE:-}"

if [[ "${RUN_MODE}" == "debug" ]]; then
  export SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/debug_post}"
else
  export SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/full_train_450}"
fi

# -------- Cluster environment --------
if command -v module >/dev/null 2>&1; then
  module purge || true
  if [[ -n "${PYTHON_MODULE}" ]]; then
    module load "${PYTHON_MODULE}"
  fi
  if [[ -n "${CUDA_MODULE}" ]]; then
    module load "${CUDA_MODULE}"
  fi
fi

if [[ -n "${VENV_DIR}" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Missing virtual environment directory: ${VENV_DIR}" >&2
    exit 1
  fi
  set +u
  source "${VENV_DIR}/bin/activate"
  set -u
  export PYTHON_BIN="${PYTHON_BIN:-python}"
elif [[ -n "${CONDA_ENV}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    set -u
  elif command -v mamba >/dev/null 2>&1; then
    set +u
    source "$(mamba info --base)/etc/profile.d/conda.sh"
    mamba activate "${CONDA_ENV}"
    set -u
  else
    echo "CONDA_ENV=${CONDA_ENV} was set, but conda/mamba was not found." >&2
    echo "Set PYTHON_MODULE, VENV_DIR, or PYTHON_BIN for this cluster." >&2
    exit 1
  fi
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi

# -------- Project checks --------
cd "${PROJECT_DIR}"
mkdir -p "${SAVE_DIR}/logs"

if [[ ! -f train_sam2unet_post.py || ! -f sam2unet_model.py ]]; then
  echo "Missing train_sam2unet_post.py or sam2unet_model.py in ${PROJECT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "Missing TrainingData directory: ${TRAIN_DIR}" >&2
  exit 1
fi

# -------- Runtime configuration used by train_sam2unet_post.py --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export SAM2UNET_TRAIN_DIR="${TRAIN_DIR}"
export SAM2UNET_SAVE_DIR="${SAVE_DIR}"
export SAM2UNET_BATCH_SIZE="${SAM2UNET_BATCH_SIZE:-1}"
export SAM2UNET_ACCUMULATION_STEPS="${SAM2UNET_ACCUMULATION_STEPS:-1}"
export SAM2UNET_NUM_WORKERS="${SAM2UNET_NUM_WORKERS:-4}"
export SAM2UNET_PREFETCH_FACTOR="${SAM2UNET_PREFETCH_FACTOR:-2}"
export SAM2UNET_CROP_SIZE="${SAM2UNET_CROP_SIZE:-96,96,96}"
export SAM2UNET_SW_BATCH_SIZE="${SAM2UNET_SW_BATCH_SIZE:-1}"

if [[ "${RUN_MODE}" == "debug" ]]; then
  export SAM2UNET_DEBUG_CASE_LIMIT="${SAM2UNET_DEBUG_CASE_LIMIT:-20}"
  export SAM2UNET_EPOCHS="${SAM2UNET_EPOCHS:-2}"
  export SAM2UNET_WARMUP_EPOCHS="${SAM2UNET_WARMUP_EPOCHS:-1}"
  export SAM2UNET_CHECKPOINT_INTERVAL="${SAM2UNET_CHECKPOINT_INTERVAL:-1}"
else
  unset SAM2UNET_DEBUG_CASE_LIMIT
  export SAM2UNET_EPOCHS="${SAM2UNET_EPOCHS:-450}"
  export SAM2UNET_WARMUP_EPOCHS="${SAM2UNET_WARMUP_EPOCHS:-30}"
  export SAM2UNET_CHECKPOINT_INTERVAL="${SAM2UNET_CHECKPOINT_INTERVAL:-10}"
fi

echo "===== SLURM job info ====="
echo "job id: ${SLURM_JOB_ID:-NA}"
echo "node: ${SLURMD_NODENAME:-$(hostname)}"
echo "submit dir: ${SLURM_SUBMIT_DIR:-NA}"
echo "project: ${PROJECT_DIR}"
echo "train dir: ${SAM2UNET_TRAIN_DIR}"
echo "save dir: ${SAM2UNET_SAVE_DIR}"
echo "run mode: ${RUN_MODE}"
echo "python: ${PYTHON_BIN}"
echo "=========================="

nvidia-smi || true

"${PYTHON_BIN}" - <<'PY'
import torch, nibabel, einops, matplotlib
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

"${PYTHON_BIN}" -u train_sam2unet_post.py 2>&1 | tee "${SAVE_DIR}/logs/train-${SLURM_JOB_ID:-local}.log"
