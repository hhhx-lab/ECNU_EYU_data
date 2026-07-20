#!/usr/bin/env bash
#SBATCH --job-name=g1_diffv3_4modal
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=40:30:00
#SBATCH --signal=B:TERM@1800
#SBATCH --array=0-3%4
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_%A_%a.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_%A_%a.err

set -euo pipefail

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-g1_diffusion_v3}"
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
MODALITIES=(t1c t1n t2w t2f)
MODALITY="${MODALITIES[${SLURM_ARRAY_TASK_ID}]}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export PROJ
export RUN_ROOT="${RUN_ROOT:-${PROJ}/runs/g1_diffusion_v3}"
export PYTHON_BIN="$(command -v python)"
export NUM_STEPS="${NUM_STEPS:-150000}"
export N_STEPS=1000
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-6}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LOADER_WORKERS="${LOADER_WORKERS:-0}"
export USE_COMPILE="${USE_COMPILE:-1}"

test -s "${RUN_ROOT}/PREPARED.ok"
test -s "${RUN_ROOT}/splits/current/lesions.csv"
cd "${CODE_DIR}"
echo "SLURM_ARRAY_MODALITY task=${SLURM_ARRAY_TASK_ID} modality=${MODALITY} gpu=${CUDA_VISIBLE_DEVICES:-allocated}"
exec cloud/train_modality.sh "${MODALITY}"
