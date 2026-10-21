#!/usr/bin/env bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/logs/%x_%j.err
set -euo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1
PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
export PYTHONPATH="${ROOT}/runtime:${PROJECT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository:${PYTHONPATH:-}"
export nnUNet_raw="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_preprocessed"
export nnUNet_results="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_results"
export nnUNet_extTrainer="${ROOT}/runtime/custom_nnunet"
export nnUNet_compile=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
exec "$PYTHON" runtime/workflow.py route --route "${1:?route required}"
