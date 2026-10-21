#!/bin/bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_k1_selected103_official_met_20261008_r1/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_k1_selected103_official_met_20261008_r1/logs/%x_%j.err

set -euo pipefail
umask 077
ROUTE="${1:?route required}"
case "$ROUTE" in K1) ;; *) exit 2 ;; esac
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_k1_selected103_official_met_20261008_r1
module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
export PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU
export PYTHONPATH="$ROOT/runtime"
export nnUNet_compile=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_raw=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency/nnUNet_raw
export nnUNet_preprocessed=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency/nnUNet_preprocessed
export nnUNet_results="$ROOT/model_snapshots"
export nnUNet_extTrainer="$ROOT/runtime/custom_nnunet"
cd "$ROOT"
/public/home/zqchen/.conda/envs/segmamba/bin/python -B "$ROOT/runtime/workflow.py" route --route "$ROUTE"
