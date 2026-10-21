#!/bin/bash
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=s2k1p0supp4
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_k1_small_lesion_p0_20261015_r4_full_wt/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_k1_small_lesion_p0_20261015_r4_full_wt/logs/%x_%j.err
set -euo pipefail
umask 077
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_k1_small_lesion_p0_20261015_r4_full_wt
export PYTHONDONTWRITEBYTECODE=1 MKL_THREADING_LAYER=GNU nnUNet_compile=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
cd "$ROOT"
exec /public/home/zqchen/.conda/envs/segmamba/bin/python -B "$ROOT/runtime/supplement.py" run
