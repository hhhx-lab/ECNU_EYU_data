#!/bin/bash
#SBATCH --job-name=g1d1m3t2w
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@300
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261014_manual_r3/logs/g1d1m3t2w_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261014_manual_r3/logs/g1d1m3t2w_%j.err
set -euo pipefail
if [[ "${G1_STATIC_DRY_RUN:-0}" == "1" ]]; then
  printf "G1_R3_BASH_U_DRY_RUN_PASS\n"
  exit 0
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=20261013
unset PYTHONPATH
cd /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261014_manual_r3
printf "START %s UTC " "${SLURM_JOB_ID}"; date -u +%FT%TZ
hostname
exec /public/home/zqchen/.conda/envs/segmamba/bin/python -B -u runtime/train.py --modality t2w
