#!/bin/bash
#SBATCH --job-name=g1d1m8p1
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261016_resume_localfix_r8/logs/g1d1m8p1_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261016_resume_localfix_r8/logs/g1d1m8p1_%j.err
set -euo pipefail
if [[ "${G1_STATIC_DRY_RUN:-0}" == "1" ]]; then
  printf "G1_R8_BASH_U_DRY_RUN_PASS\n"
  exit 0
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=20261013
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES=''
cd /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261016_resume_localfix_r8
printf "START %s UTC " "${SLURM_JOB_ID}"; date -u +%FT%TZ
hostname
exec /public/home/zqchen/.conda/envs/segmamba/bin/python -B -u runtime/prepare.py
