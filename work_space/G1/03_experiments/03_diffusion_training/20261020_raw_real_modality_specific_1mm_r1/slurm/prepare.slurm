#!/bin/bash
#SBATCH --job-name=g1raw1prepare
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_raw_real_modality_specific_1mm_20261020_r1/logs/g1raw1prepare_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_raw_real_modality_specific_1mm_20261020_r1/logs/g1raw1prepare_%j.err
set -euo pipefail
if [[ "${G1_STATIC_DRY_RUN:-0}" == "1" ]]; then
  printf "G1_RAW_R1_BASH_U_DRY_RUN_PASS\n"
  exit 0
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=20261013
unset PYTHONPATH
cd /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_raw_real_modality_specific_1mm_20261020_r1
export CUDA_VISIBLE_DEVICES=''
exec /public/home/zqchen/.conda/envs/segmamba/bin/python -B -u runtime/prepare.py
