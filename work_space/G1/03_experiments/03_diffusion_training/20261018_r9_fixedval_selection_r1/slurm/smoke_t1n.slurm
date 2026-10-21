#!/bin/bash
#SBATCH --job-name=g1r9v_smoke_t1n
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_r9_fixedval_selection_20261018_r1/logs/smoke_t1n_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_r9_fixedval_selection_20261018_r1/logs/smoke_t1n_%j.err
set -euo pipefail
if [[ "${G1_STATIC_DRY_RUN:-0}" == "1" ]]; then
  printf "G1_R9_EVAL_BASH_U_DRY_RUN_PASS\n"
  exit 0
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=20261013 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH
cd /public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_r9_fixedval_selection_20261018_r1
printf "START %s UTC " "${SLURM_JOB_ID}"; date -u +%FT%TZ
hostname
exec /public/home/zqchen/.conda/envs/segmamba/bin/python -B -u runtime/evaluate.py --modality t1n --smoke
