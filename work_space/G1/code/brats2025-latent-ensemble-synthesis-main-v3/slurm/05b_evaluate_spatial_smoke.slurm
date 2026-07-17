#!/bin/bash
#SBATCH -J g1v3_spatial14
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/g1v3_spatial14_%j.out
#SBATCH -e logs/g1v3_spatial14_%j.err

set -euo pipefail

CODE_DIR="${G1_V3_CODE_DIR:-${SLURM_SUBMIT_DIR:-}}"
test -n "${CODE_DIR}" || { echo "ERROR: set G1_V3_CODE_DIR or submit from the V3 code root"; exit 1; }
CODE_DIR="$(cd "${CODE_DIR}" && pwd)"
test -f "${CODE_DIR}/evaluate.py" || { echo "ERROR: invalid V3 code root: ${CODE_DIR}"; exit 1; }

export G1_V3_CODE_DIR="${CODE_DIR}"
export EVAL_SPLIT=val
export EVAL_EXPECTED_CASES=14
export EVAL_CASE_LIST="${CODE_DIR}/slurm/spatial_regression_cases.txt"
export EVAL_RUN_ID="spatial_smoke_${SLURM_JOB_ID}"
export EVAL_RUN_DIR="data/evaluation/val/${EVAL_RUN_ID}"
export G1_LOCKED_EVALUATION_MANIFEST="${G1_LOCKED_EVALUATION_MANIFEST:-${CODE_DIR}/data/evaluation/val/run_3056246/evaluation_run.json}"
export UPDATE_LATEST_RUN=0
export PUBLISH_EVAL_RESULT=0

exec bash "${CODE_DIR}/slurm/05_evaluate_val.slurm"
