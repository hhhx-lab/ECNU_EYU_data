#!/bin/bash
#SBATCH -J g1v2_train
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 120:00:00
#SBATCH -o logs/g1v2_train_%j.out
#SBATCH -e logs/g1v2_train_%j.err

set -euo pipefail

if [ -n "${G1_CODE_DIR:-}" ]; then
  CODE_DIR="$(cd "${G1_CODE_DIR}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  CODE_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
test -f "${CODE_DIR}/README.md" || { echo "ERROR: submit from the G1 V2 code root or set G1_CODE_DIR"; exit 1; }
G1_V2_CONDA_ENV="${G1_V2_CONDA_ENV:-segmamba}"
TRAIN_TARGET="${TRAIN_TARGET:-both}"
RUN_ID="${G1_MODEL_RUN_ID:-run_${SLURM_JOB_ID}}"
export G1_BBDM_S="${G1_BBDM_S:-0.01}"
export G1_WEIGHT_DECAY="${G1_WEIGHT_DECAY:-0.0}"
export G1_ENCDEC_OUTPUT_PATH="${G1_ENCDEC_OUTPUT_PATH:-training/endec/${RUN_ID}}"
export G1_BBDM_OUTPUT_PATH="${G1_BBDM_OUTPUT_PATH:-training/bbdm/${RUN_ID}}"

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V2_CONDA_ENV}"
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
mkdir -p logs
test -f data/data_csv.csv || { echo "ERROR: data/data_csv.csv not found"; exit 1; }
test -f data/selected_vae.json || { echo "ERROR: run 03_encode_latents.slurm first"; exit 1; }
G1_VAE_WEIGHTS="$(python -c 'import json; print(json.load(open("data/selected_vae.json"))["selected_weights"])')"
export G1_VAE_WEIGHTS
test -f "${G1_VAE_WEIGHTS}" || { echo "ERROR: selected VAE not found"; exit 1; }

case "${TRAIN_TARGET}" in
  both|endec|bbdm) ;;
  *) echo "ERROR: TRAIN_TARGET must be both, endec, or bbdm"; exit 1 ;;
esac
if [ "${TRAIN_TARGET}" = "both" ] && [ -n "${G1_RESUME_CHECKPOINT:-}" ]; then
  echo "ERROR: resume one model at a time with TRAIN_TARGET=endec or bbdm"
  exit 1
fi

echo "===== G1 V2 Train Models ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Train target: ${TRAIN_TARGET}"
echo "VAE: ${G1_VAE_WEIGHTS}"
echo "EncDec output: ${G1_ENCDEC_OUTPUT_PATH}"
echo "BBDM output: ${G1_BBDM_OUTPUT_PATH}"
echo "BBDM s: ${G1_BBDM_S}"
echo "Weight decay: ${G1_WEIGHT_DECAY}"
echo "Resume checkpoint: ${G1_RESUME_CHECKPOINT:-NONE}"
nvidia-smi

if [ "${TRAIN_TARGET}" = "both" ] || [ "${TRAIN_TARGET}" = "endec" ]; then
  python -u training_endec.py
  ENDEC_RUN_ABS="$(python -c 'import os; print(os.path.abspath(os.environ["G1_ENCDEC_OUTPUT_PATH"]))')"
  find "${ENDEC_RUN_ABS}/check_points" -maxdepth 1 -name 'model_*.pt' -print -quit | grep -q . || {
    echo "ERROR: EncDec produced no checkpoint"; exit 1;
  }
  mkdir -p training/endec
  printf '%s\n' "${ENDEC_RUN_ABS}" > training/endec/latest_run.txt
fi
if [ "${TRAIN_TARGET}" = "both" ] || [ "${TRAIN_TARGET}" = "bbdm" ]; then
  python -u training_bbdm.py
  BBDM_RUN_ABS="$(python -c 'import os; print(os.path.abspath(os.environ["G1_BBDM_OUTPUT_PATH"]))')"
  find "${BBDM_RUN_ABS}/check_points" -maxdepth 1 -name 'model_*.pt' -print -quit | grep -q . || {
    echo "ERROR: BBDM produced no checkpoint"; exit 1;
  }
  mkdir -p training/bbdm
  printf '%s\n' "${BBDM_RUN_ABS}" > training/bbdm/latest_run.txt
fi

echo "===== Training finished ====="
echo "Date: $(date)"
