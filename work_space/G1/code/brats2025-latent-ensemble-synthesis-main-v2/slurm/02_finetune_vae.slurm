#!/bin/bash
#SBATCH -J g1v2_vae
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 18:00:00
#SBATCH -o logs/g1v2_vae_%j.out
#SBATCH -e logs/g1v2_vae_%j.err

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
CONDA_ENV="${CONDA_ENV:-brats_g1_v2}"
VAE_EPOCHS="${VAE_EPOCHS:-3}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-2}"
VAE_VAL_INTERVAL="${VAE_VAL_INTERVAL:-1}"
VAE_SAVE_INTERVAL="${VAE_SAVE_INTERVAL:-1}"
VAE_AMP_DTYPE="${VAE_AMP_DTYPE:-bfloat16}"
VAE_PATCH_SIZE="${VAE_PATCH_SIZE:-128 128 96}"
VAE_TUMOR_PATCH_PROBABILITY="${VAE_TUMOR_PATCH_PROBABILITY:-0.8}"
VAE_QUICK_VAL_SUBJECTS="${VAE_QUICK_VAL_SUBJECTS:-20}"
VAE_EARLY_STOPPING_PATIENCE="${VAE_EARLY_STOPPING_PATIENCE:-2}"
VAE_OUTPUT_DIR="${VAE_OUTPUT_DIR:-training/vae_finetuned/run_${SLURM_JOB_ID}}"
VAE_MAX_TRAIN_SUBJECTS="${VAE_MAX_TRAIN_SUBJECTS:-}"
VAE_MAX_VAL_SUBJECTS="${VAE_MAX_VAL_SUBJECTS:-}"
PUBLISH_VAE_SELECTION="${PUBLISH_VAE_SELECTION:-1}"

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${CODE_DIR}"
mkdir -p logs "${VAE_OUTPUT_DIR}" training/vae_finetuned

echo "===== G1 V2 VAE Fine-tuning ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Code dir: ${CODE_DIR}"
echo "Output dir: ${VAE_OUTPUT_DIR}"
echo "Epochs: ${VAE_EPOCHS}"
echo "Subject accumulation batch: ${VAE_BATCH_SIZE}"
echo "AMP dtype: ${VAE_AMP_DTYPE}"
echo "Patch size: ${VAE_PATCH_SIZE}"
echo "Tumor patch probability: ${VAE_TUMOR_PATCH_PROBABILITY}"
echo "Quick val subjects: ${VAE_QUICK_VAL_SUBJECTS}"
echo "Early stopping patience: ${VAE_EARLY_STOPPING_PATIENCE}"
echo "Max train subjects: ${VAE_MAX_TRAIN_SUBJECTS:-ALL}"
echo "Max val subjects: ${VAE_MAX_VAL_SUBJECTS:-ALL}"
echo "Date: $(date)"
nvidia-smi

test -f data/data_csv.csv || { echo "ERROR: run 01_prepare_data.slurm first"; exit 1; }
python -u validate_g1_v2_dataset.py --skip-label-values
python -u validate_g1_v2_preparation.py

BASELINE_LIMIT_ARGS=(--device cuda:0)
TRAIN_LIMIT_ARGS=(--device cuda:0)
COMPARE_LIMIT_ARGS=(--device cuda:0)
read -r -a VAE_PATCH_SIZE_ARGS <<< "${VAE_PATCH_SIZE}"
if [ "${#VAE_PATCH_SIZE_ARGS[@]}" -ne 3 ]; then
  echo "ERROR: VAE_PATCH_SIZE must contain exactly three integers, for example: 128 128 96"
  exit 1
fi
if [ -n "${VAE_MAX_TRAIN_SUBJECTS}" ]; then
  TRAIN_LIMIT_ARGS+=(--max_train_subjects "${VAE_MAX_TRAIN_SUBJECTS}")
fi
if [ -n "${VAE_MAX_VAL_SUBJECTS}" ]; then
  BASELINE_LIMIT_ARGS+=(--max_subjects "${VAE_MAX_VAL_SUBJECTS}")
  TRAIN_LIMIT_ARGS+=(--max_val_subjects "${VAE_MAX_VAL_SUBJECTS}")
  COMPARE_LIMIT_ARGS+=(--max_subjects "${VAE_MAX_VAL_SUBJECTS}")
fi

echo "===== 1/3 Pretrained VAE baseline ====="
python -u validate_vae_recon.py \
  --data_csv data/data_csv.csv \
  --data_dir data/input \
  --vae_weights weights/vae/autoencoder_epoch273.pt \
  --output_dir "${VAE_OUTPUT_DIR}" \
  --mode baseline \
  --save_samples 0 \
  "${BASELINE_LIMIT_ARGS[@]}"

echo "===== 2/3 Fine-tune VAE on train split ====="
python -u finetune_vae.py \
  --data_csv data/data_csv.csv \
  --data_dir data/input \
  --vae_weights weights/vae/autoencoder_epoch273.pt \
  --output_dir "${VAE_OUTPUT_DIR}" \
  --epochs "${VAE_EPOCHS}" \
  --batch_size "${VAE_BATCH_SIZE}" \
  --val_interval "${VAE_VAL_INTERVAL}" \
  --save_interval "${VAE_SAVE_INTERVAL}" \
  --amp-dtype "${VAE_AMP_DTYPE}" \
  --patch-size "${VAE_PATCH_SIZE_ARGS[@]}" \
  --tumor-patch-probability "${VAE_TUMOR_PATCH_PROBABILITY}" \
  --quick-val-subjects "${VAE_QUICK_VAL_SUBJECTS}" \
  --early-stopping-patience "${VAE_EARLY_STOPPING_PATIENCE}" \
  --no-gradient-checkpointing \
  --skip_initial_validation \
  "${TRAIN_LIMIT_ARGS[@]}"

test -f "${VAE_OUTPUT_DIR}/best_model.pt" || { echo "ERROR: best_model.pt was not produced"; exit 1; }

echo "===== 3/3 Compare on untouched val split ====="
python -u validate_vae_recon.py \
  --data_csv data/data_csv.csv \
  --data_dir data/input \
  --vae_weights weights/vae/autoencoder_epoch273.pt \
  --finetuned_weights "${VAE_OUTPUT_DIR}/best_model.pt" \
  --output_dir "${VAE_OUTPUT_DIR}" \
  --mode compare \
  --save_samples 5 \
  "${COMPARE_LIMIT_ARGS[@]}"

if [ "${PUBLISH_VAE_SELECTION}" = "1" ]; then
  printf '%s\n' "${VAE_OUTPUT_DIR}" > training/vae_finetuned/latest_run.txt
fi
python - <<PY
import json
from pathlib import Path
selection = json.loads(Path("${VAE_OUTPUT_DIR}/vae_selection.json").read_text())
print(json.dumps(selection, indent=2))
PY

echo "===== VAE stage finished; inspect results before stage 03 ====="
echo "Result directory: ${VAE_OUTPUT_DIR}"
echo "Decision file: ${VAE_OUTPUT_DIR}/vae_selection.json"
echo "Date: $(date)"
