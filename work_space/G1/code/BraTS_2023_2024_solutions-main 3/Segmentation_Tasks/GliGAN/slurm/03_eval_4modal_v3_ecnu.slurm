#!/bin/bash
#SBATCH -J g1d3_eval
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o /public/home/%u/g1_diffusion_v3_production_20260715/logs/eval_%j.out
#SBATCH -e /public/home/%u/g1_diffusion_v3_production_20260715/logs/eval_%j.err

set -Eeuo pipefail

PROD_ROOT=${PROD_ROOT:-/public/home/${USER}/g1_diffusion_v3_production_20260715}
CODE_DIR=${CODE_DIR:-"${PROD_ROOT}/code/Segmentation_Tasks/GliGAN"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROD_ROOT}/checkpoints"}
CSV_PATH=${CSV_PATH:-"${PROD_ROOT}/splits/current/lesions.csv"}
LOGDIR=${LOGDIR:-brats2026_diffusion_v3_edm_zscore}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROD_ROOT}/eval/${LOGDIR}_whole_brain"}
PYTHON_BIN=${PYTHON_BIN:-/public/home/${USER}/.conda/envs/segmamba/bin/python}
MAX_CASES=${MAX_CASES:-0}

CKPT_DIR="${CHECKPOINT_ROOT}/${LOGDIR}"
for modality in t1c t1n t2w t2f; do
  test -s "${CKPT_DIR}/${modality}/weights/diffusion_100000.pt"
done
mkdir -p "${OUTPUT_DIR}" "${PROD_ROOT}/logs"

cd "${CODE_DIR}"
"${PYTHON_BIN}" src/infer/evaluate_generation.py \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --csv_path "${CSV_PATH}" \
  --dataset BRATS_2024 \
  --output_dir "${OUTPUT_DIR}" \
  --generator_type Unet_NnU \
  --crop_size 64 \
  --normalization zscore \
  --sigma_data 1.0 \
  --evaluation_mode whole_brain \
  --large_lesion_mode tile \
  --split val \
  --noise_schedule edm \
  --sampling_method edm_heun \
  --sampling_steps 18 \
  --max_cases "${MAX_CASES}"

test -s "${OUTPUT_DIR}/metrics.json"
echo "EVALUATION_PASS metrics=${OUTPUT_DIR}/metrics.json"
