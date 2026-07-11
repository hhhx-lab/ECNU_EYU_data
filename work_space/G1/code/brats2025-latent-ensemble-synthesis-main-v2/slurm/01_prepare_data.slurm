#!/bin/bash
#SBATCH -J g1v2_prep
#SBATCH -p cpu_96G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/g1v2_prep_%j.out
#SBATCH -e logs/g1v2_prep_%j.err

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
PROJECT_ROOT="${G1_PROJECT_ROOT:-$(cd "${CODE_DIR}/../../../.." && pwd)}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${HOME}/data}"
CONDA_ENV="${CONDA_ENV:-brats_g1_v2}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
TEST_FRACTION="${TEST_FRACTION:-0.10}"
EXPECTED_FAKE_T2W_COUNT="${EXPECTED_FAKE_T2W_COUNT:-265}"

module purge
module load apps/envs/miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
echo "Resolved code dir: ${CODE_DIR}"
echo "Working directory: $(pwd)"
ls -ld "${CODE_DIR}" "${CODE_DIR}/logs" "${CODE_DIR}/data" 2>/dev/null || true
mkdir -p logs data/input data/input_inference data/output

echo "===== G1 V2 Prepare Metadata and Splits ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Code dir: ${CODE_DIR}"
echo "Raw data root: ${RAW_DATA_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "Date: $(date)"

test -d "${RAW_DATA_ROOT}" || { echo "ERROR: RAW_DATA_ROOT not found: ${RAW_DATA_ROOT}"; exit 1; }
test -f weights/vae/autoencoder_epoch273.pt || { echo "ERROR: pretrained VAE is missing"; exit 1; }

python -u test_vae.py
python -u prepare_g1_v2_data.py \
  --raw-root "${RAW_DATA_ROOT}" \
  --data-dir data \
  --expected-fake-t2w-count "${EXPECTED_FAKE_T2W_COUNT}" \
  --mode symlink \
  --clean \
  --overwrite

python -u preprocess.py --metadata-only
python -u filter_g1_v2_invalid_labels.py \
  --data-csv data/data_csv.csv \
  --data-dir data/input
python -u mark_g1_train_val_test_split.py \
  --data-csv data/data_csv.csv \
  --seed "${SPLIT_SEED}" \
  --val-fraction "${VAL_FRACTION}" \
  --test-fraction "${TEST_FRACTION}"

python -u validate_g1_v2_dataset.py \
  --data-csv data/data_csv.csv \
  --data-dir data/input \
  --skip-label-values
python -u validate_g1_v2_preparation.py \
  --data-dir data \
  --expected-fake "${EXPECTED_FAKE_T2W_COUNT}" \
  --expected-total 1030 \
  --expected-train 823 \
  --expected-val 103 \
  --expected-test 104

python -u finetune_vae.py \
  --data_csv data/data_csv.csv \
  --data_dir data/input \
  --dry_run

echo "===== Preparation passed ====="
echo "CSV: data/data_csv.csv"
echo "Split summary: data/g1_split_summary.json"
echo "Label filter: data/g1_v2_label_filter_summary.json"
echo "Dataset validation: data/g1_v2_dataset_validation.json"
echo "Placement/split validation: data/g1_v2_preparation_validation.json"
echo "Date: $(date)"
