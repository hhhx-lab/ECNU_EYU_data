#!/bin/bash
#SBATCH -J g1v3_adopt
#SBATCH -p cpu_96G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o logs/g1v3_adopt_%j.out
#SBATCH -e logs/g1v3_adopt_%j.err

set -euo pipefail

if [ -n "${G1_V3_CODE_DIR:-}" ]; then
  CODE_DIR="$(cd "${G1_V3_CODE_DIR}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  CODE_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
test -f "${CODE_DIR}/README.md" || { echo "ERROR: submit from the G1 V3 code root or set G1_V3_CODE_DIR"; exit 1; }

G1_V3_CONDA_ENV="${G1_V3_CONDA_ENV:-segmamba}"
DEFAULT_SOURCE_DATA_DIR="$(dirname "${CODE_DIR}")/brats2025-latent-ensemble-synthesis-main-v2/data"
SOURCE_DATA_DIR="${G1_V3_SOURCE_DATA_DIR:-${DEFAULT_SOURCE_DATA_DIR}}"

module purge
module load apps/envs/miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V3_CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
test -x "${PYTHON_BIN}" || { echo "ERROR: Python not found: ${PYTHON_BIN}"; exit 1; }
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
mkdir -p logs data
test -d "${SOURCE_DATA_DIR}" || { echo "ERROR: prepared source data not found: ${SOURCE_DATA_DIR}"; exit 1; }
test -f weights/vae/autoencoder_epoch273.pt || { echo "ERROR: pretrained VAE is missing"; exit 1; }

echo "===== G1 V3 Adopt Verified Data ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Code dir: ${CODE_DIR}"
echo "Source data: ${SOURCE_DATA_DIR}"
echo "Conda env: ${G1_V3_CONDA_ENV}"
echo "Python: ${PYTHON_BIN}"
echo "Date: $(date)"

"${PYTHON_BIN}" -u adopt_g1_v3_prepared_data.py \
  --source-data-dir "${SOURCE_DATA_DIR}" \
  --target-data-dir data \
  --clean

"${PYTHON_BIN}" -u validate_g1_v3_dataset.py \
  --data-csv data/data_csv.csv \
  --data-dir data/input \
  --skip-label-values
"${PYTHON_BIN}" -u validate_g1_v3_preparation.py \
  --data-dir data \
  --expected-fake 265 \
  --expected-total 1030 \
  --expected-train 823 \
  --expected-val 103 \
  --expected-test 104

DRY_RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/g1v3_vae_dry.XXXXXX")"
trap 'rm -rf "${DRY_RUN_DIR}"' EXIT
"${PYTHON_BIN}" -u finetune_vae.py \
  --data_csv data/data_csv.csv \
  --data_dir data/input \
  --output_dir "${DRY_RUN_DIR}" \
  --dry_run

echo "===== G1 V3 data adoption passed ====="
echo "Training input link: $(readlink data/input)"
echo "Inference input link: $(readlink data/input_inference)"
echo "CSV: data/data_csv.csv"
echo "Split: 823 train / 103 val / 104 locked test"
echo "Missing-T2W inference: 265"
echo "Date: $(date)"
