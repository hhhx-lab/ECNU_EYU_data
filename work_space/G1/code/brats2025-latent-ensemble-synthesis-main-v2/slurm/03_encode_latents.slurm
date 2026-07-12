#!/bin/bash
#SBATCH -J g1v2_encode
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/g1v2_encode_%j.out
#SBATCH -e logs/g1v2_encode_%j.err

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

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V2_CONDA_ENV}"
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
mkdir -p logs

if [ -z "${VAE_RUN_DIR:-}" ]; then
  test -f training/vae_finetuned/latest_run.txt || { echo "ERROR: no VAE run selected"; exit 1; }
  VAE_RUN_DIR="$(head -n 1 training/vae_finetuned/latest_run.txt)"
fi
SELECTION_JSON="${VAE_RUN_DIR}/vae_selection.json"
test -f "${SELECTION_JSON}" || { echo "ERROR: missing ${SELECTION_JSON}"; exit 1; }

G1_VAE_WEIGHTS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_weights"])' "${SELECTION_JSON}")"
export G1_VAE_WEIGHTS
test -f "${G1_VAE_WEIGHTS}" || { echo "ERROR: selected VAE not found: ${G1_VAE_WEIGHTS}"; exit 1; }

echo "===== G1 V2 Encode Latents ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "VAE run: ${VAE_RUN_DIR}"
echo "Selected VAE: ${G1_VAE_WEIGHTS}"
echo "Date: $(date)"
nvidia-smi

python -u test_vae.py
python -u preprocess.py \
  --vae-weights "${G1_VAE_WEIGHTS}" \
  --respect-existing-csv \
  --clean-latents \
  --device cuda:0
python -u validate_g1_v2_dataset.py --skip-label-values
python -u generate_attmask.py
python -u compute_weights.py
cp "${SELECTION_JSON}" data/selected_vae.json

echo "===== Latent encoding passed ====="
echo "Latents: data/latents/"
echo "Selected VAE record: data/selected_vae.json"
echo "Date: $(date)"
