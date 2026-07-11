#!/bin/bash
#SBATCH -J g1v2_smoke
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH -o logs/g1v2_smoke_%j.out
#SBATCH -e logs/g1v2_smoke_%j.err

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

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
echo "Resolved code dir: ${CODE_DIR}"
echo "Working directory: $(pwd)"
ls -ld "${CODE_DIR}" "${CODE_DIR}/logs" 2>/dev/null || true
mkdir -p logs

echo "===== G1 V2 A100 Smoke ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Conda env: ${CONDA_ENV}"
nvidia-smi

python - <<'PY'
import torch
import monai
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("MONAI:", monai.__version__)
print("MAISI class:", AutoencoderKlMaisi.__name__)
if not torch.cuda.is_available():
    raise RuntimeError("A100 job started without a CUDA-visible GPU.")
PY

python -m py_compile \
  preprocess.py \
  filter_g1_v2_invalid_labels.py \
  mark_g1_train_val_test_split.py \
  validate_g1_v2_dataset.py \
  vae_patch_sampling.py \
  finetune_vae.py \
  validate_vae_recon.py \
  smoke_test_models.py \
  training_endec.py \
  training_bbdm.py \
  evaluate.py \
  main.py
python -m unittest discover -s tests -v
python -u test_vae.py
python -u smoke_test_models.py --device cuda:0 --batch-size 2

echo "===== A100 smoke passed ====="
echo "Date: $(date)"
