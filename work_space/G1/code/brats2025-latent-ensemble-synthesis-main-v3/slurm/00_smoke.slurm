#!/bin/bash
#SBATCH -J g1v3_smoke
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH -o logs/g1v3_smoke_%j.out
#SBATCH -e logs/g1v3_smoke_%j.err

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

# Early bootstrap marker (survives module failures; also ensures logs/ exists before modules).
mkdir -p "${CODE_DIR}/logs"
{
  echo "BOOTSTRAP $(date -Is 2>/dev/null || date)"
  echo "Job ID: ${SLURM_JOB_ID:-NA}"
  echo "Node: ${SLURMD_NODENAME:-NA}"
  echo "SubmitDir: ${SLURM_SUBMIT_DIR:-NA}"
  echo "CodeDir: ${CODE_DIR}"
  echo "PWD: $(pwd)"
  echo "PATH: ${PATH}"
} | tee "${CODE_DIR}/logs/g1v3_smoke_${SLURM_JOB_ID:-manual}.bootstrap"
# Also force a line to stdout/stderr immediately so Slurm log files are non-empty.
echo "BOOTSTRAP_STDOUT job=${SLURM_JOB_ID:-NA} node=${SLURMD_NODENAME:-NA}"
echo "BOOTSTRAP_STDERR job=${SLURM_JOB_ID:-NA}" >&2

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V3_CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
test -x "${PYTHON_BIN}" || { echo "ERROR: Python not found: ${PYTHON_BIN}"; exit 1; }
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
echo "Resolved code dir: ${CODE_DIR}"
echo "Working directory: $(pwd)"
ls -ld "${CODE_DIR}" "${CODE_DIR}/logs" 2>/dev/null || true
mkdir -p logs

echo "===== G1 V3 A100 Smoke ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Conda env: ${G1_V3_CONDA_ENV}"
echo "Python: ${PYTHON_BIN}"
nvidia-smi

"${PYTHON_BIN}" - <<'PY'
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

"${PYTHON_BIN}" -m py_compile \
  adopt_g1_v3_prepared_data.py \
  preprocess.py \
  prepare_g1_v3_data.py \
  filter_g1_v3_invalid_labels.py \
  mark_g1_train_val_test_split.py \
  validate_g1_v3_dataset.py \
  validate_g1_v3_preparation.py \
  vae_patch_sampling.py \
  finetune_vae.py \
  validate_vae_recon.py \
  smoke_test_models.py \
  training_endec.py \
  training_bbdm.py \
  evaluate.py \
  main.py
"${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" -u test_vae.py
"${PYTHON_BIN}" -u smoke_test_models.py --device cuda:0 --batch-size 2

echo "===== A100 smoke passed ====="
echo "Date: $(date)"
