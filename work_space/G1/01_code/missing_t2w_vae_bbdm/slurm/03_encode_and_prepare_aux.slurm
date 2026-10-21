#!/bin/bash
#SBATCH -J g1v3_encode
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/g1v3_encode_%j.out
#SBATCH -e logs/g1v3_encode_%j.err

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

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V3_CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
test -x "${PYTHON_BIN}" || { echo "ERROR: Python not found: ${PYTHON_BIN}"; exit 1; }
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
mkdir -p logs
test -f data/data_csv.csv || { echo "ERROR: run stage 01 first"; exit 1; }

if [ -z "${VAE_RUN_DIR:-}" ]; then
  test -f training/vae_finetuned/latest_run.txt || { echo "ERROR: no completed VAE run selected"; exit 1; }
  VAE_RUN_DIR="$(head -n 1 training/vae_finetuned/latest_run.txt)"
fi
SELECTION_JSON="${VAE_RUN_DIR}/vae_selection.json"
test -f "${SELECTION_JSON}" || { echo "ERROR: missing ${SELECTION_JSON}"; exit 1; }

G1_VAE_WEIGHTS="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_weights"])' "${SELECTION_JSON}")"
export G1_VAE_WEIGHTS
test -f "${G1_VAE_WEIGHTS}" || { echo "ERROR: selected VAE not found: ${G1_VAE_WEIGHTS}"; exit 1; }

echo "===== G1 V3 Encode Latents and Prepare Auxiliary Inputs ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "VAE run: ${VAE_RUN_DIR}"
echo "Selected VAE: ${G1_VAE_WEIGHTS}"
echo "Python: ${PYTHON_BIN}"
echo "Date: $(date)"
nvidia-smi

rm -rf data/latents data/attention_masks data/lesion_weights
"${PYTHON_BIN}" -u test_vae.py
"${PYTHON_BIN}" -u preprocess.py \
  --vae-weights "${G1_VAE_WEIGHTS}" \
  --respect-existing-csv \
  --clean-latents \
  --device cuda:0
"${PYTHON_BIN}" -u validate_g1_v3_dataset.py --skip-label-values
"${PYTHON_BIN}" -u generate_attmask.py
"${PYTHON_BIN}" -u precompute_lesion_weights.py \
  --data_csv data/data_csv.csv \
  --data_dir data/input \
  --transforms_dir data/latents \
  --output_dir data/lesion_weights
"${PYTHON_BIN}" -u compute_weights.py
cp "${SELECTION_JSON}" data/selected_vae.json

"${PYTHON_BIN}" - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

data = Path("data")
with (data / "data_csv.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 1030, len(rows)
assert Counter(row["split"] for row in rows) == {"train": 823, "val": 103, "test": 104}

latent_cases = [path for path in (data / "latents").iterdir() if path.is_dir()]
latent_files = list((data / "latents").glob("*/*_latent.npy"))
spatial_transforms = list((data / "latents").glob("*/spatial_transform.json"))
attention_files = list((data / "attention_masks").glob("*/*_attmask_64_64_40.npy"))
weight_files = list((data / "lesion_weights").glob("*/*_lesion_weights_64_64_40.npy"))
assert len(latent_cases) == 1030, len(latent_cases)
assert len(latent_files) == 1030 * 4, len(latent_files)
assert len(spatial_transforms) == 1030, len(spatial_transforms)
assert len(attention_files) == 1030, len(attention_files)
assert len(weight_files) == 1030, len(weight_files)

weights = json.loads((data / "channel_weights.json").read_text())
assert weights["num_training_subjects"] == 823, weights
assert len(weights["channel_importance_weights"]) == 4, weights
selection = json.loads((data / "selected_vae.json").read_text())
assert Path(selection["selected_weights"]).is_file(), selection
print("Auxiliary contract passed:", {
    "cases": len(latent_cases),
    "latents": len(latent_files),
    "spatial_transforms": len(spatial_transforms),
    "attention_masks": len(attention_files),
    "lesion_weights": len(weight_files),
})
PY

echo "===== Latent/auxiliary preparation passed ====="
echo "Latents: data/latents/"
echo "Attention masks: data/attention_masks/"
echo "Lesion weights: data/lesion_weights/"
echo "Channel weights: data/channel_weights.json"
echo "Selected VAE: data/selected_vae.json"
echo "Date: $(date)"
