#!/bin/bash
#SBATCH -J g1v2_infer
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/g1v2_infer_%j.out
#SBATCH -e logs/g1v2_infer_%j.err

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
SYNTHESIS_TYPE="${SYNTHESIS_TYPE:-ensamble}"
INFER_RUN_ID="${INFER_RUN_ID:-run_${SLURM_JOB_ID}}"

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V2_CONDA_ENV}"
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
INFER_RUN_DIR="${INFER_RUN_DIR:-data/output/${INFER_RUN_ID}}"
export INFER_RUN_DIR SYNTHESIS_TYPE
mkdir -p logs "${INFER_RUN_DIR}"
test -f data/selected_vae.json || { echo "ERROR: data/selected_vae.json not found"; exit 1; }
test -f data/evaluation/val/latest_run.txt || { echo "ERROR: no completed val evaluation"; exit 1; }
G1_VAE_WEIGHTS="$(python -c 'import json; print(json.load(open("data/selected_vae.json"))["selected_weights"])')"
export G1_VAE_WEIGHTS
test -f "${G1_VAE_WEIGHTS}" || { echo "ERROR: selected VAE not found: ${G1_VAE_WEIGHTS}"; exit 1; }

VAL_RUN_DIR="$(head -n 1 data/evaluation/val/latest_run.txt)"
VAL_MANIFEST="${VAL_RUN_DIR}/evaluation_run.json"
test -f "${VAL_MANIFEST}" || { echo "ERROR: val evaluation manifest missing"; exit 1; }
G1_ENCDEC_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["encdec_checkpoint"])' "${VAL_MANIFEST}")"
G1_BBDM_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["bbdm_checkpoint"])' "${VAL_MANIFEST}")"
G1_ENCDEC_CHECKPOINT_DIR="$(dirname "${G1_ENCDEC_CHECKPOINT}")"
G1_BBDM_CHECKPOINT_DIR="$(dirname "${G1_BBDM_CHECKPOINT}")"
G1_BBDM_S="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["bbdm_s"])' "${VAL_MANIFEST}")"
export G1_ENCDEC_CHECKPOINT G1_BBDM_CHECKPOINT
export G1_ENCDEC_CHECKPOINT_DIR G1_BBDM_CHECKPOINT_DIR
test -f "${G1_ENCDEC_CHECKPOINT}" || { echo "ERROR: evaluated EncDec checkpoint missing"; exit 1; }
test -f "${G1_BBDM_CHECKPOINT}" || { echo "ERROR: evaluated BBDM checkpoint missing"; exit 1; }
test -d "${G1_ENCDEC_CHECKPOINT_DIR}" || { echo "ERROR: EncDec checkpoint dir missing"; exit 1; }
test -d "${G1_BBDM_CHECKPOINT_DIR}" || { echo "ERROR: BBDM checkpoint dir missing"; exit 1; }
export G1_BBDM_S

CASE_COUNT="$(find data/input_inference -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
if [ "${CASE_COUNT}" -eq 0 ]; then
  echo "ERROR: data/input_inference contains no true missing-T2W cases."
  exit 2
fi

echo "===== G1 V2 Infer Missing T2W ====="
echo "Cases: ${CASE_COUNT}"
echo "Synthesis type: ${SYNTHESIS_TYPE}"
echo "VAE: ${G1_VAE_WEIGHTS}"
echo "EncDec checkpoints: ${G1_ENCDEC_CHECKPOINT_DIR}"
echo "BBDM checkpoints: ${G1_BBDM_CHECKPOINT_DIR}"
echo "BBDM s: ${G1_BBDM_S}"
echo "Output run: ${INFER_RUN_DIR}"
nvidia-smi

python -u main.py \
  --input_dir data/input_inference \
  --output_dir "${INFER_RUN_DIR}" \
  --synthesis_type "${SYNTHESIS_TYPE}" \
  --gpu_id 0 \
  --verbose

python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import configs
import synthesis.utils as utils

record = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "synthesis_type": os.environ["SYNTHESIS_TYPE"],
    "vae_weights": configs.PATH_NAME_WEIGHTS_VAE,
    "encdec_checkpoint": configs.PATH_NAME_WEIGHTS_ENCDEC or utils.get_chkpoint_path(configs.PATH_WEIGHTS_ENCDEC),
    "bbdm_checkpoint": configs.PATH_NAME_WEIGHTS_BBDM or utils.get_chkpoint_path(configs.PATH_WEIGHTS_BBDM),
    "bbdm_s": configs.BBDM_S,
}
run_dir = Path(os.environ["INFER_RUN_DIR"])
(run_dir / "inference_run.json").write_text(json.dumps(record, indent=2) + "\n")
(run_dir.parent / "latest_run.txt").write_text(str(run_dir.resolve()) + "\n")
PY

echo "===== Inference finished ====="
echo "Output: ${INFER_RUN_DIR}/"
echo "Date: $(date)"
