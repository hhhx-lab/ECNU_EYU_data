#!/bin/bash
#SBATCH -J g1v3_train
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 120:00:00
#SBATCH -o logs/g1v3_train_%j.out
#SBATCH -e logs/g1v3_train_%j.err

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

: "${TRAIN_TARGET:?Set TRAIN_TARGET=endec or TRAIN_TARGET=bbdm}"
case "${TRAIN_TARGET}" in
  endec|bbdm) ;;
  *) echo "ERROR: TRAIN_TARGET must be endec or bbdm"; exit 1 ;;
esac

G1_V3_CONDA_ENV="${G1_V3_CONDA_ENV:-segmamba}"
RUN_ID="${G1_MODEL_RUN_ID:-${TRAIN_TARGET}_${SLURM_JOB_ID}}"
ENDEC_BATCH_SIZE="${G1_ENCDEC_BATCH_SIZE:-12}"
ENDEC_MAX_STEPS="${G1_ENCDEC_MAX_STEPS:-67000}"
BBDM_BATCH_SIZE="${G1_BBDM_BATCH_SIZE:-8}"
BBDM_MAX_STEPS="${G1_BBDM_MAX_STEPS:-201000}"
SAVE_INTERVAL="${G1_SAVE_CHECKPOINT_INTERVAL:-5000}"
export G1_ENCDEC_BATCH_SIZE="${ENDEC_BATCH_SIZE}"
export G1_ENCDEC_MAX_STEPS="${ENDEC_MAX_STEPS}"
export G1_BBDM_BATCH_SIZE="${BBDM_BATCH_SIZE}"
export G1_BBDM_MAX_STEPS="${BBDM_MAX_STEPS}"
export G1_SAVE_CHECKPOINT_INTERVAL="${SAVE_INTERVAL}"
export G1_BBDM_S="${G1_BBDM_S:-0.01}"
export G1_WEIGHT_DECAY="${G1_WEIGHT_DECAY:-0.0}"
export G1_ENCDEC_OUTPUT_PATH="${G1_ENCDEC_OUTPUT_PATH:-training/endec/${RUN_ID}}"
export G1_BBDM_OUTPUT_PATH="${G1_BBDM_OUTPUT_PATH:-training/bbdm/${RUN_ID}}"
export G1_RESUME_CHECKPOINT="${G1_RESUME_CHECKPOINT:-}"

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
for required in \
  data/data_csv.csv \
  data/selected_vae.json \
  data/channel_weights.json; do
  test -f "${required}" || { echo "ERROR: missing ${required}; run stage 03 first"; exit 1; }
done
for required_dir in data/latents data/attention_masks data/lesion_weights; do
  test -d "${required_dir}" || { echo "ERROR: missing ${required_dir}; run stage 03 first"; exit 1; }
done
if [ -n "${G1_RESUME_CHECKPOINT}" ]; then
  test -f "${G1_RESUME_CHECKPOINT}" || { echo "ERROR: resume checkpoint not found: ${G1_RESUME_CHECKPOINT}"; exit 1; }
fi

G1_VAE_WEIGHTS="$("${PYTHON_BIN}" -c 'import json; print(json.load(open("data/selected_vae.json"))["selected_weights"])')"
export G1_VAE_WEIGHTS
test -f "${G1_VAE_WEIGHTS}" || { echo "ERROR: selected VAE not found: ${G1_VAE_WEIGHTS}"; exit 1; }

echo "===== G1 V3 Train ${TRAIN_TARGET} ====="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "VAE: ${G1_VAE_WEIGHTS}"
echo "BBDM s: ${G1_BBDM_S}"
echo "Weight decay: ${G1_WEIGHT_DECAY}"
echo "Resume checkpoint: ${G1_RESUME_CHECKPOINT:-NONE}"
echo "EncDec defaults: batch=${ENDEC_BATCH_SIZE}, steps=${ENDEC_MAX_STEPS}"
echo "BBDM defaults: batch=${BBDM_BATCH_SIZE}, steps=${BBDM_MAX_STEPS}"
echo "Checkpoint interval: ${SAVE_INTERVAL}"
echo "Python: ${PYTHON_BIN}"
nvidia-smi

if [ "${TRAIN_TARGET}" = "endec" ]; then
  "${PYTHON_BIN}" -u training_endec.py \
    --use_seg_loss \
    --batch_size "${ENDEC_BATCH_SIZE}" \
    --max_train_steps "${ENDEC_MAX_STEPS}" \
    --save_checkpoint_interval "${SAVE_INTERVAL}"
  RUN_ABS="$(cd "${G1_ENCDEC_OUTPUT_PATH}" && pwd)"
else
  "${PYTHON_BIN}" -u training_bbdm.py \
    --use_seg_loss \
    --batch_size "${BBDM_BATCH_SIZE}" \
    --max_train_steps "${BBDM_MAX_STEPS}" \
    --save_checkpoint_interval "${SAVE_INTERVAL}"
  RUN_ABS="$(cd "${G1_BBDM_OUTPUT_PATH}" && pwd)"
fi

export RUN_ABS TRAIN_TARGET
"${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import synthesis.utils as utils

run_dir = Path(os.environ["RUN_ABS"])
config_path = run_dir / "model_config.json"
if not config_path.is_file():
    raise FileNotFoundError(config_path)
checkpoint = Path(utils.get_chkpoint_path(str(run_dir / "check_points"))).resolve()
config = json.loads(config_path.read_text())
args = config["args_train"]
expected = {
    "endec": (int(os.environ.get("G1_ENCDEC_BATCH_SIZE", "12")), int(os.environ.get("G1_ENCDEC_MAX_STEPS", "67000"))),
    "bbdm": (int(os.environ.get("G1_BBDM_BATCH_SIZE", "8")), int(os.environ.get("G1_BBDM_MAX_STEPS", "201000"))),
}[os.environ["TRAIN_TARGET"]]
assert args["batch_size"] == expected[0], (args, expected)
assert args["max_train_steps"] == expected[1], (args, expected)
assert args["use_seg_loss"] is True, args
record = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "target": os.environ["TRAIN_TARGET"],
    "run_dir": str(run_dir),
    "checkpoint": str(checkpoint),
    "vae_weights": os.environ["G1_VAE_WEIGHTS"],
    "bbdm_s": float(os.environ["G1_BBDM_S"]),
    "weight_decay": float(os.environ["G1_WEIGHT_DECAY"]),
    "batch_size": args["batch_size"],
    "max_train_steps": args["max_train_steps"],
}
(run_dir / "training_run.json").write_text(json.dumps(record, indent=2) + "\n")
latest = run_dir.parent / "latest_run.txt"
latest.write_text(str(run_dir) + "\n")
print(json.dumps(record, indent=2))
PY

echo "===== ${TRAIN_TARGET} training finished ====="
echo "Run: ${RUN_ABS}"
echo "Date: $(date)"
