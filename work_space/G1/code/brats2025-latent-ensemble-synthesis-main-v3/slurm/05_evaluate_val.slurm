#!/bin/bash
#SBATCH -J g1v3_eval
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/g1v3_eval_%j.out
#SBATCH -e logs/g1v3_eval_%j.err

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
SYNTHESIS_TYPE="${SYNTHESIS_TYPE:-ensamble}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EVAL_RUN_ID="${EVAL_RUN_ID:-run_${SLURM_JOB_ID}}"
case "${SYNTHESIS_TYPE}" in encdec|bbdm|ensamble) ;; *) echo "ERROR: invalid SYNTHESIS_TYPE"; exit 1 ;; esac
case "${EVAL_SPLIT}" in val) EXPECTED_CASES=103 ;; test) EXPECTED_CASES=104 ;; *) echo "ERROR: EVAL_SPLIT must be val or test"; exit 1 ;; esac

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V3_CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
test -x "${PYTHON_BIN}" || { echo "ERROR: Python not found: ${PYTHON_BIN}"; exit 1; }
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-data/evaluation/${EVAL_SPLIT}/${EVAL_RUN_ID}}"
mkdir -p logs "${EVAL_RUN_DIR}/synthesized"
test -f data/selected_vae.json || { echo "ERROR: run stage 03 first"; exit 1; }
G1_VAE_WEIGHTS="$("${PYTHON_BIN}" -c 'import json; print(json.load(open("data/selected_vae.json"))["selected_weights"])')"
export G1_VAE_WEIGHTS
test -f "${G1_VAE_WEIGHTS}" || { echo "ERROR: selected VAE not found: ${G1_VAE_WEIGHTS}"; exit 1; }

if [ -z "${G1_ENCDEC_CHECKPOINT_DIR:-}" ]; then
  test -f training/endec/latest_run.txt || { echo "ERROR: no completed EncDec run"; exit 1; }
  ENDEC_RUN_DIR="$(head -n 1 training/endec/latest_run.txt)"
  G1_ENCDEC_CHECKPOINT_DIR="${ENDEC_RUN_DIR}/check_points"
fi
if [ -z "${G1_BBDM_CHECKPOINT_DIR:-}" ]; then
  test -f training/bbdm/latest_run.txt || { echo "ERROR: no completed BBDM run"; exit 1; }
  BBDM_RUN_DIR="$(head -n 1 training/bbdm/latest_run.txt)"
  G1_BBDM_CHECKPOINT_DIR="${BBDM_RUN_DIR}/check_points"
else
  BBDM_RUN_DIR="$(cd "${G1_BBDM_CHECKPOINT_DIR}/.." && pwd)"
fi
export G1_ENCDEC_CHECKPOINT_DIR G1_BBDM_CHECKPOINT_DIR
test -d "${G1_ENCDEC_CHECKPOINT_DIR}" || { echo "ERROR: EncDec checkpoint directory missing"; exit 1; }
test -d "${G1_BBDM_CHECKPOINT_DIR}" || { echo "ERROR: BBDM checkpoint directory missing"; exit 1; }
test -f "${BBDM_RUN_DIR}/model_config.json" || { echo "ERROR: BBDM model_config.json missing"; exit 1; }
G1_BBDM_S="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["networks_config"]["bb_scheduler"]["s"])' "${BBDM_RUN_DIR}/model_config.json")"
export G1_BBDM_S EVAL_RUN_DIR EVAL_SPLIT SYNTHESIS_TYPE EXPECTED_CASES

echo "===== G1 V3 Evaluate ${EVAL_SPLIT} ====="
echo "Expected cases: ${EXPECTED_CASES}"
echo "Synthesis type: ${SYNTHESIS_TYPE}"
echo "VAE: ${G1_VAE_WEIGHTS}"
echo "EncDec checkpoints: ${G1_ENCDEC_CHECKPOINT_DIR}"
echo "BBDM checkpoints: ${G1_BBDM_CHECKPOINT_DIR}"
echo "BBDM s: ${G1_BBDM_S}"
echo "Evaluation run: ${EVAL_RUN_DIR}"
nvidia-smi

"${PYTHON_BIN}" -u evaluate.py \
  --input_dir data/input \
  --data_csv data/data_csv.csv \
  --split "${EVAL_SPLIT}" \
  --synthesis_type "${SYNTHESIS_TYPE}" \
  --gpu_id 0 \
  --verbose \
  --save_csv "${EVAL_RUN_DIR}/metrics.csv" \
  --save_output \
  --output_dir "${EVAL_RUN_DIR}/synthesized"

"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import configs
import synthesis.utils as utils

run_dir = Path(os.environ["EVAL_RUN_DIR"]).resolve()
with (run_dir / "metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
expected = int(os.environ["EXPECTED_CASES"])
if len(rows) != expected:
    raise RuntimeError(f"Evaluation row count {len(rows)} != expected {expected}")
outputs = list((run_dir / "synthesized").glob("*.nii.gz"))
if len(outputs) != expected:
    raise RuntimeError(f"Evaluation NIfTI count {len(outputs)} != expected {expected}")

metric_keys = ("whole_SSIM", "whole_PSNR", "whole_MSE", "whole_MAE", "brain_SSIM", "brain_PSNR", "brain_MSE", "brain_MAE")
summary = {key: sum(float(row[key]) for row in rows) / len(rows) for key in metric_keys}
record = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "split": os.environ["EVAL_SPLIT"],
    "case_count": len(rows),
    "synthesis_type": os.environ["SYNTHESIS_TYPE"],
    "vae_weights": configs.PATH_NAME_WEIGHTS_VAE,
    "encdec_checkpoint": configs.PATH_NAME_WEIGHTS_ENCDEC or utils.get_chkpoint_path(configs.PATH_WEIGHTS_ENCDEC),
    "bbdm_checkpoint": configs.PATH_NAME_WEIGHTS_BBDM or utils.get_chkpoint_path(configs.PATH_WEIGHTS_BBDM),
    "bbdm_s": configs.BBDM_S,
    "mean_metrics": summary,
    "operator_approved": False,
}
(run_dir / "evaluation_run.json").write_text(json.dumps(record, indent=2) + "\n")
(run_dir.parent / "latest_run.txt").write_text(str(run_dir) + "\n")
print(json.dumps(record, indent=2))
PY

cp "${EVAL_RUN_DIR}/metrics.csv" "data/eval_metrics_${EVAL_SPLIT}.csv"
echo "===== Evaluation finished; inspect metrics/images before stage 06 ====="
echo "Run: ${EVAL_RUN_DIR}"
echo "Date: $(date)"
