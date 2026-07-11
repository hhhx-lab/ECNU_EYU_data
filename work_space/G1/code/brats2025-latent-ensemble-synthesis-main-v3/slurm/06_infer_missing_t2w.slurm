#!/bin/bash
#SBATCH -J g1v3_infer
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/g1v3_infer_%j.out
#SBATCH -e logs/g1v3_infer_%j.err

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
INFER_RUN_ID="${INFER_RUN_ID:-run_${SLURM_JOB_ID}}"
EXPECTED_INFERENCE_CASES="${EXPECTED_INFERENCE_CASES:-265}"
VAL_APPROVED="${VAL_APPROVED:-0}"
case "${SYNTHESIS_TYPE}" in encdec|bbdm|ensamble) ;; *) echo "ERROR: invalid SYNTHESIS_TYPE"; exit 1 ;; esac
if [ "${VAL_APPROVED}" != "1" ]; then
  echo "ERROR: validation is a manual gate. Inspect stage 05, then submit with VAL_APPROVED=1."
  exit 1
fi

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V3_CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
test -x "${PYTHON_BIN}" || { echo "ERROR: Python not found: ${PYTHON_BIN}"; exit 1; }
export PYTHONUNBUFFERED=1

cd "${CODE_DIR}"
INFER_RUN_DIR="${INFER_RUN_DIR:-data/output/${INFER_RUN_ID}}"
mkdir -p logs "${INFER_RUN_DIR}"
test -f data/selected_vae.json || { echo "ERROR: run stage 03 first"; exit 1; }
test -f data/evaluation/val/latest_run.txt || { echo "ERROR: no completed val evaluation"; exit 1; }

G1_VAE_WEIGHTS="$("${PYTHON_BIN}" -c 'import json; print(json.load(open("data/selected_vae.json"))["selected_weights"])')"
export G1_VAE_WEIGHTS
test -f "${G1_VAE_WEIGHTS}" || { echo "ERROR: selected VAE not found: ${G1_VAE_WEIGHTS}"; exit 1; }

VAL_RUN_DIR="$(head -n 1 data/evaluation/val/latest_run.txt)"
VAL_MANIFEST="${VAL_RUN_DIR}/evaluation_run.json"
test -f "${VAL_MANIFEST}" || { echo "ERROR: val evaluation manifest missing"; exit 1; }
G1_ENCDEC_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["encdec_checkpoint"])' "${VAL_MANIFEST}")"
G1_BBDM_CHECKPOINT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["bbdm_checkpoint"])' "${VAL_MANIFEST}")"
G1_BBDM_S="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["bbdm_s"])' "${VAL_MANIFEST}")"
G1_ENCDEC_CHECKPOINT_DIR="$(dirname "${G1_ENCDEC_CHECKPOINT}")"
G1_BBDM_CHECKPOINT_DIR="$(dirname "${G1_BBDM_CHECKPOINT}")"
export G1_ENCDEC_CHECKPOINT G1_BBDM_CHECKPOINT G1_BBDM_S
export G1_ENCDEC_CHECKPOINT_DIR G1_BBDM_CHECKPOINT_DIR
test -f "${G1_ENCDEC_CHECKPOINT}" || { echo "ERROR: evaluated EncDec checkpoint missing"; exit 1; }
test -f "${G1_BBDM_CHECKPOINT}" || { echo "ERROR: evaluated BBDM checkpoint missing"; exit 1; }

CASE_COUNT="$("${PYTHON_BIN}" -c 'from pathlib import Path; print(sum(p.is_dir() for p in Path("data/input_inference").iterdir()))')"
if [ "${CASE_COUNT}" -ne "${EXPECTED_INFERENCE_CASES}" ]; then
  echo "ERROR: inference case count ${CASE_COUNT} != expected ${EXPECTED_INFERENCE_CASES}"
  exit 2
fi

export INFER_RUN_DIR SYNTHESIS_TYPE EXPECTED_INFERENCE_CASES VAL_RUN_DIR
echo "===== G1 V3 Infer Missing T2W ====="
echo "Cases: ${CASE_COUNT}"
echo "Synthesis type: ${SYNTHESIS_TYPE}"
echo "VAE: ${G1_VAE_WEIGHTS}"
echo "EncDec checkpoint: ${G1_ENCDEC_CHECKPOINT}"
echo "BBDM checkpoint: ${G1_BBDM_CHECKPOINT}"
echo "BBDM s: ${G1_BBDM_S}"
echo "Approved validation run: ${VAL_RUN_DIR}"
echo "Output run: ${INFER_RUN_DIR}"
nvidia-smi

"${PYTHON_BIN}" -u main.py \
  --input_dir data/input_inference \
  --output_dir "${INFER_RUN_DIR}" \
  --synthesis_type "${SYNTHESIS_TYPE}" \
  --gpu_id 0 \
  --verbose

"${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import configs

run_dir = Path(os.environ["INFER_RUN_DIR"]).resolve()
case_dirs = [path for path in run_dir.iterdir() if path.is_dir()]
generated = list(run_dir.glob("*/*-t2w.nii.gz"))
expected = int(os.environ["EXPECTED_INFERENCE_CASES"])
if len(case_dirs) != expected:
    raise RuntimeError(f"Output case count {len(case_dirs)} != expected {expected}")
if len(generated) != expected:
    raise RuntimeError(f"Generated T2W count {len(generated)} != expected {expected}")
for case_dir in case_dirs:
    case_id = case_dir.name
    required = [
        case_dir / f"{case_id}-t1n.nii.gz",
        case_dir / f"{case_id}-t1c.nii.gz",
        case_dir / f"{case_id}-t2w.nii.gz",
        case_dir / f"{case_id}-t2f.nii.gz",
        case_dir / f"{case_id}-seg.nii.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete output case {case_id}: {missing}")

record = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "case_count": len(case_dirs),
    "synthesis_type": os.environ["SYNTHESIS_TYPE"],
    "validation_run": os.environ["VAL_RUN_DIR"],
    "operator_approved_validation": True,
    "vae_weights": configs.PATH_NAME_WEIGHTS_VAE,
    "encdec_checkpoint": configs.PATH_NAME_WEIGHTS_ENCDEC,
    "bbdm_checkpoint": configs.PATH_NAME_WEIGHTS_BBDM,
    "bbdm_s": configs.BBDM_S,
}
(run_dir / "inference_run.json").write_text(json.dumps(record, indent=2) + "\n")
(run_dir.parent / "latest_run.txt").write_text(str(run_dir) + "\n")
print(json.dumps(record, indent=2))
PY

echo "===== Missing-T2W inference finished ====="
echo "G2 handoff: ${INFER_RUN_DIR}/"
echo "Date: $(date)"
