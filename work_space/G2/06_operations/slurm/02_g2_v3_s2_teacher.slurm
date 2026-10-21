#!/bin/bash
#SBATCH -J g2_v3_s2teach
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o work_space/G2/logs/g2_v3_s2_teacher_%j.out
#SBATCH -e work_space/G2/logs/g2_v3_s2_teacher_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "${PROJECT_ROOT}" ]; then
  echo "ERROR: set PROJECT_ROOT or submit from the repository root" >&2
  exit 1
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
G1_V3_CODE_DIR="${G1_V3_CODE_DIR:-${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3}"
G1_V3_VAL_RUN_DIR="${G1_V3_VAL_RUN_DIR:-$(head -n 1 "${G1_V3_CODE_DIR}/data/evaluation/val/latest_run.txt")}"
if [[ "${G1_V3_VAL_RUN_DIR}" != /* ]]; then
  G1_V3_VAL_RUN_DIR="${G1_V3_CODE_DIR}/${G1_V3_VAL_RUN_DIR}"
fi
G1_V3_VAL_RUN_DIR="$(cd "${G1_V3_VAL_RUN_DIR}" && pwd)"
RUN_ID="$(basename "${G1_V3_VAL_RUN_DIR}")"
SYNTHETIC_ROOT="${G1_V3_SYNTHETIC_ROOT:-${G1_V3_VAL_RUN_DIR}/synthesized}"

S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
S2_CHECKPOINT="${S2_TEACHER_CHECKPOINT:-${PROJECT_ROOT}/work_space/S2/results/checkpoint_final.pth}"
BASELINE_PREDICTIONS="${S2_BASELINE_PREDICTIONS:-${PROJECT_ROOT}/work_space/S2/results/s2_eval_results/prediction}"
MAPPING_CSV="${G2_REALONLY_MAPPING:-${PROJECT_ROOT}/work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv}"
SPLIT_JSON="${G2_FIXED_SPLIT:-${PROJECT_ROOT}/work_space/G2/results/splits/splits_final_train_val_test.json}"
TEACHER_SCRIPT="${PROJECT_ROOT}/work_space/G2/code/g2_s2_v3_teacher_eval.py"
EXPECTED_CASES="${EXPECTED_CASES:-103}"

RUNTIME_ROOT="${S2_TEACHER_RUNTIME_ROOT:-${PROJECT_ROOT}/work_space/S2/data/v3_teacher/${RUN_ID}}"
INPUT_ROOT="${RUNTIME_ROOT}/nnunet_input"
REFERENCE_ROOT="${RUNTIME_ROOT}/reference"
CASE_MAP="${RUNTIME_ROOT}/nnunet_to_source_id.tsv"
PREDICTION_ROOT="${S2_TEACHER_PREDICTION_ROOT:-${PROJECT_ROOT}/work_space/S2/results/v3_teacher_predictions/${RUN_ID}}"
REPORT_ROOT="${S2_TEACHER_REPORT_ROOT:-${PROJECT_ROOT}/work_space/G2/results/qc/v3_s2_teacher/${RUN_ID}}"

for path in \
  "${TEACHER_SCRIPT}" \
  "${S2_REPO}/infer.sh" \
  "${S2_CHECKPOINT}" \
  "${MAPPING_CSV}" \
  "${SPLIT_JSON}" \
  "${G1_V3_VAL_RUN_DIR}/metrics.csv"; do
  test -e "${path}" || { echo "ERROR: required input missing: ${path}" >&2; exit 1; }
done
test -d "${SYNTHETIC_ROOT}" || { echo "ERROR: synthetic root missing: ${SYNTHETIC_ROOT}" >&2; exit 1; }
test -d "${BASELINE_PREDICTIONS}" || { echo "ERROR: baseline predictions missing: ${BASELINE_PREDICTIONS}" >&2; exit 1; }

if [ -n "${S2_PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${S2_PYTHON_BIN}"
else
  module purge
  module load apps/envs/miniconda3/25.5.1
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${S2_CONDA_ENV:-segmamba}"
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
test -x "${PYTHON_BIN}" || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert metadata.version("nnunetv2") == "2.8.0", metadata.version("nnunetv2")
print("torch:", torch.__version__)
print("nnunetv2:", metadata.version("nnunetv2"))
print("gpu:", torch.cuda.get_device_name(0))
PY

mkdir -p "${PROJECT_ROOT}/work_space/G2/logs" "${RUNTIME_ROOT}"
rm -rf "${INPUT_ROOT}" "${REFERENCE_ROOT}" "${PREDICTION_ROOT}" "${REPORT_ROOT}"

"${PYTHON_BIN}" -u "${TEACHER_SCRIPT}" prepare \
  --project-root "${PROJECT_ROOT}" \
  --real-root "${G1_V3_CODE_DIR}/data/input" \
  --mapping-csv "${MAPPING_CSV}" \
  --split-json "${SPLIT_JSON}" \
  --stage5-metrics "${G1_V3_VAL_RUN_DIR}/metrics.csv" \
  --synthetic-root "${SYNTHETIC_ROOT}" \
  --input-root "${INPUT_ROOT}" \
  --reference-root "${REFERENCE_ROOT}" \
  --case-map "${CASE_MAP}" \
  --expected-cases "${EXPECTED_CASES}" \
  --materialize-mode symlink

export nnUNet_raw="${S2_REPO}/data/nnunet_raw"
export nnUNet_preprocessed="${S2_REPO}/data/nnunet_preprocessed"
export nnUNet_results="${S2_REPO}/data/nnunet_results"
"${PYTHON_BIN}" -u "${TEACHER_SCRIPT}" install-model \
  --checkpoint "${S2_CHECKPOINT}" \
  --nnunet-results-root "${nnUNet_results}" \
  --overwrite

export S2_EXPERIMENT_MODE=current
export S2_EXPECTED_INFERENCE_COUNT="${EXPECTED_CASES}"
export S2_USE_INFERENCE_TRAINER_SHIM=1
cd "${S2_REPO}"
bash infer.sh "${INPUT_ROOT}" "${PREDICTION_ROOT}"

PREDICTION_COUNT="$(find "${PREDICTION_ROOT}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)"
test "${PREDICTION_COUNT}" -eq "${EXPECTED_CASES}" || {
  echo "ERROR: S2 teacher prediction count ${PREDICTION_COUNT} != ${EXPECTED_CASES}" >&2
  exit 2
}

"${PYTHON_BIN}" -u "${TEACHER_SCRIPT}" compare \
  --baseline-prediction-root "${BASELINE_PREDICTIONS}" \
  --generated-prediction-root "${PREDICTION_ROOT}" \
  --reference-root "${REFERENCE_ROOT}" \
  --case-map "${CASE_MAP}" \
  --output-root "${REPORT_ROOT}" \
  --expected-cases "${EXPECTED_CASES}" \
  --overwrite

echo "===== G1 V3 S2 teacher evaluation finished ====="
echo "Run: ${RUN_ID}"
echo "Predictions: ${PREDICTION_ROOT}"
echo "Report: ${REPORT_ROOT}/TEACHER_REPORT.md"
echo "Summary: ${REPORT_ROOT}/summary.json"
