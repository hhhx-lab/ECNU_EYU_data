#!/bin/bash
#SBATCH -J g2_v3_pairqc
#SBATCH -p cpu_96G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o work_space/G2/logs/g2_v3_pairqc_%j.out
#SBATCH -e work_space/G2/logs/g2_v3_pairqc_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [ -z "${PROJECT_ROOT}" ]; then
  echo "ERROR: set PROJECT_ROOT or submit from the repository root" >&2
  exit 1
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
# Accept either repo README or the G2 paired-QC script as root marker.
if [ ! -f "${PROJECT_ROOT}/README.md" ] && [ ! -f "${PROJECT_ROOT}/work_space/G2/code/g2_v3_paired_quality.py" ]; then
  echo "ERROR: PROJECT_ROOT looks wrong: ${PROJECT_ROOT}" >&2
  echo "Expected README.md or work_space/G2/code/g2_v3_paired_quality.py" >&2
  exit 1
fi
G1_V3_CODE_DIR="${G1_V3_CODE_DIR:-${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3}"
PAIR_QC_SCRIPT="${PROJECT_ROOT}/work_space/G2/code/g2_v3_paired_quality.py"
G1_V3_CONDA_ENV="${G1_V3_CONDA_ENV:-segmamba}"
EXPECTED_CASES="${EXPECTED_CASES:-103}"

test -f "${PAIR_QC_SCRIPT}" || { echo "ERROR: paired QC script missing: ${PAIR_QC_SCRIPT}" >&2; exit 1; }
test -d "${G1_V3_CODE_DIR}/data/input" || { echo "ERROR: G1 V3 input missing" >&2; exit 1; }
test -f "${G1_V3_CODE_DIR}/data/evaluation/val/latest_run.txt" || {
  echo "ERROR: no completed G1 V3 validation run" >&2
  exit 1
}

VAL_RUN_DIR="${G1_V3_VAL_RUN_DIR:-$(head -n 1 "${G1_V3_CODE_DIR}/data/evaluation/val/latest_run.txt")}"
if [[ "${VAL_RUN_DIR}" != /* ]]; then
  VAL_RUN_DIR="${G1_V3_CODE_DIR}/${VAL_RUN_DIR}"
fi
VAL_RUN_DIR="$(cd "${VAL_RUN_DIR}" && pwd)"
VAL_RUN_ID="$(basename "${VAL_RUN_DIR}")"
SYNTHETIC_ROOT="${VAL_RUN_DIR}/synthesized"
STAGE5_METRICS="${VAL_RUN_DIR}/metrics.csv"
OUTPUT_ROOT="${G2_V3_PAIR_QC_OUTPUT:-${PROJECT_ROOT}/work_space/G2/results/qc/v3_paired_validation/${VAL_RUN_ID}}"

test -d "${SYNTHETIC_ROOT}" || { echo "ERROR: synthesized directory missing: ${SYNTHETIC_ROOT}" >&2; exit 1; }
test -f "${STAGE5_METRICS}" || { echo "ERROR: stage-5 metrics missing: ${STAGE5_METRICS}" >&2; exit 1; }
mkdir -p "${PROJECT_ROOT}/work_space/G2/logs"

module purge
module load apps/envs/miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${G1_V3_CONDA_ENV}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

OVERWRITE_ARGS=()
if [ "${G2_QC_OVERWRITE:-0}" = "1" ]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

echo "===== G2 V3 Paired Validation QC ====="
echo "Validation run : ${VAL_RUN_DIR}"
echo "Real root      : ${G1_V3_CODE_DIR}/data/input"
echo "Synthetic root : ${SYNTHETIC_ROOT}"
echo "Output root    : ${OUTPUT_ROOT}"
echo "Expected cases : ${EXPECTED_CASES}"

"${PYTHON_BIN}" -u "${PAIR_QC_SCRIPT}" \
  --real-root "${G1_V3_CODE_DIR}/data/input" \
  --synthetic-root "${SYNTHETIC_ROOT}" \
  --stage5-metrics "${STAGE5_METRICS}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-cases "${EXPECTED_CASES}" \
  "${OVERWRITE_ARGS[@]}"

CASE_ROWS="$(( $(wc -l < "${OUTPUT_ROOT}/case_metrics.csv") - 1 ))"
REGION_ROWS="$(( $(wc -l < "${OUTPUT_ROOT}/region_metrics.csv") - 1 ))"
MONTAGE_COUNT="$(find "${OUTPUT_ROOT}/montages" -maxdepth 1 -type f -name '*.png' | wc -l)"
test "${CASE_ROWS}" -eq "${EXPECTED_CASES}" || { echo "ERROR: case row count ${CASE_ROWS}" >&2; exit 2; }
test "${REGION_ROWS}" -eq "$(( EXPECTED_CASES * 5 ))" || { echo "ERROR: region row count ${REGION_ROWS}" >&2; exit 2; }
test "${MONTAGE_COUNT}" -eq "${EXPECTED_CASES}" || { echo "ERROR: montage count ${MONTAGE_COUNT}" >&2; exit 2; }

echo "===== G2 V3 Paired QC finished ====="
echo "Cases: ${CASE_ROWS}; regions: ${REGION_ROWS}; montages: ${MONTAGE_COUNT}"
echo "Review: ${OUTPUT_ROOT}/QC_REPORT.md"
