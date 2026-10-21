#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

PROJ="${PROJ:-/root/brats2026/ECNU_EYU_data}"
RUN_ROOT="${RUN_ROOT:-/root/brats2026/runs/g1_diffusion_v3}"
RAW_ROOT="${RAW_ROOT:-/root/brats2026/raw_source/MICCAI-LH-BraTS2025-MET-Challenge-Training}"
CORRECTED_LABEL_ROOT="${CORRECTED_LABEL_ROOT:-/root/brats2026/corrected_labels}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/root/brats2026/envs/g1_diffusion_v3}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${PROJ}/work_space/G2/results/manifests/g1_v2_source_manifest.csv}"
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
DATASET_DIR="${RUN_ROOT}/DataSet"
SPLIT_DIR="${RUN_ROOT}/splits/current"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
CSV_PATH="${SPLIT_DIR}/lesions.csv"
LOGDIR="${LOGDIR:-brats2026_diffusion_v3_edm_zscore}"

test -x "${PYTHON_BIN}"
test -d "${RAW_ROOT}"
test -d "${CORRECTED_LABEL_ROOT}"
test -s "${SOURCE_MANIFEST}"
mkdir -p "${RUN_ROOT}" "${SPLIT_DIR}" "${CHECKPOINT_ROOT}" "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/PREPARED.ok"

cd "${CODE_DIR}"
"${PYTHON_BIN}" tests/test_axis_contract.py

PREPARE_ARGS=(
  --project-root "${PROJ}"
  --source-manifest "${SOURCE_MANIFEST}"
  --output-dir "${DATASET_DIR}"
  --split-dir "${SPLIT_DIR}"
  --raw-root "${RAW_ROOT}"
  --corrected-label-root "${CORRECTED_LABEL_ROOT}"
)
if [[ -d "${DATASET_DIR}" && -f "${DATASET_DIR}/.g1_diffusion_dataset" ]]; then
  PREPARE_ARGS+=(--clean)
fi
"${PYTHON_BIN}" scripts/prepare_dataset_from_g2_manifest.py "${PREPARE_ARGS[@]}"

"${PYTHON_BIN}" src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${DATASET_DIR}" \
  --split_manifest "${SOURCE_MANIFEST}" \
  --csv_path "${CSV_PATH}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --crop_size 64 \
  --merge_dist 16 \
  --debug False

"${PYTHON_BIN}" scripts/filter_invalid_lesion_rows.py \
  --lesions "${CSV_PATH}" \
  --backup "${SPLIT_DIR}/lesions_before_scan_content_qc.csv" \
  --audit "${SPLIT_DIR}/lesion_scan_content_rejections.csv" \
  --summary "${SPLIT_DIR}/lesion_scan_content_qc.json" \
  --target-size 64

"${PYTHON_BIN}" scripts/validate_prepared_diffusion_dataset.py \
  --membership "${SPLIT_DIR}/dataset_membership.csv" \
  --lesions "${CSV_PATH}" \
  --expected-train 823 \
  --expected-val 103

printf 'created_at=%s\ncsv=%s\ndataset=%s\n' \
  "$(date -Is)" "${CSV_PATH}" "${DATASET_DIR}" > "${RUN_ROOT}/PREPARED.ok"
echo "PREPARE_DATASET_PASS run_root=${RUN_ROOT}"
