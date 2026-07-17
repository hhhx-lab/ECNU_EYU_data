#!/bin/bash
#SBATCH --job-name=s4_prepare_g2
#SBATCH --partition=cpu_96G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/s4_prepare_g2_%j.out
#SBATCH --error=logs/s4_prepare_g2_%j.err

set -euo pipefail

# Build the authoritative real-only G2 case-folder view used by S4. This job
# only creates symlinks and manifests; it never copies MRI volumes.
# Slurm may stage the script under /opt/gridview/slurm on compute nodes, so
# derive the project from the submit directory instead of BASH_SOURCE.
S4_ROOT="${S4_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${S4_ROOT}" ]]; then
  echo "S4_ROOT is required when SLURM_SUBMIT_DIR is unavailable." >&2
  exit 2
fi
PROJ="${PROJ:-$(cd "${S4_ROOT}/../.." && pwd)}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-${PROJ}/work_space/S3/data/MICCAI-LH-BraTS2025-MET-Challenge-Training}"
CORRECTED_LABELS_ROOT="${CORRECTED_LABELS_ROOT:-${PROJ}/work_space/S3/data/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels}"
G2_ROOT="${G2_ROOT:-${PROJ}/work_space/G2}"
G2_RESULTS_ROOT="${G2_RESULTS_ROOT:-${G2_ROOT}/results}"
G2_OUTPUT_ROOT="${G2_OUTPUT_ROOT:-${G2_RESULTS_ROOT}/materialized_s4_realonly}"
FIXED_SPLIT_ROOT="${FIXED_SPLIT_ROOT:-${G2_OUTPUT_ROOT}/BraTS2026_MET_RealOnly_S4_case_folders}"
FAKE_T2W_CSV="${FAKE_T2W_CSV:-${G2_RESULTS_ROOT}/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv}"

CONDA_SH="${CONDA_SH:-/public/software/apps/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats_swin}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for required in \
  "${RAW_DATA_ROOT}" \
  "${CORRECTED_LABELS_ROOT}" \
  "${G2_ROOT}/code/g2_build_realonly_from_raw.py" \
  "${G2_ROOT}/code/g2_create_train_val_test_split.py" \
  "${G2_ROOT}/code/g2_materialize_nnunet_dataset.py" \
  "${FAKE_T2W_CSV}" \
  "${CONDA_SH}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required input is missing: ${required}" >&2
    exit 1
  fi
done

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

mkdir -p "${S4_ROOT}/logs" "${G2_RESULTS_ROOT}/manifests" "${G2_RESULTS_ROOT}/splits"

echo "===== S4 G2 fixed-split preparation ====="
echo "Raw data          : ${RAW_DATA_ROOT}"
echo "Corrected labels  : ${CORRECTED_LABELS_ROOT}"
echo "Fixed split root  : ${FIXED_SPLIT_ROOT}"
echo "Conda environment : ${CONDA_ENV}"
echo "=========================================="

"${PYTHON_BIN}" - <<'PY'
import nibabel
import numpy
print("G2 preparation imports passed")
PY

"${PYTHON_BIN}" "${G2_ROOT}/code/g2_build_realonly_from_raw.py" \
  --project-root "${PROJ}" \
  --data-root "${RAW_DATA_ROOT}" \
  --corrected-labels-root "${CORRECTED_LABELS_ROOT}" \
  --results-root "${G2_RESULTS_ROOT}" \
  --fake-t2w-cases "${FAKE_T2W_CSV}" \
  --fail-if-no-valid-cases

"${PYTHON_BIN}" "${G2_ROOT}/code/g2_materialize_nnunet_dataset.py" \
  --results-root "${G2_RESULTS_ROOT}" \
  --real-mapping "${G2_RESULTS_ROOT}/manifests/nnunet_case_mapping_realonly.csv" \
  --master-split "${G2_RESULTS_ROOT}/splits/splits_master_train_val_test.json" \
  --fake-t2w-cases "${FAKE_T2W_CSV}" \
  --output-root "${G2_OUTPUT_ROOT}" \
  --case-folder-root "${FIXED_SPLIT_ROOT}" \
  --dataset-id 264 \
  --dataset-name BraTS2026_MET_RealOnly_S4 \
  --dataset-profile real-only \
  --mode symlink \
  --clean-output

FIXED_SPLIT_ROOT="${FIXED_SPLIT_ROOT}" "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["FIXED_SPLIT_ROOT"])
counts = {name: sum(path.is_dir() for path in (root / name).iterdir()) for name in ("train", "val", "test")}
print("fixed_split_case_counts=", counts)
if not all(counts.values()):
    raise SystemExit("one or more fixed split partitions are empty")
PY

echo "===== DONE: $(date) ====="
echo "S4 fixed split: ${FIXED_SPLIT_ROOT}"
