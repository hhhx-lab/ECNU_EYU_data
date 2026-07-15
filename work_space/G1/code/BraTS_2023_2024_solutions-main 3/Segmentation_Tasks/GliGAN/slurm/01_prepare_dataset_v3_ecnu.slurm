#!/bin/bash
#SBATCH -J g1d3_prep
#SBATCH -p cpu_96G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /public/home/%u/g1_diffusion_v3_production_20260715/logs/prepare_%j.out
#SBATCH -e /public/home/%u/g1_diffusion_v3_production_20260715/logs/prepare_%j.err

set -Eeuo pipefail

PROD_ROOT=${PROD_ROOT:-/public/home/${USER}/g1_diffusion_v3_production_20260715}
CODE_DIR=${CODE_DIR:-"${PROD_ROOT}/code/Segmentation_Tasks/GliGAN"}
PROJECT_ROOT=${PROJECT_ROOT:-/public/home/${USER}/projects/ECNU_EYU_data}
RAW_ROOT=${RAW_ROOT:-/public/home/${USER}/data}
MANIFEST=${MANIFEST:-"${PROD_ROOT}/manifests/g1_v2_source_manifest.csv"}
CORRECTED_ROOT=${CORRECTED_ROOT:-"${PROD_ROOT}/corrected_labels"}
DATASET_DIR=${DATASET_DIR:-"${PROD_ROOT}/DataSet"}
SPLIT_DIR=${SPLIT_DIR:-"${PROD_ROOT}/splits/current"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROD_ROOT}/checkpoints"}
LOGDIR=${LOGDIR:-brats2026_diffusion_v3_edm_zscore}
CSV_PATH=${CSV_PATH:-"${SPLIT_DIR}/lesions.csv"}
PYTHON_BIN=${PYTHON_BIN:-/public/home/${USER}/.conda/envs/segmamba/bin/python}
EXPECTED_TRAIN=${EXPECTED_TRAIN:-823}
EXPECTED_VAL=${EXPECTED_VAL:-103}

mkdir -p "${PROD_ROOT}/logs" "${SPLIT_DIR}" "${CHECKPOINT_ROOT}"
rm -f "${PROD_ROOT}/PREPARED.ok"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "code_dir=${CODE_DIR}"
echo "raw_root=${RAW_ROOT}"
echo "manifest=${MANIFEST}"
"${PYTHON_BIN}" -c 'import nibabel, numpy, scipy; print("python_dependencies=PASS")'

cd "${CODE_DIR}"
"${PYTHON_BIN}" tests/test_axis_contract.py

PREPARE_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --source-manifest "${MANIFEST}"
  --raw-root "${RAW_ROOT}"
  --corrected-label-root "${CORRECTED_ROOT}"
  --output-dir "${DATASET_DIR}"
  --split-dir "${SPLIT_DIR}"
)
if [[ -d "${DATASET_DIR}" && -f "${DATASET_DIR}/.g1_diffusion_dataset" ]]; then
  PREPARE_ARGS+=(--clean)
fi
"${PYTHON_BIN}" scripts/prepare_dataset_from_g2_manifest.py "${PREPARE_ARGS[@]}"

"${PYTHON_BIN}" src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${DATASET_DIR}" \
  --split_manifest "${MANIFEST}" \
  --csv_path "${CSV_PATH}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --crop_size 64 \
  --merge_dist 16 \
  --debug False

"${PYTHON_BIN}" - "${SPLIT_DIR}/dataset_membership.csv" "${CSV_PATH}" \
  "${EXPECTED_TRAIN}" "${EXPECTED_VAL}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

membership_path, lesions_path = sys.argv[1], sys.argv[2]
expected_train, expected_val = map(int, sys.argv[3:5])
with open(membership_path, newline="") as handle:
    membership = list(csv.DictReader(handle))
case_counts = Counter(row["split"] for row in membership)
assert case_counts == {"train": expected_train, "val": expected_val}, case_counts

group_splits = defaultdict(set)
for row in membership:
    group_splits[row["patient_group"]].add(row["split"])
leaks = [group for group, splits in group_splits.items() if len(splits) > 1]
assert not leaks, leaks[:10]

with open(lesions_path, newline="") as handle:
    lesions = list(csv.DictReader(handle))
assert lesions, "No lesion rows"
lesion_counts = Counter(row["split"] for row in lesions)
assert set(lesion_counts) == {"train", "val"}, lesion_counts
assert all(int(row["n_voxels"]) > 0 for row in lesions)
print(f"dataset_contract=PASS cases={dict(case_counts)} lesions={dict(lesion_counts)}")
PY

printf 'job_id=%s\ncreated_at=%s\ncsv=%s\n' \
  "${SLURM_JOB_ID}" "$(date -Is)" "${CSV_PATH}" > "${PROD_ROOT}/PREPARED.ok"
echo "PREPARE_DATASET_PASS"
