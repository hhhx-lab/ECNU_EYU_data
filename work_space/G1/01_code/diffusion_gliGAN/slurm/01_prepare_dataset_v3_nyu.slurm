#!/bin/bash
#SBATCH --job-name=g1_diffv3_prep
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_prep_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_prep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-g1_diffusion_v3}
RUN_ROOT=${RUN_ROOT:-"${PROJ}/runs/g1_diffusion_v3"}
MANIFEST=${MANIFEST:-"${PROJ}/work_space/G2/results/manifests/g1_v2_source_manifest.csv"}
DATASET_DIR=${DATASET_DIR:-"${RUN_ROOT}/DataSet"}
SPLIT_DIR=${SPLIT_DIR:-"${RUN_ROOT}/splits/current"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${RUN_ROOT}/checkpoints"}
LOGDIR=${LOGDIR:-brats2026_diffusion_v3_edm_zscore}
CSV_PATH=${CSV_PATH:-"${SPLIT_DIR}/lesions.csv"}
EXPECTED_TRAIN=${EXPECTED_TRAIN:-823}
EXPECTED_VAL=${EXPECTED_VAL:-103}
# Optional flattened layout override. Default uses relative paths inside MANIFEST.
RAW_ROOT=${RAW_ROOT:-}

echo "===== Job Info ====="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : ${SLURMD_NODENAME:-unknown}"
echo "Date      : $(date)"
echo "Code      : ${CODE_DIR}"
echo "Run root  : ${RUN_ROOT}"
echo "Manifest  : ${MANIFEST}"
echo "Dataset   : ${DATASET_DIR}"
echo "===================="

mkdir -p "${PROJ}/logs" "${RUN_ROOT}" "${SPLIT_DIR}" "${CHECKPOINT_ROOT}"
rm -f "${RUN_ROOT}/PREPARED.ok"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
echo "Python : $(which python)"
python -c 'import nibabel, numpy, scipy; print("python_dependencies=PASS")'
python tests/test_axis_contract.py

PREPARE_ARGS=(
  --project-root "${PROJ}"
  --source-manifest "${MANIFEST}"
  --output-dir "${DATASET_DIR}"
  --split-dir "${SPLIT_DIR}"
)
if [[ -n "${RAW_ROOT}" ]]; then
  PREPARE_ARGS+=(--raw-root "${RAW_ROOT}")
fi
if [[ -d "${DATASET_DIR}" && -f "${DATASET_DIR}/.g1_diffusion_dataset" ]]; then
  PREPARE_ARGS+=(--clean)
fi
python scripts/prepare_dataset_from_g2_manifest.py "${PREPARE_ARGS[@]}"

python src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${DATASET_DIR}" \
  --split_manifest "${MANIFEST}" \
  --csv_path "${CSV_PATH}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --crop_size 64 \
  --merge_dist 16 \
  --debug False

python - "${SPLIT_DIR}/dataset_membership.csv" "${CSV_PATH}" \
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

printf 'job_id=%s\ncreated_at=%s\ncsv=%s\ndataset=%s\n' \
  "${SLURM_JOB_ID}" "$(date -Is)" "${CSV_PATH}" "${DATASET_DIR}" > "${RUN_ROOT}/PREPARED.ok"
echo "PREPARE_DATASET_PASS"
echo "===== DONE: $(date) ====="
