#!/bin/bash
#SBATCH --job-name=g1_diffv2_csv
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_csv_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_csv_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-brats}
LOGDIR=${LOGDIR:-brats2026_diffusion_v2_complete_only}
DATASET=${DATASET:-BRATS_2024}
CROP_SIZE=${CROP_SIZE:-64}
MERGE_DIST=${MERGE_DIST:-16}
VAL_FILE="${CODE_DIR}/splits/brats2026_v2_complete_only_val_patients_one_line.txt"

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "Date   : $(date)"
echo "Code   : ${CODE_DIR}"
echo "Logdir : ${LOGDIR}"
echo "===================="

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
echo "Python : $(which python)"

if [ ! -d DataSet ]; then
  echo "ERROR: DataSet directory not found: ${CODE_DIR}/DataSet" >&2
  exit 1
fi

VAL_PATIENTS=$(tr -d '\n\r ' < "${VAL_FILE}")
echo "Validation patient suffix count: $(tr ',' '\n' < "${VAL_FILE}" | sed '/^$/d' | wc -l | tr -d ' ')"
echo "DataSet case count: $(find DataSet -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"

python src/train/csv_creator.py \
  --dataset "${DATASET}" \
  --datadir DataSet \
  --logdir "${LOGDIR}" \
  --crop_size "${CROP_SIZE}" \
  --merge_dist "${MERGE_DIST}" \
  --val_patients "${VAL_PATIENTS}"

CSV_PATH="../../Checkpoint/${LOGDIR}/${LOGDIR}.csv"
python - "${CSV_PATH}" <<'PY'
import csv
import sys
from collections import Counter

csv_path = sys.argv[1]
with open(csv_path, newline="") as f:
    rows = list(csv.DictReader(f))
print("CSV:", csv_path)
print("Rows:", len(rows))
print("Patients:", len({r["patient_id"] for r in rows}))
print("Split rows:", dict(Counter(r["split"] for r in rows)))
print("Split patients:", {
    split: len({r["patient_id"] for r in rows if r["split"] == split})
    for split in sorted({r["split"] for r in rows})
})
PY

echo "===== DONE: $(date) ====="
