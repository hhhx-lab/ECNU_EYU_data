#!/bin/bash
#SBATCH -J g1d3_pre64
#SBATCH -p v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /public/home/%u/g1_diffusion_v3_smoke_20260715/logs/preflight64_%j.out
#SBATCH -e /public/home/%u/g1_diffusion_v3_smoke_20260715/logs/preflight64_%j.err

set -Eeuo pipefail

SMOKE_ROOT=${SMOKE_ROOT:-/public/home/${USER}/g1_diffusion_v3_smoke_20260715}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-"${SMOKE_ROOT}/code/Segmentation_Tasks/GliGAN"}}
PYTHON_BIN=${PYTHON_BIN:-/public/home/${USER}/.conda/envs/segmamba/bin/python}
RUN_ROOT="${SMOKE_ROOT}/runs/preflight64_${SLURM_JOB_ID}"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
LOGDIR=crop64_batch4_eager

mkdir -p "${RUN_ROOT}" "${CHECKPOINT_ROOT}" "${SMOKE_ROOT}/logs"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib"

test -f "${CODE_DIR}/tests/create_smoke_fixture.py"
cd "${CODE_DIR}"
"${PYTHON_BIN}" tests/create_smoke_fixture.py --output-root "${RUN_ROOT}/fixture"
"${PYTHON_BIN}" src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${RUN_ROOT}/fixture/DataSet" \
  --split_manifest "${RUN_ROOT}/fixture/fixture_split_manifest.csv" \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --crop_size 64 \
  --merge_dist 16 \
  --debug False

"${PYTHON_BIN}" - "${RUN_ROOT}/lesions.csv" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = reader.fieldnames
assert fieldnames
train = next(row for row in rows if row["split"] == "train")
val = next(row for row in rows if row["split"] == "val")
expanded = []
for index in range(4):
    row = dict(train)
    row["lesion_id"] = f'{train["lesion_id"]}_repeat{index}'
    expanded.append(row)
expanded.append(val)
with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(expanded)
print("preflight_train_rows=4")
PY

nvidia-smi
"${PYTHON_BIN}" src/train/tumour_main_diffusion.py \
  --dataset BRATS_2024 \
  --modality t1c \
  --split train \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --batch_size 4 \
  --loader_workers 0 \
  --generator_type Unet_NnU \
  --network_channels 48,96,192,384 \
  --network_strides 2,2,2 \
  --crop_size 64 \
  --small_lesion_weight 3.0 \
  --patient_balance_mode sqrt \
  --num_steps 1 \
  --noise_schedule edm \
  --normalization zscore \
  --sigma_data 1.0

test -s "${CHECKPOINT_ROOT}/${LOGDIR}/t1c/weights/diffusion_1.pt"
echo "CROP64_BATCH4_EAGER_PASS run_root=${RUN_ROOT}"
