#!/bin/bash
#SBATCH --job-name=g1_diffv3_pre64
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_preflight64_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_preflight64_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-g1_diffusion_v3}
RUN_ROOT=${RUN_ROOT:-"${PROJ}/runs/g1_diffusion_v3_preflight/preflight64_${SLURM_JOB_ID}"}
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
LOGDIR=crop64_batch4_eager
BATCH_SIZE=${BATCH_SIZE:-4}

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-allocated}"
echo "Date   : $(date)"
echo "Code   : ${CODE_DIR}"
echo "Batch  : ${BATCH_SIZE}"
echo "===================="

mkdir -p "${PROJ}/logs" "${RUN_ROOT}" "${CHECKPOINT_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
echo "Python : $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python tests/create_smoke_fixture.py --output-root "${RUN_ROOT}/fixture"
python src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${RUN_ROOT}/fixture/DataSet" \
  --split_manifest "${RUN_ROOT}/fixture/fixture_split_manifest.csv" \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --crop_size 64 \
  --merge_dist 16 \
  --debug False

python - "${RUN_ROOT}/lesions.csv" <<'PY'
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

python src/train/tumour_main_diffusion.py \
  --dataset BRATS_2024 \
  --modality t1c \
  --split train \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --batch_size "${BATCH_SIZE}" \
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
echo "===== DONE: $(date) ====="
