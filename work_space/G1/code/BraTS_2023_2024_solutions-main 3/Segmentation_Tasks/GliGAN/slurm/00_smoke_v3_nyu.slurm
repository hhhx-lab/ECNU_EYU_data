#!/bin/bash
#SBATCH --job-name=g1_diffv3_smoke
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_smoke_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_smoke_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-g1_diffusion_v3}
RUN_ROOT=${RUN_ROOT:-"${PROJ}/runs/g1_diffusion_v3_smoke/${SLURM_JOB_ID}"}
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
LOGDIR=axis_edm_smoke

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-allocated}"
echo "Date   : $(date)"
echo "Code   : ${CODE_DIR}"
echo "Run    : ${RUN_ROOT}"
echo "===================="

mkdir -p "${PROJ}/logs" "${RUN_ROOT}" "${CHECKPOINT_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
echo "Python : $(which python)"
python -c "import torch, monai, nibabel; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('monai', monai.__version__, 'nibabel', nibabel.__version__)"

python tests/create_smoke_fixture.py --output-root "${RUN_ROOT}/fixture"

python src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${RUN_ROOT}/fixture/DataSet" \
  --split_manifest "${RUN_ROOT}/fixture/fixture_split_manifest.csv" \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --crop_size 16 \
  --merge_dist 4 \
  --debug False

python - "${RUN_ROOT}/lesions.csv" <<'PY'
import csv
import sys
from collections import Counter

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
assert rows, "CSV has no rows"
assert Counter(row["split"] for row in rows) == {"train": 2, "val": 1}
assert all(row["patient_group"].startswith("BraTS-MET-") for row in rows)
train = next(row for row in rows if row["split"] == "train")
assert 26 <= int(train["center_x"]) <= 28, train
assert 2 <= int(train["center_z"]) <= 4, train
train_rows = [row for row in rows if row["split"] == "train"]
assert {row["patient_group"] for row in train_rows} == {"BraTS-MET-99998"}
assert all(int(row["patient_n_crops"]) == 2 for row in train_rows)
assert all(int(row["patient_n_crops"]) == 1 for row in rows if row["split"] == "val")
print("csv_contract=PASS")
PY

python src/train/tumour_main_diffusion.py \
  --dataset BRATS_2024 \
  --modality t1c \
  --split train \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --batch_size 1 \
  --loader_workers 0 \
  --generator_type Unet_NnU \
  --crop_size 16 \
  --small_lesion_weight 1.0 \
  --num_steps 1 \
  --noise_schedule edm \
  --normalization zscore \
  --sigma_data 1.0

CKPT_DIR="${CHECKPOINT_ROOT}/${LOGDIR}"
test -s "${CKPT_DIR}/t1c/weights/diffusion_1.pt"

python src/infer/evaluate_generation.py \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --dataset BRATS_2024 \
  --output_dir "${RUN_ROOT}/eval" \
  --generator_type Unet_NnU \
  --crop_size 16 \
  --normalization zscore \
  --split val \
  --modality t1c \
  --evaluation_mode patch \
  --noise_schedule edm \
  --sampling_method edm_heun \
  --sampling_steps 2 \
  --max_cases 1

VAL_CASE=BraTS-MET-99999-000
python src/infer/generate_from_label.py \
  --label_path "${RUN_ROOT}/fixture/DataSet/${VAL_CASE}/${VAL_CASE}-seg.nii.gz" \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --dataset BRATS_2024 \
  --output_dir "${RUN_ROOT}/generated" \
  --generator_type Unet_NnU \
  --crop_size 16 \
  --merge_dist 4 \
  --noise_schedule edm \
  --sampling_method edm_heun \
  --sampling_steps 2 \
  --modality t1c

python src/infer/on_the_fly_augmentation.py \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --label_pool_csv "${RUN_ROOT}/lesions.csv" \
  --dataset BRATS_2024 \
  --sampling_method edm_heun \
  --sampling_steps 2 \
  --generator_type Unet_NnU \
  --normalization zscore \
  --crop_size 16 \
  --allow_partial

python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

root = Path(sys.argv[1])
generated = root / "generated" / "BraTS-MET-99999-000-t1c.nii.gz"
metrics = root / "eval" / "metrics.json"
assert generated.is_file(), generated
assert metrics.is_file(), metrics
array = nib.load(generated).get_fdata()
assert array.shape == (32, 28, 20), array.shape
assert np.isfinite(array).all()
assert np.count_nonzero(array) > 0
expected_window = np.zeros(array.shape, dtype=bool)
expected_window[0:11, 14:25, 10:20] = True
assert np.count_nonzero(array[expected_window]) > 0
assert np.count_nonzero(array[~expected_window]) == 0
payload = json.loads(metrics.read_text())
assert payload.get("summary", {}).get("t1c"), payload
checkpoint = torch.load(
    root / "checkpoints" / "axis_edm_smoke" / "t1c" / "weights" / "diffusion_1.pt",
    map_location="cpu",
)
for key in (
    "normalization", "crop_size", "generator_type", "noise_schedule", "dataset",
    "network_channels", "network_strides",
):
    assert key in checkpoint, key
assert checkpoint["network_channels"]
assert len(checkpoint["network_strides"]) == len(checkpoint["network_channels"]) - 1
print("gpu_pipeline_contract=PASS")
PY

echo "SMOKE_TEST_PASS run_root=${RUN_ROOT}"
echo "===== DONE: $(date) ====="
