#!/bin/bash
#SBATCH -J g1d3_smoke
#SBATCH -p v100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /public/home/%u/g1_diffusion_v3_smoke_20260715/logs/smoke_%j.out
#SBATCH -e /public/home/%u/g1_diffusion_v3_smoke_20260715/logs/smoke_%j.err

set -Eeuo pipefail

SMOKE_ROOT=${SMOKE_ROOT:-/public/home/${USER}/g1_diffusion_v3_smoke_20260715}
CODE_DIR=${CODE_DIR:-${SLURM_SUBMIT_DIR:-"${SMOKE_ROOT}/code/Segmentation_Tasks/GliGAN"}}
PYTHON_BIN=${PYTHON_BIN:-/public/home/${USER}/.conda/envs/segmamba/bin/python}
RUN_ROOT="${SMOKE_ROOT}/runs/${SLURM_JOB_ID}"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
LOGDIR=axis_edm_smoke

mkdir -p "${RUN_ROOT}" "${CHECKPOINT_ROOT}" "${SMOKE_ROOT}/logs"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${RUN_ROOT}/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "code_dir=${CODE_DIR}"
echo "run_root=${RUN_ROOT}"
test -f "${CODE_DIR}/tests/create_smoke_fixture.py"
nvidia-smi
"${PYTHON_BIN}" -c 'import torch, monai, nibabel; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0)); print("monai", monai.__version__, "nibabel", nibabel.__version__)'

cd "${CODE_DIR}"

"${PYTHON_BIN}" tests/create_smoke_fixture.py --output-root "${RUN_ROOT}/fixture"

"${PYTHON_BIN}" src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${RUN_ROOT}/fixture/DataSet" \
  --split_manifest "${RUN_ROOT}/fixture/fixture_split_manifest.csv" \
  --csv_path "${RUN_ROOT}/lesions.csv" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --logdir "${LOGDIR}" \
  --crop_size 16 \
  --merge_dist 4 \
  --debug False

"${PYTHON_BIN}" - "${RUN_ROOT}/lesions.csv" <<'PY'
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

"${PYTHON_BIN}" src/train/tumour_main_diffusion.py \
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

"${PYTHON_BIN}" src/infer/evaluate_generation.py \
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
"${PYTHON_BIN}" src/infer/generate_from_label.py \
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

"${PYTHON_BIN}" src/infer/on_the_fly_augmentation.py \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --label_pool_csv "${RUN_ROOT}/lesions.csv" \
  --dataset BRATS_2024 \
  --sampling_method edm_heun \
  --sampling_steps 2 \
  --generator_type Unet_NnU \
  --normalization zscore \
  --crop_size 16 \
  --allow_partial

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
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
# Validation lesion is x=4:7, y=18:21, z=14:17. With the 4-voxel
# inference margin, generation must stay inside this native NIfTI window.
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
