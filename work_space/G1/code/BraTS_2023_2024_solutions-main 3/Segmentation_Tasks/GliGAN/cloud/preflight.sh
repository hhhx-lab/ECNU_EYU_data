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
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/root/brats2026/envs/g1_diffusion_v3}"
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-/root/brats2026/preflight/$(hostname)}"
PREFLIGHT_BATCH_SIZE="${PREFLIGHT_BATCH_SIZE:-8}"
USE_COMPILE="${USE_COMPILE:-1}"

rm -rf "${PREFLIGHT_ROOT}"
mkdir -p "${PREFLIGHT_ROOT}/checkpoints"
cd "${CODE_DIR}"

"${PYTHON_BIN}" -c 'import monai, nibabel, torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
"${PYTHON_BIN}" tests/test_axis_contract.py
"${PYTHON_BIN}" tests/create_smoke_fixture.py --output-root "${PREFLIGHT_ROOT}/fixture"
"${PYTHON_BIN}" src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir "${PREFLIGHT_ROOT}/fixture/DataSet" \
  --split_manifest "${PREFLIGHT_ROOT}/fixture/fixture_split_manifest.csv" \
  --csv_path "${PREFLIGHT_ROOT}/lesions.csv" \
  --checkpoint_root "${PREFLIGHT_ROOT}/checkpoints" \
  --logdir crop64_preflight \
  --crop_size 64 \
  --merge_dist 16 \
  --debug False
"${PYTHON_BIN}" scripts/expand_preflight_csv.py \
  --csv "${PREFLIGHT_ROOT}/lesions.csv" \
  --batch-size "${PREFLIGHT_BATCH_SIZE}"

TRAIN_CMD=(
  "${PYTHON_BIN}" src/train/tumour_main_diffusion.py
  --dataset BRATS_2024
  --modality t1c
  --split train
  --csv_path "${PREFLIGHT_ROOT}/lesions.csv"
  --checkpoint_root "${PREFLIGHT_ROOT}/checkpoints"
  --logdir crop64_preflight
  --batch_size "${PREFLIGHT_BATCH_SIZE}"
  --loader_workers 0
  --generator_type Unet_NnU
  --network_channels 48,96,192,384
  --network_strides 2,2,2
  --crop_size 64
  --small_lesion_weight 3.0
  --patient_balance_mode sqrt
  --num_steps 1
  --checkpoint_interval 1
  --keep_last_checkpoints 1
  --n_steps 1000
  --noise_schedule edm
  --normalization zscore
  --sigma_data 1.0
)
if [[ "${USE_COMPILE}" == "1" ]]; then
  TRAIN_CMD+=(--use_compile)
fi
"${TRAIN_CMD[@]}"

test -s "${PREFLIGHT_ROOT}/checkpoints/crop64_preflight/t1c/weights/diffusion_1.pt"
echo "CLOUD_PREFLIGHT_PASS batch=${PREFLIGHT_BATCH_SIZE} compile=${USE_COMPILE}"
