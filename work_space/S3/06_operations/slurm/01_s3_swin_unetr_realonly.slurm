#!/bin/bash
#SBATCH --job-name=s3_swin_realonly
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=logs/s3_swin_realonly_%j.out
#SBATCH --error=logs/s3_swin_realonly_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -euo pipefail

# Submit from work_space/S3:
#   mkdir -p logs
#   sbatch slurm/01_s3_swin_unetr_realonly.slurm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
export PROJ

S3_ROOT="${PROJ}/work_space/S3"
CODE_DIR="${S3_ROOT}/code"
RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJ}/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training}"
G2_RESULTS="${PROJ}/work_space/G2/results"
G2_RAW_INTAKE_SCRIPT="${PROJ}/work_space/G2/code/g2_build_realonly_from_raw.py"
G2_MAPPING_CSV="${G2_MAPPING_CSV:-${PROJ}/work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv}"
G2_SPLIT_JSON="${G2_SPLIT_JSON:-${PROJ}/work_space/G2/results/splits/splits_final_train_val_test.json}"
SPLIT_JSON="${SPLIT_JSON:-${S3_ROOT}/data/splits/s3_g2_fixed_split.json}"
LOGDIR="${LOGDIR:-${S3_ROOT}/results/realonly/swin_unetr}"
G2_REFRESH_REALONLY="${G2_REFRESH_REALONLY:-1}"

CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2026_s3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

S3_MAX_EPOCHS="${S3_MAX_EPOCHS:-200}"
S3_BATCH_SIZE="${S3_BATCH_SIZE:-1}"
S3_VAL_EVERY="${S3_VAL_EVERY:-10}"
S3_LR="${S3_LR:-1e-4}"
S3_WEIGHT_DECAY="${S3_WEIGHT_DECAY:-1e-5}"
S3_WORKERS="${S3_WORKERS:-8}"
S3_ROI_X="${S3_ROI_X:-96}"
S3_ROI_Y="${S3_ROI_Y:-96}"
S3_ROI_Z="${S3_ROI_Z:-96}"
S3_FEATURE_SIZE="${S3_FEATURE_SIZE:-48}"
S3_USE_CHECKPOINT="${S3_USE_CHECKPOINT:-true}"

mkdir -p "${S3_ROOT}/data/splits" "${LOGDIR}" "${S3_ROOT}/logs"

echo "===== S3 Swin UNETR real-only job ====="
echo "Job ID        : ${SLURM_JOB_ID:-local}"
echo "Node          : ${SLURMD_NODENAME:-unknown}"
echo "Project       : ${PROJ}"
echo "Code dir      : ${CODE_DIR}"
echo "Raw data dir  : ${RAW_DATA_DIR}"
echo "G2 mapping    : ${G2_MAPPING_CSV}"
echo "G2 split      : ${G2_SPLIT_JSON}"
echo "S3 split JSON : ${SPLIT_JSON}"
echo "Logdir        : ${LOGDIR}"
echo "======================================="

if [[ ! -d "${CODE_DIR}" ]]; then
    echo "Missing S3 code directory: ${CODE_DIR}" >&2
    exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" ]]; then
    echo "Missing raw data directory: ${RAW_DATA_DIR}" >&2
    exit 1
fi
if [[ ! -f "${G2_RAW_INTAKE_SCRIPT}" ]]; then
    echo "Missing G2 raw intake script: ${G2_RAW_INTAKE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_SH}" >&2
    echo "Submit with --export=ALL,CONDA_SH=/path/to/conda.sh if needed." >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${CODE_DIR}"

"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

if [[ "${G2_REFRESH_REALONLY}" == "1" || ! -f "${G2_MAPPING_CSV}" || ! -f "${G2_SPLIT_JSON}" ]]; then
    echo "===== Refresh G2 real-only mapping/split from raw data ====="
    "${PYTHON_BIN}" "${G2_RAW_INTAKE_SCRIPT}" \
        --project-root "${PROJ}" \
        --data-root "${RAW_DATA_DIR}" \
        --results-root "${G2_RESULTS}" \
        --fail-if-no-valid-cases
fi
if [[ ! -f "${G2_MAPPING_CSV}" ]]; then
    echo "Missing G2 mapping CSV after raw intake: ${G2_MAPPING_CSV}" >&2
    exit 1
fi
if [[ ! -f "${G2_SPLIT_JSON}" ]]; then
    echo "Missing G2 split JSON after raw intake: ${G2_SPLIT_JSON}" >&2
    exit 1
fi

echo "===== Build S3 split from G2 fixed split ====="
"${PYTHON_BIN}" make_split.py \
    --data-dir "${RAW_DATA_DIR}" \
    --g2-split-json "${G2_SPLIT_JSON}" \
    --g2-mapping-csv "${G2_MAPPING_CSV}" \
    --project-root "${PROJ}" \
    --output "${SPLIT_JSON}"

TRAIN_ARGS=(
    --data_dir "${RAW_DATA_DIR}"
    --json_list "${SPLIT_JSON}"
    --logdir "${LOGDIR}"
    --max_epochs "${S3_MAX_EPOCHS}"
    --batch_size "${S3_BATCH_SIZE}"
    --val_every "${S3_VAL_EVERY}"
    --optim_lr "${S3_LR}"
    --reg_weight "${S3_WEIGHT_DECAY}"
    --workers "${S3_WORKERS}"
    --roi_x "${S3_ROI_X}"
    --roi_y "${S3_ROI_Y}"
    --roi_z "${S3_ROI_Z}"
    --feature_size "${S3_FEATURE_SIZE}"
    --out_channels 5
    --in_channels 4
    --gpu 0
)

if [[ "${S3_USE_CHECKPOINT}" == "true" || "${S3_USE_CHECKPOINT}" == "1" ]]; then
    TRAIN_ARGS+=(--use_checkpoint)
fi

echo "===== Train S3 Swin UNETR real-only ====="
"${PYTHON_BIN}" main.py "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOGDIR}/train-${SLURM_JOB_ID:-local}.log"

echo "===== DONE: $(date) ====="
echo "Split JSON : ${SPLIT_JSON}"
echo "Logdir     : ${LOGDIR}"
echo "Best model : ${LOGDIR}/best_model_dice.pth"
