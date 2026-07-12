#!/bin/bash
#SBATCH --job-name=s2_ro_fixed
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=logs/s2_realonly_fixed_%j.out
#SBATCH --error=logs/s2_realonly_fixed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu
#SBATCH --requeue

# S2 real-only fixed-split training. Cross-validation is intentionally disabled.

set -e

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
S2_REPO="${PROJ}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2026_s2}"
S2_EXPERIMENT_MODE="${S2_EXPERIMENT_MODE:-current}"
export S2_EXPERIMENT_MODE

case "${S2_EXPERIMENT_MODE}" in
    current)
        DEFAULT_S2_DATASET_ID=263
        DEFAULT_S2_DATASET_NAME=Dataset263_BraTS2026_MET_RealOnly_Current
        ;;
    legacy)
        DEFAULT_S2_DATASET_ID=260
        DEFAULT_S2_DATASET_NAME=Dataset260_BraTS2026_MET_RealOnly
        ;;
    *)
        echo "S2_EXPERIMENT_MODE must be current or legacy, got: ${S2_EXPERIMENT_MODE}" >&2
        exit 2
        ;;
esac

export nnUNet_raw="${nnUNet_raw:-${PROJ}/work_space/S2/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJ}/work_space/S2/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJ}/work_space/S2/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S2_REPO}/data/splits/${S2_EXPERIMENT_MODE}}"
export BRATS_S2_REPO_DIR="${S2_REPO}"
export S2_DATASET_ID="${S2_DATASET_ID:-${DEFAULT_S2_DATASET_ID}}"
export S2_DATASET_NAME="${S2_DATASET_NAME:-${DEFAULT_S2_DATASET_NAME}}"
export NNUNET_DATASET_DIR="${NNUNET_DATASET_DIR:-${nnUNet_raw}/${S2_DATASET_NAME}}"
export S2_PREPROCESSED_DATA_IDENTIFIER="${S2_PREPROCESSED_DATA_IDENTIFIER:-nnUNetPlans_3d_fullres}"
export S2_CONTINUE="${S2_CONTINUE:-auto}"
export S2_SKIP_COMPLETED="${S2_SKIP_COMPLETED:-1}"

if [[ "${S2_DATASET_ID}" != "${DEFAULT_S2_DATASET_ID}" || "${S2_DATASET_NAME}" != "${DEFAULT_S2_DATASET_NAME}" ]]; then
    echo "${S2_EXPERIMENT_MODE} mode is locked to dataset ${DEFAULT_S2_DATASET_ID}/${DEFAULT_S2_DATASET_NAME}." >&2
    exit 2
fi

echo "===== S2 Fixed-Split Job ====="
echo "Job ID    : ${SLURM_JOB_ID:-local}"
echo "Mode      : ${S2_EXPERIMENT_MODE}"
echo "Split     : fixed train/validation"
echo "nnU-Net key: fold_0 (storage/API compatibility only)"
echo "Node      : ${SLURMD_NODENAME:-unknown}"
echo "Date      : $(date)"
echo "Project   : ${PROJ}"
echo "Results   : ${nnUNet_results}"
echo "============================"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "S2 cross-validation is disabled; do not submit this script with --array." >&2
    exit 2
fi
if [[ ! -d "${S2_REPO}" ]]; then
    echo "Missing S2 repository: ${S2_REPO}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_SH}" >&2
    exit 1
fi
if [[ ! -s "${BRATS_SPLIT_DIR}/train_fixed.txt" ]]; then
    echo "Missing fixed training split; run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi
if [[ ! -s "${BRATS_SPLIT_DIR}/val_fixed.txt" ]]; then
    echo "Missing fixed validation split; run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi

PREPROCESSED_DATASET="${nnUNet_preprocessed}/${S2_DATASET_NAME}"
if [[ ! -f "${PREPROCESSED_DATASET}/nnUNetPlans.json" ]]; then
    echo "Missing nnUNetPlans.json: ${PREPROCESSED_DATASET}/nnUNetPlans.json" >&2
    echo "Run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi
if [[ ! -d "${PREPROCESSED_DATASET}/${S2_PREPROCESSED_DATA_IDENTIFIER}" ]]; then
    echo "Missing preprocessed data: ${PREPROCESSED_DATASET}/${S2_PREPROCESSED_DATA_IDENTIFIER}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the S2 GPU job")
print("gpu:", torch.cuda.get_device_name(0))
PY

cd "${S2_REPO}"
bash train.sh

echo "===== DONE: fixed split, $(date) ====="
