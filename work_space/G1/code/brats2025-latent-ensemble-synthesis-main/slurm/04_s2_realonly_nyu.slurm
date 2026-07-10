#!/bin/bash
#SBATCH --job-name=s2_ro_cv
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --array=0-4%5
#SBATCH --output=logs/s2_realonly_%A_%a.out
#SBATCH --error=logs/s2_realonly_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu
#SBATCH --requeue

# S2 real-only five-fold training array. Run the preparation job first.
# Submit --array=1-4%4 to continue after the completed fold 0 baseline.

set -e

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
S2_REPO="${PROJ}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2026_s2}"

export nnUNet_raw="${nnUNet_raw:-${PROJ}/work_space/S2/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJ}/work_space/S2/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJ}/work_space/S2/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S2_REPO}/data/splits}"
export BRATS_S2_REPO_DIR="${S2_REPO}"
export S2_DATASET_ID="${S2_DATASET_ID:-260}"
export S2_DATASET_NAME="${S2_DATASET_NAME:-Dataset260_BraTS2026_MET_RealOnly}"
export S2_FOLD="${SLURM_ARRAY_TASK_ID:-${S2_FOLD:-0}}"
export S2_CONTINUE="${S2_CONTINUE:-auto}"
export S2_SKIP_COMPLETED="${S2_SKIP_COMPLETED:-1}"

echo "===== S2 Five-Fold Job ====="
echo "Array job : ${SLURM_ARRAY_JOB_ID:-local}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-local}"
echo "Fold      : ${S2_FOLD}"
echo "Node      : ${SLURMD_NODENAME:-unknown}"
echo "Date      : $(date)"
echo "Project   : ${PROJ}"
echo "Results   : ${nnUNet_results}"
echo "============================"

if [[ ! "${S2_FOLD}" =~ ^[0-4]$ ]]; then
    echo "Fold must be one integer from 0 to 4, got: ${S2_FOLD}" >&2
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
if [[ ! -s "${BRATS_SPLIT_DIR}/train_fold${S2_FOLD}.txt" ]]; then
    echo "Missing train split for fold ${S2_FOLD}; run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi
if [[ ! -s "${BRATS_SPLIT_DIR}/val_fold${S2_FOLD}.txt" ]]; then
    echo "Missing validation split for fold ${S2_FOLD}; run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi

PREPROCESSED_DATASET="${nnUNet_preprocessed}/${S2_DATASET_NAME}"
if [[ ! -f "${PREPROCESSED_DATASET}/nnUNetPlans.json" ]]; then
    echo "Missing nnUNetPlans.json: ${PREPROCESSED_DATASET}/nnUNetPlans.json" >&2
    echo "Run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi
if [[ ! -d "${PREPROCESSED_DATASET}/nnUNetPlans_3d_fullres" ]]; then
    echo "Missing 3d_fullres preprocessed data: ${PREPROCESSED_DATASET}/nnUNetPlans_3d_fullres" >&2
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

echo "===== DONE: fold ${S2_FOLD}, $(date) ====="
