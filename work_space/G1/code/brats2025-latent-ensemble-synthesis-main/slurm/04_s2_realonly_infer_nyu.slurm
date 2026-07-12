#!/bin/bash
#SBATCH --job-name=s2_ro_infer
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/s2_realonly_infer_%j.out
#SBATCH --error=logs/s2_realonly_infer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
S2_REPO="${PROJ}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2026_s2}"
S2_EXPERIMENT_MODE="${S2_EXPERIMENT_MODE:-current}"
case "${S2_EXPERIMENT_MODE}" in
    current)
        DEFAULT_S2_DATASET_ID=263
        DEFAULT_S2_DATASET_NAME=Dataset263_BraTS2026_MET_RealOnly_Current
        DEFAULT_S2_OUTPUT_ID=realonly_current_fixed_inference
        ;;
    legacy)
        DEFAULT_S2_DATASET_ID=260
        DEFAULT_S2_DATASET_NAME=Dataset260_BraTS2026_MET_RealOnly
        DEFAULT_S2_OUTPUT_ID=legacy_dataset260_fixed_inference
        ;;
    *)
        echo "S2_EXPERIMENT_MODE must be current or legacy, got: ${S2_EXPERIMENT_MODE}" >&2
        exit 2
        ;;
esac
S2_INFERENCE_INPUT="${S2_INFERENCE_INPUT:-}"
S2_INFERENCE_OUTPUT="${S2_INFERENCE_OUTPUT:-${PROJ}/work_space/S2/results/${DEFAULT_S2_OUTPUT_ID}}"

export nnUNet_raw="${nnUNet_raw:-${PROJ}/work_space/S2/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJ}/work_space/S2/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJ}/work_space/S2/data/nnunet_results}"
export BRATS_S2_REPO_DIR="${S2_REPO}"
export S2_DATASET_ID="${S2_DATASET_ID:-${DEFAULT_S2_DATASET_ID}}"
export S2_DATASET_NAME="${S2_DATASET_NAME:-${DEFAULT_S2_DATASET_NAME}}"
export S2_EXPERIMENT_MODE

if [[ "${S2_DATASET_ID}" != "${DEFAULT_S2_DATASET_ID}" || "${S2_DATASET_NAME}" != "${DEFAULT_S2_DATASET_NAME}" ]]; then
    echo "${S2_EXPERIMENT_MODE} mode is locked to dataset ${DEFAULT_S2_DATASET_ID}/${DEFAULT_S2_DATASET_NAME}." >&2
    exit 2
fi

if [[ -z "${S2_INFERENCE_INPUT}" ]]; then
    echo "S2_INFERENCE_INPUT is required and must point to nnU-Net four-channel input." >&2
    exit 2
fi
if [[ ! -d "${S2_INFERENCE_INPUT}" ]]; then
    echo "S2 inference input directory does not exist: ${S2_INFERENCE_INPUT}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_SH}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the S2 inference job")
print("torch:", torch.__version__)
print("gpu:", torch.cuda.get_device_name(0))
PY

cd "${S2_REPO}"
bash infer.sh "${S2_INFERENCE_INPUT}" "${S2_INFERENCE_OUTPUT}"

echo "===== S2 INFERENCE DONE: $(date) ====="
echo "Output: ${S2_INFERENCE_OUTPUT}"
