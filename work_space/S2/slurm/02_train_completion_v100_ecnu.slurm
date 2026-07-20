#!/bin/bash
#SBATCH --job-name=s2_comp_v100
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=work_space/S2/logs/s2_completion_v100_%j.out
#SBATCH --error=work_space/S2/logs/s2_completion_v100_%j.err
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/${USER}/projects/ECNU_EYU_data}"
S2_HPC_USER_ROOT="${S2_HPC_USER_ROOT:-/hpc_stor/data/${USER}}"
S2_STORAGE_ROOT="${S2_STORAGE_ROOT:-${S2_HPC_USER_ROOT}/ECNU_EYU_data/work_space/S2/data/ecnu_completion}"
STORAGE_POLICY="${PROJECT_ROOT}/work_space/S2/slurm/s2_storage_policy.sh"
S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
BASELINE_CHECKPOINT="${S2_PRETRAINED_WEIGHTS:-${PROJECT_ROOT}/work_space/S2/results/checkpoint_final.pth}"

[[ -f "${STORAGE_POLICY}" ]] || { echo "Missing storage policy: ${STORAGE_POLICY}" >&2; exit 1; }
source "${STORAGE_POLICY}"
s2_configure_storage_policy
trap s2_stop_storage_watchdog EXIT
trap 's2_stop_storage_watchdog; exit 130' INT
trap 's2_stop_storage_watchdog; exit 143' TERM

export nnUNet_raw="${S2_STORAGE_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${S2_STORAGE_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${S2_STORAGE_ROOT}/nnUNet_results"
export BRATS_SPLIT_DIR="${S2_REPO}/data/splits/completion_warmstart"
export S2_EXPERIMENT_MODE=completion_warmstart
export S2_PRETRAINED_WEIGHTS="${BASELINE_CHECKPOINT}"
export S2_COMPLETION_EPOCHS="${S2_COMPLETION_EPOCHS:-200}"
export S2_COMPLETION_INITIAL_LR="${S2_COMPLETION_INITIAL_LR:-0.001}"
export S2_COMPLETION_SAVE_EVERY="${S2_COMPLETION_SAVE_EVERY:-25}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export nnUNet_compile="${nnUNet_compile:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

DATASET_NAME=Dataset264_BraTS2026_MET_Completion
PREPROCESSED="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_3d_fullres"
[[ -f "${BASELINE_CHECKPOINT}" ]] || { echo "Missing baseline checkpoint: ${BASELINE_CHECKPOINT}" >&2; exit 1; }
[[ -d "${PREPROCESSED}" ]] || { echo "Missing Dataset264 preprocessing: ${PREPROCESSED}" >&2; exit 1; }
[[ -s "${BRATS_SPLIT_DIR}/train_fixed.txt" ]] || { echo "Missing Dataset264 train split" >&2; exit 1; }
[[ -s "${BRATS_SPLIT_DIR}/val_fixed.txt" ]] || { echo "Missing Dataset264 val split" >&2; exit 1; }
s2_require_public_emergency_marker
s2_start_storage_watchdog

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${S2_CONDA_ENV:-segmamba}"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
props = torch.cuda.get_device_properties(0)
memory_gib = props.total_memory / 1024**3
print("gpu:", props.name, f"{memory_gib:.1f} GiB")
if memory_gib < 30:
    raise SystemExit(f"Dataset264 requires at least 30 GiB VRAM, found {memory_gib:.1f}")
PY

cd "${S2_REPO}"
bash train.sh

RESULT_FOLD="${nnUNet_results}/${DATASET_NAME}/nnUNetTrainerBraTS2026RCCompletionFineTune__nnUNetPlans__3d_fullres/fold_0"
[[ -f "${RESULT_FOLD}/checkpoint_final.pth" ]] || { echo "Missing final checkpoint: ${RESULT_FOLD}" >&2; exit 1; }
[[ -f "${RESULT_FOLD}/validation/summary.json" ]] || { echo "Missing validation summary: ${RESULT_FOLD}" >&2; exit 1; }

PUBLISH_DIR="${S2_PUBLISH_DIR:-${S2_STORAGE_ROOT}/published/completion_only_ecnu}"
mkdir -p "${PUBLISH_DIR}"
cp -f "${RESULT_FOLD}/checkpoint_final.pth" "${PUBLISH_DIR}/checkpoint_final.pth"
cp -f "${RESULT_FOLD}/checkpoint_best.pth" "${PUBLISH_DIR}/checkpoint_best.pth"
cp -f "${RESULT_FOLD}/validation/summary.json" "${PUBLISH_DIR}/validation_summary.json"
cp -f "${RESULT_FOLD}/debug.json" "${PUBLISH_DIR}/debug.json"

s2_check_active_storage_free_space
echo "S2_COMPLETION_TRAIN_PASS result=${RESULT_FOLD} published=${PUBLISH_DIR} mode=${S2_STORAGE_MODE}"
s2_stop_storage_watchdog
s2_cleanup_public_emergency_regenerable_data
