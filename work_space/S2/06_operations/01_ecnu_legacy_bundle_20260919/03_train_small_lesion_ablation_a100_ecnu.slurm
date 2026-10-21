#!/bin/bash
#SBATCH --job-name=s2_small_ablation
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=work_space/S2/logs/s2_small_ablation_%x_%j.out
#SBATCH --error=work_space/S2/logs/s2_small_ablation_%x_%j.err
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/${USER}/projects/ECNU_EYU_data}"
S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
S2_STORAGE_ROOT="${S2_STORAGE_ROOT:-${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency}"
S2_HPC_USER_ROOT="${S2_HPC_USER_ROOT:-/public/home/${USER}}"
STORAGE_POLICY="${PROJECT_ROOT}/work_space/S2/slurm/s2_storage_policy.sh"
BASELINE_CHECKPOINT="${S2_PRETRAINED_WEIGHTS:-${PROJECT_ROOT}/work_space/S2/results/checkpoint_final.pth}"
DATASET_NAME=Dataset264_BraTS2026_MET_Completion
S2_ABLATION_CANDIDATE="${S2_ABLATION_CANDIDATE:?Set S2_ABLATION_CANDIDATE to a1, focal, or a1_focal}"

case "${S2_ABLATION_CANDIDATE}" in
    a1)
        S2_TRAINER=nnUNetTrainerBraTS2026RCA1CompletionFineTune
        ;;
    focal)
        S2_TRAINER=nnUNetTrainerBraTS2026RCFocalCompletionFineTune
        ;;
    a1_focal)
        S2_TRAINER=nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune
        ;;
    *)
        echo "Unsupported S2_ABLATION_CANDIDATE: ${S2_ABLATION_CANDIDATE}" >&2
        exit 2
        ;;
esac

[[ -f "${STORAGE_POLICY}" ]] || { echo "Missing storage policy: ${STORAGE_POLICY}" >&2; exit 1; }
source "${STORAGE_POLICY}"
export S2_ALLOW_PUBLIC_EMERGENCY="${S2_ALLOW_PUBLIC_EMERGENCY:-1}"
export S2_CLEAN_PUBLIC_AFTER_SUCCESS=0
s2_configure_storage_policy
trap s2_stop_storage_watchdog EXIT
trap 's2_stop_storage_watchdog; exit 130' INT
trap 's2_stop_storage_watchdog; exit 143' TERM
s2_require_public_emergency_marker
s2_start_storage_watchdog

export nnUNet_raw="${S2_STORAGE_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${S2_STORAGE_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${S2_STORAGE_ROOT}/nnUNet_results"
export BRATS_SPLIT_DIR="${S2_REPO}/data/splits/completion_warmstart"
export NNUNET_DATASET_DIR="${nnUNet_raw}/${DATASET_NAME}"
export S2_EXPERIMENT_MODE=completion_warmstart
export S2_TRAINER
export S2_PRETRAINED_WEIGHTS="${BASELINE_CHECKPOINT}"
export S2_COMPLETION_EPOCHS="${S2_COMPLETION_EPOCHS:-200}"
export S2_COMPLETION_INITIAL_LR="${S2_COMPLETION_INITIAL_LR:-0.001}"
export S2_COMPLETION_SAVE_EVERY="${S2_COMPLETION_SAVE_EVERY:-25}"
export S2_FOCAL_GAMMA="${S2_FOCAL_GAMMA:-2.0}"
export S2_CONTINUE="${S2_CONTINUE:-auto}"
export S2_SKIP_COMPLETED="${S2_SKIP_COMPLETED:-1}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-4}"
export nnUNet_compile="${nnUNet_compile:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PREPROCESSED="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_3d_fullres"
[[ -f "${BASELINE_CHECKPOINT}" ]] || { echo "Missing baseline checkpoint: ${BASELINE_CHECKPOINT}" >&2; exit 1; }
[[ -d "${PREPROCESSED}" ]] || { echo "Missing Dataset264 preprocessing: ${PREPROCESSED}" >&2; exit 1; }
[[ -s "${BRATS_SPLIT_DIR}/train_fixed.txt" ]] || { echo "Missing Dataset264 train split" >&2; exit 1; }
[[ -s "${BRATS_SPLIT_DIR}/val_fixed.txt" ]] || { echo "Missing Dataset264 validation split" >&2; exit 1; }

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${S2_CONDA_ENV:-segmamba}"

# Fail before invoking nnU-Net if the active environment drifted from the
# declared runtime contract. This catches missing transitive nnU-Net packages.
python -m pip check
python - <<'PY'
import importlib.metadata as md

required = {
    "nnunetv2": "2.8.0",
    "batchgeneratorsv2": "0.3.3",
    "blosc2": "3.3.2",
    "graphviz": "0.21",
    "imagecodecs": "2025.3.30",
    "seaborn": "0.13.2",
    "yacs": "0.1.8",
    "timm": "1.0.22",
    "ninja": "1.13.0",
}
for package, expected in required.items():
    actual = md.version(package)
    if actual != expected:
        raise SystemExit(f"{package} expected {expected}, got {actual}")
print("S2_RUNTIME_DEPENDENCY_PASS")
PY

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
props = torch.cuda.get_device_properties(0)
memory_gib = props.total_memory / 1024**3
print("gpu:", props.name, f"{memory_gib:.1f} GiB")
if memory_gib < 30:
    raise SystemExit(f"S2 ablation requires at least 30 GiB VRAM, found {memory_gib:.1f}")
PY

echo "S2_ABLATION_START candidate=${S2_ABLATION_CANDIDATE} trainer=${S2_TRAINER}"
cd "${S2_REPO}"
bash train.sh

RESULT_FOLD="${nnUNet_results}/${DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__3d_fullres/fold_0"
[[ -s "${RESULT_FOLD}/checkpoint_final.pth" ]] || { echo "Missing final checkpoint: ${RESULT_FOLD}" >&2; exit 1; }
[[ -s "${RESULT_FOLD}/validation/summary.json" ]] || { echo "Missing validation summary: ${RESULT_FOLD}" >&2; exit 1; }
PREDICTION_COUNT=$(find "${RESULT_FOLD}/validation" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
[[ "${PREDICTION_COUNT}" == "103" ]] || { echo "Incomplete validation predictions: ${PREDICTION_COUNT}/103" >&2; exit 1; }
sha256sum "${RESULT_FOLD}/checkpoint_final.pth" > "${RESULT_FOLD}/checkpoint_final.sha256"
printf 'candidate=%s\ntrainer=%s\nvalidation_predictions=%s\n' \
    "${S2_ABLATION_CANDIDATE}" "${S2_TRAINER}" "${PREDICTION_COUNT}" \
    > "${RESULT_FOLD}/S2_ABLATION_COMPLETE.ok"
s2_check_active_storage_free_space
echo "S2_ABLATION_PASS candidate=${S2_ABLATION_CANDIDATE} result=${RESULT_FOLD}"
