#!/usr/bin/env bash
#SBATCH --job-name=s2_met_aug_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=60:00:00
#SBATCH --output=/root/brats2026/logs/%x_%j.out
#SBATCH --error=/root/brats2026/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

PAIR_ARM="${S2_PAIR_ARM:-${1:-}}"
case "${PAIR_ARM}" in
    control|route_a) ;;
    *) echo "Usage: S2_PAIR_ARM={control|route_a} $0" >&2; exit 2 ;;
esac

s2_uhost_activate_runtime
s2_uhost_require_single_visible_gpu
s2_uhost_require_contract_inputs
s2_uhost_require_free_space

COMPONENT_MANIFEST="${ROUTE_ROOT}/component_pool/component_manifest.json"
ROUTE_CONFIG="${ROUTE_ROOT}/route_a_config.json"
VALID_MASK_MANIFEST="${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json"
ROUTE_APPROVAL="${ROUTE_ROOT}/route_a_approval.json"
SMOKE_APPROVAL="${ROUTE_ROOT}/TRAINING_SMOKE_APPROVED.ok"

for required in \
    "${COMPONENT_MANIFEST}" "${ROUTE_CONFIG}" "${VALID_MASK_MANIFEST}" \
    "${ROUTE_APPROVAL}" "${SMOKE_APPROVAL}"; do
    s2_uhost_require_file "${required}" "Route A Gate 3 prerequisite"
done

export S2_MET_AUG_COMPONENT_MANIFEST="${COMPONENT_MANIFEST}"
export S2_MET_AUG_ROUTE_CONFIG="${ROUTE_CONFIG}"
export S2_MET_AUG_VALID_MASK_MANIFEST="${VALID_MASK_MANIFEST}"
export S2_MET_AUG_ROUTE_GATE="${ROUTE_APPROVAL}"
export S2_MET_AUG_G1_CODE_DIR="${G1_CODE_DIR}"
export S2_MET_AUG_G1_CHECKPOINT_ROOT="${G1_CHECKPOINT_ROOT}"
export S2_MET_AUG_G1_CHECKPOINT_SELECTION="${G1_SELECTION}"
export S2_MET_AUG_G2_QC_GATE="${G2_PARENT_GATE}"
export S2_PRETRAINED_WEIGHTS="${E_CHECKPOINT}"
export BRATS_SPLIT_DIR="${S2_REPOSITORY}/data/splits/completion_warmstart"
export S2_COMPLETION_EPOCHS=200
export S2_COMPLETION_INITIAL_LR=0.001
export S2_COMPLETION_SAVE_EVERY=25
export S2_FOCAL_GAMMA=2.0
export S2_PAIRED_TRAINING_SEED=20260724
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export S2_CONTINUE=auto
export S2_SKIP_COMPLETED=1
export nnUNet_compile=0
export nnUNet_n_proc_DA=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

case "${PAIR_ARM}" in
    control)
        export S2_EXPERIMENT_MODE=met_aug_route_a_control
        export S2_MET_AUG_ENABLE=0
        export nnUNet_results="${ROUTE_ROOT}/gate3_e_continue/nnUNet_results"
        TRAINER=nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune
        ;;
    route_a)
        export S2_EXPERIMENT_MODE=met_aug_route_a
        export S2_MET_AUG_ENABLE=1
        export nnUNet_results="${ROUTE_ROOT}/gate3_met_aug_a/nnUNet_results"
        TRAINER=nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune
        ;;
esac

mkdir -p "${nnUNet_results}" "${ROUTE_ROOT}/logs"
cd "${S2_REPOSITORY}"
echo "S2_MET_AUG_PAIR_START arm=${PAIR_ARM} gpu=${CUDA_VISIBLE_DEVICES:-unset} results=${nnUNet_results}"
bash train.sh

RESULT_FOLD="${nnUNet_results}/Dataset264_BraTS2026_MET_Completion/${TRAINER}__nnUNetPlans__3d_fullres/fold_0"
s2_uhost_require_file "${RESULT_FOLD}/checkpoint_final.pth" "${PAIR_ARM} final checkpoint"
s2_uhost_require_file "${RESULT_FOLD}/validation/summary.json" "${PAIR_ARM} validation summary"
PREDICTION_COUNT=$(find "${RESULT_FOLD}/validation" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
if [[ "${PREDICTION_COUNT}" != "103" ]]; then
    echo "Incomplete ${PAIR_ARM} validation predictions: ${PREDICTION_COUNT}/103" >&2
    exit 1
fi
CHECKPOINT_SHA=$(s2_uhost_sha256 "${RESULT_FOLD}/checkpoint_final.pth")
printf 'arm=%s\ntrainer=%s\ncheckpoint_sha256=%s\nvalidation_predictions=%s\ncompleted_at=%s\n' \
    "${PAIR_ARM}" "${TRAINER}" "${CHECKPOINT_SHA}" "${PREDICTION_COUNT}" "$(date -u +%FT%TZ)" \
    > "${ROUTE_ROOT}/PAIR_${PAIR_ARM}_COMPLETE.ok"
s2_uhost_require_free_space
echo "S2_MET_AUG_PAIR_PASS arm=${PAIR_ARM} checkpoint_sha256=${CHECKPOINT_SHA}"
