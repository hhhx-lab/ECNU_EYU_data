#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2_EXPERIMENT_MODE="${S2_EXPERIMENT_MODE:-current}"
DEFAULT_S2_SPLIT_MODE="${S2_EXPERIMENT_MODE}"

case "${S2_EXPERIMENT_MODE}" in
    current)
        DEFAULT_S2_DATASET_ID=263
        DEFAULT_S2_DATASET_NAME=Dataset263_BraTS2026_MET_RealOnly_Current
        DEFAULT_S2_TRAIN_COUNT=823
        DEFAULT_S2_VAL_COUNT=103
        DEFAULT_S2_TRAINER=nnUNetTrainerBraTS2026RC
        ;;
    legacy)
        DEFAULT_S2_DATASET_ID=260
        DEFAULT_S2_DATASET_NAME=Dataset260_BraTS2026_MET_RealOnly
        DEFAULT_S2_TRAIN_COUNT=828
        DEFAULT_S2_VAL_COUNT=207
        DEFAULT_S2_TRAINER=nnUNetTrainerBraTS2026RC
        ;;
    completion_warmstart)
        DEFAULT_S2_DATASET_ID=264
        DEFAULT_S2_DATASET_NAME=Dataset264_BraTS2026_MET_Completion
        DEFAULT_S2_TRAIN_COUNT=1035
        DEFAULT_S2_VAL_COUNT=103
        DEFAULT_S2_TRAINER=nnUNetTrainerBraTS2026RCCompletionFineTune
        ;;
    completion_online)
        echo "completion_online is retired: it uses the legacy whole-label bridge. Use met_aug_route_a after Route A approval." >&2
        exit 2
        ;;
    met_aug_route_a)
        DEFAULT_S2_DATASET_ID=264
        DEFAULT_S2_DATASET_NAME=Dataset264_BraTS2026_MET_Completion
        DEFAULT_S2_TRAIN_COUNT=1035
        DEFAULT_S2_VAL_COUNT=103
        DEFAULT_S2_TRAINER=nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune
        DEFAULT_S2_SPLIT_MODE=completion_warmstart
        ;;
    met_aug_route_a_control)
        DEFAULT_S2_DATASET_ID=264
        DEFAULT_S2_DATASET_NAME=Dataset264_BraTS2026_MET_Completion
        DEFAULT_S2_TRAIN_COUNT=1035
        DEFAULT_S2_VAL_COUNT=103
        DEFAULT_S2_TRAINER=nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune
        DEFAULT_S2_SPLIT_MODE=completion_warmstart
        ;;
    met_aug_route_a_fix_v3_emergency)
        DEFAULT_S2_DATASET_ID=264
        DEFAULT_S2_DATASET_NAME=Dataset264_BraTS2026_MET_Completion
        DEFAULT_S2_TRAIN_COUNT=1035
        DEFAULT_S2_VAL_COUNT=103
        DEFAULT_S2_TRAINER=nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune
        DEFAULT_S2_SPLIT_MODE=completion_warmstart
        ;;
    *)
        echo "S2_EXPERIMENT_MODE must be current, legacy, completion_warmstart, met_aug_route_a, met_aug_route_a_control, or met_aug_route_a_fix_v3_emergency, got: ${S2_EXPERIMENT_MODE}" >&2
        exit 2
        ;;
esac

export nnUNet_raw="${nnUNet_raw:-${REPO_DIR}/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_DIR}/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_DIR}/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${REPO_DIR}/data/splits/${DEFAULT_S2_SPLIT_MODE}}"
export BRATS_S2_REPO_DIR="${BRATS_S2_REPO_DIR:-${REPO_DIR}}"
export S2_DATASET_ID="${S2_DATASET_ID:-${DEFAULT_S2_DATASET_ID}}"
export S2_DATASET_NAME="${S2_DATASET_NAME:-${DEFAULT_S2_DATASET_NAME}}"
export NNUNET_DATASET_DIR="${NNUNET_DATASET_DIR:-${nnUNet_raw}/${S2_DATASET_NAME}}"
S2_EXPECTED_TRAIN_COUNT="${S2_EXPECTED_TRAIN_COUNT:-${DEFAULT_S2_TRAIN_COUNT}}"
S2_EXPECTED_VAL_COUNT="${S2_EXPECTED_VAL_COUNT:-${DEFAULT_S2_VAL_COUNT}}"

S2_TRAINER="${S2_TRAINER:-${DEFAULT_S2_TRAINER}}"
S2_CONFIGURATION="${S2_CONFIGURATION:-3d_fullres}"
S2_PREPROCESSED_DATA_IDENTIFIER="${S2_PREPROCESSED_DATA_IDENTIFIER:-nnUNetPlans_3d_fullres}"
S2_CONTINUE="${S2_CONTINUE:-auto}"
S2_SKIP_COMPLETED="${S2_SKIP_COMPLETED:-1}"

if [[ -n "${S2_FOLD:-}" && "${S2_FOLD}" != "0" ]]; then
    echo "S2 cross-validation is disabled; S2_FOLD must be unset or 0." >&2
    exit 2
fi
if [[ "${S2_CONTINUE}" != "auto" && "${S2_CONTINUE}" != "0" && "${S2_CONTINUE}" != "1" ]]; then
    echo "S2_CONTINUE must be auto, 0, or 1, got: ${S2_CONTINUE}" >&2
    exit 2
fi
if [[ "${S2_SKIP_COMPLETED}" != "0" && "${S2_SKIP_COMPLETED}" != "1" ]]; then
    echo "S2_SKIP_COMPLETED must be 0 or 1, got: ${S2_SKIP_COMPLETED}" >&2
    exit 2
fi
if [[ "${S2_DATASET_NAME}" != Dataset${S2_DATASET_ID}_* ]]; then
    echo "S2_DATASET_NAME must start with Dataset${S2_DATASET_ID}_, got: ${S2_DATASET_NAME}" >&2
    exit 2
fi
if [[ "${S2_DATASET_ID}" != "${DEFAULT_S2_DATASET_ID}" || "${S2_DATASET_NAME}" != "${DEFAULT_S2_DATASET_NAME}" ]]; then
    echo "${S2_EXPERIMENT_MODE} mode is locked to dataset ${DEFAULT_S2_DATASET_ID}/${DEFAULT_S2_DATASET_NAME}." >&2
    exit 2
fi
if [[ "${S2_EXPERIMENT_MODE}" == met_aug_route_a* && "${S2_TRAINER}" != "${DEFAULT_S2_TRAINER}" ]]; then
    echo "${S2_EXPERIMENT_MODE} is locked to trainer ${DEFAULT_S2_TRAINER}." >&2
    exit 2
fi
if [[ "${S2_EXPERIMENT_MODE}" == met_aug_route_a* ]]; then
    export S2_PAIRED_TRAINING_SEED="${S2_PAIRED_TRAINING_SEED:-20260724}"
    if [[ "${S2_PAIRED_TRAINING_SEED}" != "20260724" ]]; then
        echo "${S2_EXPERIMENT_MODE} is locked to S2_PAIRED_TRAINING_SEED=20260724." >&2
        exit 2
    fi
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    if [[ "${CUBLAS_WORKSPACE_CONFIG}" != ":4096:8" ]]; then
        echo "${S2_EXPERIMENT_MODE} is locked to CUBLAS_WORKSPACE_CONFIG=:4096:8." >&2
        exit 2
    fi
    export S2_COMPLETION_EPOCHS="${S2_COMPLETION_EPOCHS:-200}"
    export S2_COMPLETION_INITIAL_LR="${S2_COMPLETION_INITIAL_LR:-0.001}"
    export S2_COMPLETION_SAVE_EVERY="${S2_COMPLETION_SAVE_EVERY:-25}"
    export S2_FOCAL_GAMMA="${S2_FOCAL_GAMMA:-2.0}"
    export nnUNet_compile="${nnUNet_compile:-0}"
    if [[ "${S2_COMPLETION_EPOCHS}" != "200" || \
          "${S2_COMPLETION_INITIAL_LR}" != "0.001" || \
          "${S2_COMPLETION_SAVE_EVERY}" != "25" || \
          "${S2_FOCAL_GAMMA}" != "2.0" || \
          "${nnUNet_compile}" != "0" ]]; then
        echo "${S2_EXPERIMENT_MODE} requires epochs=200, lr=0.001, save_every=25, focal_gamma=2.0, nnUNet_compile=0." >&2
        exit 2
    fi
fi

export nnUNet_extTrainer="${REPO_DIR}/custom_nnunet"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python - <<'PY'
import importlib.util
import os
import shutil
from pathlib import Path

repo_dir = Path(os.environ["BRATS_S2_REPO_DIR"])
source_dir = repo_dir / "custom_nnunet"
filenames = [
    "nnUNetTrainerBraTS2026RC.py",
    "nnUNetTrainerBraTS2026RCCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCOnlineDiffusion.py",
    "nnUNetTrainerBraTS2026RCMetAugFocalCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCMetAugControlFocalCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune.py",
    "small_lesion_variants.py",
    "small_lesion_trainer_mixins.py",
    "nnUNetTrainerBraTS2026RCA1CompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCFocalCompletionFineTune.py",
    "nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune.py",
    "online_diffusion_transform.py",
    "online_diffusion_contract.py",
    "met_aug_core.py",
    "met_aug_data_loader.py",
    "met_aug_diffusion.py",
    "met_aug_fix_v2.py",
    "met_aug_fix_v3.py",
    "met_aug_fix_v3_emergency.py",
    "met_aug_gate.py",
    "met_aug_paired_training.py",
    "met_aug_transform.py",
]
spec = importlib.util.find_spec("nnunetv2")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Cannot find nnunetv2 in the active Python environment.")
pkg_root = Path(list(spec.submodule_search_locations)[0])
trainer_dir = pkg_root / "training" / "nnUNetTrainer"
if not trainer_dir.exists():
    raise SystemExit(f"Cannot find nnU-Net trainer directory: {trainer_dir}")
for filename in filenames:
    src = source_dir / filename
    dst = trainer_dir / filename
    if not src.is_file():
        raise SystemExit(f"Missing custom trainer component: {src}")
    if not dst.exists() or dst.read_bytes() != src.read_bytes():
        shutil.copy2(src, dst)
    print(f"Custom trainer component ready: {dst}")
PY

TRAIN_FILE="${BRATS_SPLIT_DIR}/train_fixed.txt"
VAL_FILE="${BRATS_SPLIT_DIR}/val_fixed.txt"
RESULT_FOLD_DIR="${nnUNet_results}/${S2_DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__${S2_CONFIGURATION}/fold_0"
CHECKPOINT_LATEST="${RESULT_FOLD_DIR}/checkpoint_latest.pth"
CHECKPOINT_FINAL="${RESULT_FOLD_DIR}/checkpoint_final.pth"
VALIDATION_DIR="${RESULT_FOLD_DIR}/validation"
VALIDATION_SUMMARY="${VALIDATION_DIR}/summary.json"
if [[ "${S2_EXPERIMENT_MODE}" == "met_aug_route_a" || "${S2_EXPERIMENT_MODE}" == "met_aug_route_a_fix_v3_emergency" ]]; then
    EXPECTED_MET_AUG_AUDIT_PATH="${RESULT_FOLD_DIR}/met_aug_events.jsonl"
    export S2_MET_AUG_AUDIT_PATH="${S2_MET_AUG_AUDIT_PATH:-${EXPECTED_MET_AUG_AUDIT_PATH}}"
    if [[ "${S2_MET_AUG_AUDIT_PATH}" != "${EXPECTED_MET_AUG_AUDIT_PATH}" ]]; then
        echo "met_aug_route_a requires its audit log inside the current fold result directory: ${EXPECTED_MET_AUG_AUDIT_PATH}" >&2
        exit 2
    fi
fi

if [[ ! -s "${TRAIN_FILE}" || ! -s "${VAL_FILE}" ]]; then
    echo "Missing fixed split files:" >&2
    echo "  ${TRAIN_FILE}" >&2
    echo "  ${VAL_FILE}" >&2
    echo "Run 04_s2_realonly_prepare_nyu.slurm first." >&2
    exit 1
fi

python "${REPO_DIR}/scripts/05_validate_fixed_split_cache.py" \
    --train-file "${TRAIN_FILE}" \
    --val-file "${VAL_FILE}" \
    --dataset-dir "${NNUNET_DATASET_DIR}" \
    --preprocessed-dir "${nnUNet_preprocessed}/${S2_DATASET_NAME}/${S2_PREPROCESSED_DATA_IDENTIFIER}" \
    --output-json "${BRATS_SPLIT_DIR}/fixed_split_cache_audit.json"

echo "Starting BraTS2026 RC training..."
echo "Mode       : ${S2_EXPERIMENT_MODE}"
echo "Split      : fixed train/validation (nnU-Net internal key: fold_0)"
TRAIN_COUNT=$(wc -l < "${TRAIN_FILE}" | tr -d ' ')
VAL_COUNT=$(wc -l < "${VAL_FILE}" | tr -d ' ')
echo "Train split: ${TRAIN_FILE} (${TRAIN_COUNT})"
echo "Val split  : ${VAL_FILE} (${VAL_COUNT})"
echo "Results    : ${RESULT_FOLD_DIR}"
if [[ "${TRAIN_COUNT}" != "${S2_EXPECTED_TRAIN_COUNT}" || "${VAL_COUNT}" != "${S2_EXPECTED_VAL_COUNT}" ]]; then
    echo "Fixed split count mismatch for ${S2_EXPERIMENT_MODE}: expected ${S2_EXPECTED_TRAIN_COUNT}/${S2_EXPECTED_VAL_COUNT}, got ${TRAIN_COUNT}/${VAL_COUNT}" >&2
    exit 1
fi

VALIDATION_PREDICTIONS=0
if [[ -d "${VALIDATION_DIR}" ]]; then
    VALIDATION_PREDICTIONS=$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
fi

if [[ "${S2_EXPERIMENT_MODE}" == "met_aug_route_a" ]]; then
    if [[ "${S2_MET_AUG_ENABLE:-0}" != "1" ]]; then
        echo "met_aug_route_a requires S2_MET_AUG_ENABLE=1 after Route A approval." >&2
        exit 2
    fi
    for required_path_var in \
        S2_MET_AUG_COMPONENT_MANIFEST \
        S2_MET_AUG_ROUTE_CONFIG \
        S2_MET_AUG_VALID_MASK_MANIFEST \
        S2_MET_AUG_ROUTE_GATE \
        S2_MET_AUG_G1_CHECKPOINT_SELECTION \
        S2_MET_AUG_G2_QC_GATE; do
        if [[ -z "${!required_path_var:-}" || ! -f "${!required_path_var}" ]]; then
            echo "met_aug_route_a requires file variable ${required_path_var}." >&2
            exit 2
        fi
    done
    for required_dir_var in S2_MET_AUG_G1_CODE_DIR S2_MET_AUG_G1_CHECKPOINT_ROOT; do
        if [[ -z "${!required_dir_var:-}" || ! -d "${!required_dir_var}" ]]; then
            echo "met_aug_route_a requires directory variable ${required_dir_var}." >&2
            exit 2
        fi
    done
fi
if [[ "${S2_EXPERIMENT_MODE}" == "met_aug_route_a_fix_v3_emergency" ]]; then
    if [[ "${S2_MET_AUG_ENABLE:-0}" != "1" ]]; then
        echo "met_aug_route_a_fix_v3_emergency requires S2_MET_AUG_ENABLE=1." >&2
        exit 2
    fi
    for required_path_var in \
        S2_MET_AUG_COMPONENT_MANIFEST \
        S2_MET_AUG_ROUTE_CONFIG \
        S2_MET_AUG_VALID_MASK_MANIFEST \
        S2_MET_AUG_EMERGENCY_DECISION \
        S2_MET_AUG_FIX_V3_CALIBRATION \
        S2_MET_AUG_ORIGINAL_E_CHECKPOINT \
        S2_MET_AUG_FIX_V2_FAILURE_AUDIT \
        S2_MET_AUG_G1_CHECKPOINT_SELECTION \
        S2_MET_AUG_G2_QC_GATE; do
        if [[ -z "${!required_path_var:-}" || ! -f "${!required_path_var}" ]]; then
            echo "met_aug_route_a_fix_v3_emergency requires file variable ${required_path_var}." >&2
            exit 2
        fi
    done
    for required_dir_var in S2_MET_AUG_G1_CODE_DIR S2_MET_AUG_G1_CHECKPOINT_ROOT; do
        if [[ -z "${!required_dir_var:-}" || ! -d "${!required_dir_var}" ]]; then
            echo "met_aug_route_a_fix_v3_emergency requires directory variable ${required_dir_var}." >&2
            exit 2
        fi
    done
fi
if [[ "${S2_EXPERIMENT_MODE}" == "met_aug_route_a_control" && "${S2_MET_AUG_ENABLE:-0}" != "0" ]]; then
    echo "met_aug_route_a_control requires S2_MET_AUG_ENABLE=0." >&2
    exit 2
fi

TRAIN_CMD=(
    nnUNetv2_train
    "${S2_DATASET_ID}" "${S2_CONFIGURATION}" 0
    -tr "${S2_TRAINER}"
    -num_gpus 1
)

if [[ -f "${CHECKPOINT_FINAL}" ]]; then
    if [[ "${S2_SKIP_COMPLETED}" == "1" && -f "${VALIDATION_SUMMARY}" && "${VALIDATION_PREDICTIONS}" == "${VAL_COUNT}" ]]; then
        echo "Fixed-split model is complete: final checkpoint and ${VALIDATION_PREDICTIONS}/${VAL_COUNT} validation predictions exist. Skipping."
        exit 0
    fi
    echo "Final checkpoint exists but validation output is missing or incomplete (${VALIDATION_PREDICTIONS}/${VAL_COUNT}); running validation only."
    TRAIN_CMD+=(--val)
elif [[ "${S2_CONTINUE}" == "1" ]]; then
    if [[ ! -f "${CHECKPOINT_LATEST}" ]]; then
        echo "S2_CONTINUE=1 but checkpoint_latest.pth is missing: ${CHECKPOINT_LATEST}" >&2
        exit 1
    fi
    TRAIN_CMD+=(--c)
elif [[ "${S2_CONTINUE}" == "auto" && -f "${CHECKPOINT_LATEST}" ]]; then
    echo "Found checkpoint_latest.pth; resuming fixed-split training."
    TRAIN_CMD+=(--c)
else
    echo "No resumable checkpoint found; starting fixed-split training from scratch."
    if [[ "${S2_EXPERIMENT_MODE}" == "completion_warmstart" || "${S2_EXPERIMENT_MODE}" == met_aug_route_a* ]]; then
        S2_PRETRAINED_WEIGHTS="${S2_PRETRAINED_WEIGHTS:-}"
        if [[ ! -f "${S2_PRETRAINED_WEIGHTS}" ]]; then
            echo "${S2_EXPERIMENT_MODE} requires S2_PRETRAINED_WEIGHTS: ${S2_PRETRAINED_WEIGHTS:-unset}" >&2
            exit 1
        fi
        if [[ "${S2_EXPERIMENT_MODE}" == met_aug_route_a* ]]; then
            EXPECTED_E_SHA256="4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267"
            if command -v sha256sum >/dev/null 2>&1; then
                ACTUAL_PRETRAINED_SHA256="$(sha256sum "${S2_PRETRAINED_WEIGHTS}" | awk '{print $1}')"
            elif command -v shasum >/dev/null 2>&1; then
                ACTUAL_PRETRAINED_SHA256="$(shasum -a 256 "${S2_PRETRAINED_WEIGHTS}" | awk '{print $1}')"
            else
                echo "No SHA256 command is available." >&2
                exit 1
            fi
            if [[ "${ACTUAL_PRETRAINED_SHA256}" != "${EXPECTED_E_SHA256}" ]]; then
                echo "${S2_EXPERIMENT_MODE} must warm-start from the frozen E checkpoint; SHA256 mismatch." >&2
                exit 1
            fi
        fi
        if [[ "${S2_TRAINER}" == *RCA1* ]]; then
            export S2_PARTIAL_PRETRAINED_WEIGHTS="${S2_PRETRAINED_WEIGHTS}"
            echo "Partial warm-starting ${S2_EXPERIMENT_MODE} from: ${S2_PRETRAINED_WEIGHTS}"
        else
            TRAIN_CMD+=(-pretrained_weights "${S2_PRETRAINED_WEIGHTS}")
            echo "Warm-starting ${S2_EXPERIMENT_MODE} from: ${S2_PRETRAINED_WEIGHTS}"
        fi
    fi
fi

printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

if [[ ! -f "${CHECKPOINT_FINAL}" ]]; then
    echo "Fixed-split command exited without checkpoint_final.pth: ${CHECKPOINT_FINAL}" >&2
    exit 1
fi
if [[ ! -f "${VALIDATION_SUMMARY}" ]]; then
    echo "Fixed-split command exited without validation summary: ${VALIDATION_SUMMARY}" >&2
    exit 1
fi
VALIDATION_PREDICTIONS=$(find "${VALIDATION_DIR}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
if [[ "${VALIDATION_PREDICTIONS}" != "${VAL_COUNT}" ]]; then
    echo "Fixed validation output is incomplete: ${VALIDATION_PREDICTIONS}/${VAL_COUNT}" >&2
    exit 1
fi
echo "Fixed-split training complete: final checkpoint and ${VALIDATION_PREDICTIONS} validation predictions verified."
