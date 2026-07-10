#!/bin/bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage:"
    echo "bash infer.sh INPUT_FOLDER OUTPUT_FOLDER"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export nnUNet_raw="${nnUNet_raw:-${REPO_DIR}/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_DIR}/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_DIR}/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${REPO_DIR}/data/splits}"
export BRATS_S2_REPO_DIR="${BRATS_S2_REPO_DIR:-${REPO_DIR}}"
export S2_DATASET_ID="${S2_DATASET_ID:-260}"
S2_DATASET_NAME="${S2_DATASET_NAME:-Dataset260_BraTS2026_MET_RealOnly}"
S2_TRAINER="${S2_TRAINER:-nnUNetTrainerBraTS2026RC}"
S2_CONFIGURATION="${S2_CONFIGURATION:-3d_fullres}"
S2_FOLDS="${S2_FOLDS:-0 1 2 3 4}"

export nnUNet_extTrainer="${REPO_DIR}/custom_nnunet"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python - <<'PY'
import importlib.util
import os
import shutil
from pathlib import Path

repo_dir = Path(os.environ["BRATS_S2_REPO_DIR"])
src = repo_dir / "custom_nnunet" / "nnUNetTrainerBraTS2026RC.py"
spec = importlib.util.find_spec("nnunetv2")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Cannot find nnunetv2 in the active Python environment.")
pkg_root = Path(list(spec.submodule_search_locations)[0])
dst = pkg_root / "training" / "nnUNetTrainer" / src.name
if not dst.parent.exists():
    raise SystemExit(f"Cannot find nnU-Net trainer directory: {dst.parent}")
if not dst.exists() or dst.read_bytes() != src.read_bytes():
    shutil.copy2(src, dst)
print(f"Custom trainer ready: {dst}")
PY

INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"
read -r -a FOLD_ARGS <<< "${S2_FOLDS}"

if [[ ${#FOLD_ARGS[@]} -eq 0 ]]; then
    echo "S2_FOLDS must contain at least one fold, for example: '0 1 2 3 4'" >&2
    exit 2
fi

RESULT_BASE="${nnUNet_results}/${S2_DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__${S2_CONFIGURATION}"
for FOLD in "${FOLD_ARGS[@]}"; do
    if [[ ! "${FOLD}" =~ ^[0-4]$ ]]; then
        echo "Invalid fold in S2_FOLDS: ${FOLD}" >&2
        exit 2
    fi
    if [[ ! -f "${RESULT_BASE}/fold_${FOLD}/checkpoint_final.pth" ]]; then
        echo "Missing final checkpoint for fold ${FOLD}: ${RESULT_BASE}/fold_${FOLD}/checkpoint_final.pth" >&2
        exit 1
    fi
done

echo "Input  : ${INPUT_FOLDER}"
echo "Output : ${OUTPUT_FOLDER}"
echo "Trainer: ${S2_TRAINER}"
echo "Folds  : ${FOLD_ARGS[*]}"

nnUNetv2_predict \
    -i "${INPUT_FOLDER}" \
    -o "${OUTPUT_FOLDER}" \
    -d "${S2_DATASET_ID}" \
    -c "${S2_CONFIGURATION}" \
    -tr "${S2_TRAINER}" \
    -f "${FOLD_ARGS[@]}"
