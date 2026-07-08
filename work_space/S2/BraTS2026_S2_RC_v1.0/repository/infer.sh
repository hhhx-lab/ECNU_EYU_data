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

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

nnUNetv2_predict \
    -i ${INPUT_FOLDER} \
    -o ${OUTPUT_FOLDER} \
    -d "${S2_DATASET_ID}" \
    -c 3d_fullres \
    -f 0
