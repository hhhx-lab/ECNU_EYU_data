#!/bin/bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage:"
    echo "bash infer.sh INPUT_FOLDER OUTPUT_FOLDER"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2_EXPERIMENT_MODE="${S2_EXPERIMENT_MODE:-current}"

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

export nnUNet_raw="${nnUNet_raw:-${REPO_DIR}/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_DIR}/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_DIR}/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${REPO_DIR}/data/splits/${S2_EXPERIMENT_MODE}}"
export BRATS_S2_REPO_DIR="${BRATS_S2_REPO_DIR:-${REPO_DIR}}"
export S2_DATASET_ID="${S2_DATASET_ID:-${DEFAULT_S2_DATASET_ID}}"
S2_DATASET_NAME="${S2_DATASET_NAME:-${DEFAULT_S2_DATASET_NAME}}"
S2_TRAINER="${S2_TRAINER:-nnUNetTrainerBraTS2026RC}"
S2_CONFIGURATION="${S2_CONFIGURATION:-3d_fullres}"

if [[ -n "${S2_FOLDS:-}" && "${S2_FOLDS}" != "0" ]]; then
    echo "S2 cross-validation is disabled; S2_FOLDS must be unset or 0." >&2
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
if [[ ! -d "${INPUT_FOLDER}" ]]; then
    echo "S2 inference input directory does not exist: ${INPUT_FOLDER}" >&2
    exit 1
fi

python - "${INPUT_FOLDER}" <<'PY'
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
pattern = re.compile(r"^(.+)_([0-9]{4})\.nii\.gz$")
channels_by_case = defaultdict(set)
unexpected = []
for path in root.iterdir():
    if not path.is_file():
        continue
    match = pattern.match(path.name)
    if match is None:
        if path.name.endswith(".nii.gz"):
            unexpected.append(path.name)
        continue
    case_id, channel = match.groups()
    channels_by_case[case_id].add(channel)

if not channels_by_case:
    raise SystemExit(f"No nnU-Net input cases were found in {root}")
expected = {"0000", "0001", "0002", "0003"}
invalid = {
    case_id: sorted(channels)
    for case_id, channels in channels_by_case.items()
    if channels != expected
}
if unexpected or invalid:
    raise SystemExit(
        "Invalid S2 inference input: "
        f"unexpected_nifti={unexpected[:10]}, invalid_channels={list(invalid.items())[:10]}"
    )
print(f"S2 inference input verified: {len(channels_by_case)} cases, channels 0000-0003")
PY

RESULT_BASE="${nnUNet_results}/${S2_DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__${S2_CONFIGURATION}"
CHECKPOINT="${RESULT_BASE}/fold_0/checkpoint_final.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing fixed-split final checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

echo "Input  : ${INPUT_FOLDER}"
echo "Output : ${OUTPUT_FOLDER}"
echo "Trainer: ${S2_TRAINER}"
echo "Mode   : ${S2_EXPERIMENT_MODE}"
echo "Split  : fixed model (nnU-Net internal key: fold_0)"

nnUNetv2_predict \
    -i "${INPUT_FOLDER}" \
    -o "${OUTPUT_FOLDER}" \
    -d "${S2_DATASET_ID}" \
    -c "${S2_CONFIGURATION}" \
    -tr "${S2_TRAINER}" \
    -f 0
