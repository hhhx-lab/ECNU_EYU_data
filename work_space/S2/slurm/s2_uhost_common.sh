#!/usr/bin/env bash

set -euo pipefail

S2_UHOST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S2_UHOST_ENV_FILE="${S2_UHOST_ENV_FILE:-${S2_UHOST_SCRIPT_DIR}/.env.uhost}"

if [[ ! -f "${S2_UHOST_ENV_FILE}" ]]; then
    echo "Missing UHost environment file: ${S2_UHOST_ENV_FILE}" >&2
    echo "Create it from ${S2_UHOST_SCRIPT_DIR}/.env.uhost.example." >&2
    exit 1
fi

set -a
source "${S2_UHOST_ENV_FILE}"
set +a

PROJECT_ROOT="${PROJECT_ROOT:-/root/brats2026/ECNU_EYU_data}"
S2_REPOSITORY="${S2_REPOSITORY:-${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository}"
S2_DATA_ROOT="${S2_DATA_ROOT:-/root/brats2026/data/s2_dataset264}"
NNUNET_RAW_ROOT="${NNUNET_RAW_ROOT:-${S2_DATA_ROOT}/nnUNet_raw}"
NNUNET_PREPROCESSED_ROOT="${NNUNET_PREPROCESSED_ROOT:-${S2_DATA_ROOT}/nnUNet_preprocessed}"
ROUTE_ROOT="${ROUTE_ROOT:-/root/brats2026/runs/s2_met_aug_route_a_20260725}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/root/brats2026/envs/s2_met_aug_h20}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
MIN_FREE_GIB="${MIN_FREE_GIB:-150}"

export PROJECT_ROOT S2_REPOSITORY S2_DATA_ROOT NNUNET_RAW_ROOT
export NNUNET_PREPROCESSED_ROOT ROUTE_ROOT CONDA_ENV_PATH PYTHON_BIN
export PYTHONUNBUFFERED=1

s2_uhost_require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -s "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        return 1
    fi
}
s2_uhost_require_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        return 1
    fi
}

s2_uhost_activate_runtime() {
    s2_uhost_require_file "${PYTHON_BIN}" "isolated Python runtime"
    export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
    hash -r
}

s2_uhost_sha256() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{print $1}'
    else
        "${PYTHON_BIN}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${path}"
    fi
}

s2_uhost_require_free_space() {
    if ! [[ "${MIN_FREE_GIB}" =~ ^[0-9]+$ ]] || (( MIN_FREE_GIB < 50 )); then
        echo "MIN_FREE_GIB must be an integer of at least 50." >&2
        return 1
    fi
    local available_kib
    available_kib=$(df -Pk "${S2_DATA_ROOT%/*}" | awk 'NR==2 {print $4}')
    local required_kib=$((MIN_FREE_GIB * 1024 * 1024))
    if (( available_kib < required_kib )); then
        echo "Insufficient disk space: available=$((available_kib / 1024 / 1024))GiB required=${MIN_FREE_GIB}GiB" >&2
        return 1
    fi
    echo "S2_UHOST_STORAGE_PASS available_gib=$((available_kib / 1024 / 1024))"
}

s2_uhost_require_single_visible_gpu() {
    s2_uhost_activate_runtime
    "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        "Each S2 job must see exactly one GPU; bind CUDA_VISIBLE_DEVICES before launch. "
        f"observed={torch.cuda.device_count()}"
    )
props = torch.cuda.get_device_properties(0)
memory_gib = props.total_memory / 1024**3
if props.major != 9 or memory_gib < 90:
    raise SystemExit(f"Expected one H20-class sm_90 GPU with >=90GiB, got {props.name}/{memory_gib:.1f}GiB")
print(f"S2_UHOST_GPU_PASS name={props.name} capability={props.major}.{props.minor} memory_gib={memory_gib:.1f}")
PY
}

s2_uhost_bind_contract() {
    export nnUNet_raw="${NNUNET_RAW_ROOT}"
    export nnUNet_preprocessed="${NNUNET_PREPROCESSED_ROOT}"
    export DATASET_NAME=Dataset264_BraTS2026_MET_Completion
    export DATASET_DIR="${nnUNet_raw}/${DATASET_NAME}"
    export PREPROCESSED_DIR="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_3d_fullres"
    export TRAIN_FILE="${S2_REPOSITORY}/data/splits/completion_warmstart/train_fixed.txt"
    export MAPPING_CSV="${MAPPING_CSV:-${PROJECT_ROOT}/work_space/G2/results/manifests/nnunet_case_mapping_master.csv}"
    export G1_CODE_DIR="${G1_CODE_DIR:-${PROJECT_ROOT}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN}"
    export G1_CHECKPOINT_ROOT="${G1_CHECKPOINT_ROOT:-${PROJECT_ROOT}/work_space/G1/results/g1_diffusion_v3_final_20260720}"
    export G1_SELECTION="${G1_SELECTION:-${PROJECT_ROOT}/work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/checkpoint_selection.json}"
    export G2_PARENT_GATE="${G2_PARENT_GATE:-${PROJECT_ROOT}/work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/g2_diffusion_qc_gate.json}"
    export E_CHECKPOINT="${E_CHECKPOINT:-${PROJECT_ROOT}/work_space/S2/results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth}"
}

s2_uhost_require_contract_inputs() {
    s2_uhost_bind_contract
    s2_uhost_require_dir "${S2_REPOSITORY}" "S2 repository"
    s2_uhost_require_dir "${DATASET_DIR}" "Dataset264 raw dataset"
    s2_uhost_require_dir "${PREPROCESSED_DIR}" "Dataset264 3d_fullres cache"
    s2_uhost_require_file "${TRAIN_FILE}" "fixed training split"
    s2_uhost_require_file "${MAPPING_CSV}" "G2 case mapping"
    s2_uhost_require_dir "${G1_CODE_DIR}" "G1 runtime code"
    s2_uhost_require_dir "${G1_CHECKPOINT_ROOT}" "G1 checkpoint root"
    s2_uhost_require_file "${G1_SELECTION}" "G1 checkpoint selection"
    s2_uhost_require_file "${G2_PARENT_GATE}" "G2 parent gate"
    s2_uhost_require_file "${E_CHECKPOINT}" "frozen E checkpoint"
}
