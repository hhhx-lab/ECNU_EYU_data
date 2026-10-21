#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/${USER}/projects/ECNU_EYU_data}"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency}"
S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
DATASET_NAME="Dataset264_BraTS2026_MET_Completion"
SOURCE_PREPROCESSED="${SOURCE_ROOT}/nnUNet_preprocessed/${DATASET_NAME}"
SOURCE_RAW="${SOURCE_ROOT}/nnUNet_raw/${DATASET_NAME}"
SOURCE_SPLITS="${S2_REPO}/data/splits/completion_warmstart"

DEST_HOST="${DEST_HOST:-117.50.177.229}"
DEST_PORT="${DEST_PORT:-23}"
DEST_ROOT="${DEST_ROOT:-/cloud/cloud-ssd1/brats2026/s2}"
TRANSFER_KEY="${TRANSFER_KEY:-${PROJECT_ROOT}/work_space/S2/.transfer/s2_cloud_ed25519}"
SSH_KNOWN_HOSTS="${SSH_KNOWN_HOSTS:-${TRANSFER_KEY}.known_hosts}"
TRANSFER_JOBS="${TRANSFER_JOBS:-4}"

SSH=(
    ssh -T -p "${DEST_PORT}"
    -i "${TRANSFER_KEY}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=yes
    -o UserKnownHostsFile="${SSH_KNOWN_HOSTS}"
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=20
    "root@${DEST_HOST}"
)

[[ -f "${TRANSFER_KEY}" ]] || { echo "Missing transfer key: ${TRANSFER_KEY}" >&2; exit 1; }
[[ -s "${SSH_KNOWN_HOSTS}" ]] || { echo "Missing SSH known-hosts file: ${SSH_KNOWN_HOSTS}" >&2; exit 1; }
[[ "${TRANSFER_JOBS}" =~ ^[1-6]$ ]] || {
    echo "TRANSFER_JOBS must be an integer from 1 to 6, got: ${TRANSFER_JOBS}" >&2
    exit 2
}
[[ -d "${SOURCE_PREPROCESSED}/nnUNetPlans_3d_fullres" ]] || {
    echo "Missing source preprocessed cache: ${SOURCE_PREPROCESSED}" >&2
    exit 1
}
[[ -d "${SOURCE_RAW}/imagesTr" && -d "${SOURCE_RAW}/labelsTr" ]] || {
    echo "Missing source raw Dataset264: ${SOURCE_RAW}" >&2
    exit 1
}
[[ -s "${SOURCE_SPLITS}/train_fixed.txt" && -s "${SOURCE_SPLITS}/val_fixed.txt" ]] || {
    echo "Missing completion fixed split: ${SOURCE_SPLITS}" >&2
    exit 1
}

PREFIXES=()
while IFS= read -r prefix; do
    [[ "${prefix}" =~ ^BraTSMET_[0-9]{4}$ ]] || {
        echo "Unexpected Dataset264 case prefix: ${prefix}" >&2
        exit 1
    }
    PREFIXES+=("${prefix}")
done < <(
    awk '{print substr($0, 1, 13)}' \
        "${SOURCE_SPLITS}/train_fixed.txt" "${SOURCE_SPLITS}/val_fixed.txt" \
        | sort -u
)
((${#PREFIXES[@]} > 0)) || { echo "No Dataset264 case prefixes found" >&2; exit 1; }
echo "TRANSFER_PREFIXES count=${#PREFIXES[@]} values=${PREFIXES[*]}"

remote() {
    "${SSH[@]}" "$@"
}

stats_for_pattern() {
    local directory="$1"
    local pattern="$2"
    find -L "${directory}" -maxdepth 1 -type f -name "${pattern}" -printf '%s\n' \
        | awk '{count += 1; bytes += $1} END {printf "%.0f:%.0f", count, bytes}'
}

remote_stats_for_pattern() {
    local directory="$1"
    local pattern="$2"
    remote "find '${directory}' -maxdepth 1 -type f -name '${pattern}' -printf '%s\\n' | awk '{count += 1; bytes += \$1} END {printf \"%.0f:%.0f\", count, bytes}'"
}

verify_pattern() {
    local source_dir="$1"
    local dest_dir="$2"
    local pattern="$3"
    local source_stats dest_stats
    source_stats=$(stats_for_pattern "${source_dir}" "${pattern}")
    dest_stats=$(remote_stats_for_pattern "${dest_dir}" "${pattern}")
    if [[ "${source_stats}" != "${dest_stats}" ]]; then
        echo "Transfer verification failed: pattern=${pattern} source=${source_stats} dest=${dest_stats}" >&2
        return 1
    fi
    echo "TRANSFER_BATCH_PASS pattern=${pattern} stats=${source_stats}"
}

transfer_preprocessed_batch() {
    local prefix="$1"
    local source_dir="${SOURCE_PREPROCESSED}/nnUNetPlans_3d_fullres"
    local dest_dir="${DEST_ROOT}/nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans_3d_fullres"
    local marker="${DEST_ROOT}/transfer_markers/preprocessed_${prefix}.ok"
    if remote "test -s '${marker}'"; then
        echo "TRANSFER_BATCH_SKIP kind=preprocessed prefix=${prefix}"
        return
    fi
    if verify_pattern "${source_dir}" "${dest_dir}" "${prefix}*"; then
        remote "printf '%s\n' 'verified' > '${marker}'"
        echo "TRANSFER_BATCH_REUSE kind=preprocessed prefix=${prefix}"
        return
    fi
    echo "TRANSFER_BATCH_START kind=preprocessed prefix=${prefix}"
    find "${source_dir}" -maxdepth 1 -type f -name "${prefix}*" -printf '%f\0' \
        | tar -C "${source_dir}" --null -T - -cf - \
        | remote "tar -C '${dest_dir}' -xf -"
    verify_pattern "${source_dir}" "${dest_dir}" "${prefix}*"
    remote "printf '%s\n' 'verified' > '${marker}'"
}

transfer_raw_batch() {
    local prefix="$1"
    local dest_dataset="${DEST_ROOT}/nnUNet_raw/${DATASET_NAME}"
    local marker="${DEST_ROOT}/transfer_markers/raw_${prefix}.ok"
    if remote "test -s '${marker}'"; then
        echo "TRANSFER_BATCH_SKIP kind=raw prefix=${prefix}"
        return
    fi
    if verify_pattern "${SOURCE_RAW}/imagesTr" "${dest_dataset}/imagesTr" "${prefix}*" \
        && verify_pattern "${SOURCE_RAW}/labelsTr" "${dest_dataset}/labelsTr" "${prefix}*"; then
        remote "printf '%s\n' 'verified' > '${marker}'"
        echo "TRANSFER_BATCH_REUSE kind=raw prefix=${prefix}"
        return
    fi
    echo "TRANSFER_BATCH_START kind=raw prefix=${prefix}"
    (
        cd "${SOURCE_RAW}"
        find imagesTr labelsTr -maxdepth 1 \( -type f -o -type l \) \
            -name "${prefix}*" -print0 \
            | tar -h --null -T - -cf -
    ) | remote "tar -C '${dest_dataset}' -xf -"
    verify_pattern "${SOURCE_RAW}/imagesTr" "${dest_dataset}/imagesTr" "${prefix}*"
    verify_pattern "${SOURCE_RAW}/labelsTr" "${dest_dataset}/labelsTr" "${prefix}*"
    remote "printf '%s\n' 'verified' > '${marker}'"
}

run_in_batches() {
    local function_name="$1"
    shift
    local items=("$@")
    local start offset index pid status
    local pids=()

    for ((start = 0; start < ${#items[@]}; start += TRANSFER_JOBS)); do
        pids=()
        for ((offset = 0; offset < TRANSFER_JOBS; offset += 1)); do
            index=$((start + offset))
            ((index < ${#items[@]})) || break
            "${function_name}" "${items[index]}" &
            pids+=("$!")
        done
        status=0
        for pid in "${pids[@]}"; do
            wait "${pid}" || status=1
        done
        ((status == 0)) || return 1
    done
}

remote "mkdir -p \
    '${DEST_ROOT}/transfer_markers' \
    '${DEST_ROOT}/nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans_3d_fullres' \
    '${DEST_ROOT}/nnUNet_raw/${DATASET_NAME}/imagesTr' \
    '${DEST_ROOT}/nnUNet_raw/${DATASET_NAME}/labelsTr' \
    '${DEST_ROOT}/splits/completion_warmstart'"

tar -C "${SOURCE_PREPROCESSED}" -cf - \
    dataset.json nnUNetPlans.json dataset_fingerprint.json completion_plans_audit.json gt_segmentations \
    | remote "tar -C '${DEST_ROOT}/nnUNet_preprocessed/${DATASET_NAME}' -xf -"

run_in_batches transfer_preprocessed_batch "${PREFIXES[@]}"

tar -C "${SOURCE_RAW}" -cf - dataset.json \
    | remote "tar -C '${DEST_ROOT}/nnUNet_raw/${DATASET_NAME}' -xf -"
run_in_batches transfer_raw_batch "${PREFIXES[@]}"

tar -C "${SOURCE_SPLITS}" -cf - . \
    | remote "tar -C '${DEST_ROOT}/splits/completion_warmstart' -xf -"

verify_pattern \
    "${SOURCE_PREPROCESSED}/nnUNetPlans_3d_fullres" \
    "${DEST_ROOT}/nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans_3d_fullres" \
    'BraTSMET_*'
verify_pattern \
    "${SOURCE_RAW}/imagesTr" \
    "${DEST_ROOT}/nnUNet_raw/${DATASET_NAME}/imagesTr" \
    'BraTSMET_*'
verify_pattern \
    "${SOURCE_RAW}/labelsTr" \
    "${DEST_ROOT}/nnUNet_raw/${DATASET_NAME}/labelsTr" \
    'BraTSMET_*'

EXPECTED_MARKERS=$((${#PREFIXES[@]} * 2))
ACTUAL_MARKERS=$(remote "find '${DEST_ROOT}/transfer_markers' -maxdepth 1 -type f \
    \( -name 'preprocessed_BraTSMET_????.ok' -o -name 'raw_BraTSMET_????.ok' \) \
    | wc -l" | tr -d '[:space:]')
if [[ "${ACTUAL_MARKERS}" != "${EXPECTED_MARKERS}" ]]; then
    echo "Transfer marker count mismatch: expected=${EXPECTED_MARKERS} actual=${ACTUAL_MARKERS}" >&2
    exit 1
fi

remote "printf '%s\n' 'S2_CLOUD_DATA_TRANSFER_PASS' > '${DEST_ROOT}/TRANSFER_COMPLETE.ok'"
echo "S2_CLOUD_DATA_TRANSFER_PASS destination=${DEST_HOST}:${DEST_ROOT} markers=${ACTUAL_MARKERS}"
