#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-status}"
PART="${2:-}"

DATASET_NAME="${DATASET_NAME:-Dataset264_BraTS2026_MET_Completion}"
ECNU_SOURCE_ROOT="${ECNU_SOURCE_ROOT:-/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency}"
TRANSFER_ROOT="${TRANSFER_ROOT:-/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/s2_h20_dataset264_direct_relay_20260726}"
TRANSFER_KEY="${TRANSFER_KEY:-/public/home/zqchen/.ssh/s2_h20_dataset264_20260726}"
TRANSFER_KNOWN_HOSTS="${TRANSFER_KNOWN_HOSTS:-/public/home/zqchen/.ssh/known_hosts_s2_h20}"
UHOST_HOST="${UHOST_HOST:-117.50.190.178}"
UHOST_PORT="${UHOST_PORT:-23}"
UHOST_USER="${UHOST_USER:-root}"
UHOST_DATA_ROOT="${UHOST_DATA_ROOT:-/root/brats2026/data/s2_dataset264}"
UHOST_PROJECT_ROOT="${UHOST_PROJECT_ROOT:-/root/brats2026/ECNU_EYU_data}"

SSH_OPTIONS=(
    -p "${UHOST_PORT}"
    -i "${TRANSFER_KEY}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=yes
    -o "UserKnownHostsFile=${TRANSFER_KNOWN_HOSTS}"
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=6
)
RSYNC_SSH="ssh -p ${UHOST_PORT} -i ${TRANSFER_KEY} -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=${TRANSFER_KNOWN_HOSTS} -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_transfer_contract() {
    command -v rsync >/dev/null 2>&1 || die "rsync is unavailable"
    command -v ssh >/dev/null 2>&1 || die "ssh is unavailable"
    [[ -s "${TRANSFER_KEY}" ]] || die "missing transfer key: ${TRANSFER_KEY}"
    [[ -s "${TRANSFER_KNOWN_HOSTS}" ]] || die "missing pinned H20 host key: ${TRANSFER_KNOWN_HOSTS}"
    chmod 600 "${TRANSFER_KEY}" "${TRANSFER_KNOWN_HOSTS}"
    mkdir -p "${TRANSFER_ROOT}"
}

part_source() {
    case "$1" in
        nnUNet_raw|nnUNet_preprocessed)
            printf '%s/%s/%s\n' "${ECNU_SOURCE_ROOT}" "$1" "${DATASET_NAME}"
            ;;
        *)
            die "unknown transfer part: $1"
            ;;
    esac
}

part_is_running() {
    local part="$1"
    local pid_file="${TRANSFER_ROOT}/${part}.pid"
    [[ -s "${pid_file}" ]] || return 1
    local pid
    pid=$(cat "${pid_file}")
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    ps -p "${pid}" -o command= 2>/dev/null | grep -Fq "${DATASET_NAME}"
}

status_part() {
    local part="$1"
    local pid_file="${TRANSFER_ROOT}/${part}.pid"
    local log_file="${TRANSFER_ROOT}/${part}.log"
    local complete_file="${TRANSFER_ROOT}/${part}.complete"
    local exit_file="${TRANSFER_ROOT}/${part}.exit_code"
    local pid=none
    local state=not_started

    [[ -s "${pid_file}" ]] && pid=$(cat "${pid_file}")
    if part_is_running "${part}"; then
        state=running
    elif [[ -s "${complete_file}" ]]; then
        state=complete
    elif [[ -s "${pid_file}" ]]; then
        state=stopped
    fi

    echo "part=${part} state=${state} pid=${pid} exit_code=$(cat "${exit_file}" 2>/dev/null || echo unknown)"
    if [[ -s "${log_file}" ]]; then
        tail -c 800 "${log_file}" | tr '\r' '\n' | tail -5
    fi
}

run_worker() {
    local part="$1"
    require_transfer_contract
    local source
    source=$(part_source "${part}")
    [[ -d "${source}" ]] || die "missing source dataset part: ${source}"

    local complete_file="${TRANSFER_ROOT}/${part}.complete"
    local exit_file="${TRANSFER_ROOT}/${part}.exit_code"
    rm -f "${complete_file}" "${exit_file}"
    echo "DATASET264_DIRECT_TRANSFER_START part=${part} started_at=$(date -u +%FT%TZ)"

    set +e
    rsync -aL --partial --append-verify --info=progress2 \
        -e "${RSYNC_SSH}" \
        "${source}" \
        "${UHOST_USER}@${UHOST_HOST}:${UHOST_DATA_ROOT}/${part}/"
    local rc=$?
    set -e

    printf '%s\n' "${rc}" > "${exit_file}"
    if (( rc == 0 )); then
        printf 'part=%s\ncompleted_at=%s\n' "${part}" "$(date -u +%FT%TZ)" > "${complete_file}"
        echo "DATASET264_DIRECT_TRANSFER_PASS part=${part}"
    else
        echo "DATASET264_DIRECT_TRANSFER_FAIL part=${part} rc=${rc}" >&2
    fi
    exit "${rc}"
}

start_part() {
    local part="$1"
    local source
    source=$(part_source "${part}")
    [[ -d "${source}" ]] || die "missing source dataset part: ${source}"
    if part_is_running "${part}"; then
        echo "DATASET264_DIRECT_TRANSFER_ALREADY_RUNNING part=${part} pid=$(cat "${TRANSFER_ROOT}/${part}.pid")"
        return 0
    fi

    local self
    self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
    local log_file="${TRANSFER_ROOT}/${part}.log"
    printf '\nDATASET264_DIRECT_TRANSFER_LAUNCH part=%s launched_at=%s\n' \
        "${part}" "$(date -u +%FT%TZ)" >> "${log_file}"
    nohup bash "${self}" worker "${part}" >> "${log_file}" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "${pid}" > "${TRANSFER_ROOT}/${part}.pid"
    sleep 2
    part_is_running "${part}" || {
        tail -50 "${log_file}" >&2
        die "transfer worker exited during launch: ${part}"
    }
    echo "DATASET264_DIRECT_TRANSFER_LAUNCHED part=${part} pid=${pid} log=${log_file}"
}

validate_target() {
    require_transfer_contract
    ssh "${SSH_OPTIONS[@]}" "${UHOST_USER}@${UHOST_HOST}" \
        bash -s -- "${UHOST_DATA_ROOT}" "${DATASET_NAME}" "${UHOST_PROJECT_ROOT}" <<'REMOTE'
set -euo pipefail

data_root="$1"
dataset_name="$2"
project_root="$3"
raw="${data_root}/nnUNet_raw/${dataset_name}"
pre_root="${data_root}/nnUNet_preprocessed/${dataset_name}"
pre="${pre_root}/nnUNetPlans_3d_fullres"

for path in \
    "${raw}/dataset.json" "${raw}/g2_fixed_split.json" "${raw}/g2_integrity_report.json" \
    "${raw}/g2_materialization_manifest.csv" "${pre_root}/nnUNetPlans.json"; do
    [[ -s "${path}" ]] || { echo "missing ${path}" >&2; exit 1; }
done

count_files() {
    find "$1" -maxdepth 1 -type f ${2:-} | wc -l | tr -d ' '
}

images_tr=$(count_files "${raw}/imagesTr")
labels_tr=$(count_files "${raw}/labelsTr")
images_ts=$(count_files "${raw}/imagesTs")
labels_ts=$(count_files "${raw}/labelsTs")
pkl=$(find "${pre}" -maxdepth 1 -type f -name '*.pkl' | wc -l | tr -d ' ')
data_b2nd=$(find "${pre}" -maxdepth 1 -type f -name '*.b2nd' ! -name '*_seg.b2nd' | wc -l | tr -d ' ')
seg_b2nd=$(find "${pre}" -maxdepth 1 -type f -name '*_seg.b2nd' | wc -l | tr -d ' ')

[[ "${images_tr}" == 4552 ]]
[[ "${labels_tr}" == 1138 ]]
[[ "${images_ts}" == 416 ]]
[[ "${labels_ts}" == 104 ]]
[[ "${pkl}" == 1138 ]]
[[ "${data_b2nd}" == 1138 ]]
[[ "${seg_b2nd}" == 1138 ]]

python_bin="${project_root}/../envs/s2_met_aug_h20/bin/python"
[[ -x "${python_bin}" ]] || python_bin=python3
"${python_bin}" - "${raw}" <<'PY'
import csv
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
integrity = json.loads((raw / "g2_integrity_report.json").read_text())
assert integrity["passed"] is True, integrity
assert integrity["expected_cases"] == 1242, integrity
assert integrity["checked_cases"] == 1242, integrity

splits = json.loads((raw / "g2_fixed_split.json").read_text())
assert isinstance(splits, list) and len(splits) == 1, type(splits)
counts = {key: len(splits[0][key]) for key in ("train", "val", "test")}
assert counts == {"train": 1035, "val": 103, "test": 104}, counts

with (raw / "g2_materialization_manifest.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
overridden = len({
    row["nnunet_case_id"]
    for row in rows
    if row.get("completion_raw_id", "").strip()
})
assert overridden == 212, overridden

dataset = json.loads((raw / "dataset.json").read_text())
assert dataset["numTraining"] == 1138, dataset.get("numTraining")
print("DATASET264_METADATA_PASS included_cases=1242 completion_paths_overridden=212 split=1035/103/104")
PY

available_kib=$(df -Pk "${data_root}" | awk 'NR==2 {print $4}')
(( available_kib >= 150 * 1024 * 1024 ))
du -sh "${raw}" "${pre_root}"
echo "DATASET264_DIRECT_TARGET_PASS imagesTr=${images_tr} labelsTr=${labels_tr} imagesTs=${images_ts} labelsTs=${labels_ts} pkl=${pkl} b2nd=${data_b2nd} seg_b2nd=${seg_b2nd}"
REMOTE
}

case "${ACTION}" in
    start)
        require_transfer_contract
        ssh "${SSH_OPTIONS[@]}" "${UHOST_USER}@${UHOST_HOST}" \
            "mkdir -p '${UHOST_DATA_ROOT}/nnUNet_raw' '${UHOST_DATA_ROOT}/nnUNet_preprocessed'"
        start_part nnUNet_raw
        start_part nnUNet_preprocessed
        ;;
    status)
        status_part nnUNet_raw
        status_part nnUNet_preprocessed
        ;;
    worker)
        [[ -n "${PART}" ]] || die "worker requires a transfer part"
        run_worker "${PART}"
        ;;
    validate)
        validate_target
        ;;
    *)
        echo "Usage: $0 {start|status|validate}" >&2
        exit 2
        ;;
esac
