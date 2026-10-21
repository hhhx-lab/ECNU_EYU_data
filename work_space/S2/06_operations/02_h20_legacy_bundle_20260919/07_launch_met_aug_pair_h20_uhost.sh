#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

s2_uhost_activate_runtime
s2_uhost_require_file "${ROUTE_ROOT}/TRAINING_SMOKE_APPROVED.ok" "approved training smoke"

CONTROL_GPU="${CONTROL_GPU:-0}"
ROUTE_A_GPU="${ROUTE_A_GPU:-1}"
if [[ "${CONTROL_GPU}" == "${ROUTE_A_GPU}" ]]; then
    echo "CONTROL_GPU and ROUTE_A_GPU must be different." >&2
    exit 2
fi

PID_DIR="${ROUTE_ROOT}/pids"
LOG_DIR="${ROUTE_ROOT}/logs"
LOCK_DIR="${PID_DIR}/.pair_launch.lock"
mkdir -p "${PID_DIR}" "${LOG_DIR}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Another pair launcher is active: ${LOCK_DIR}" >&2
    exit 3
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

launch_arm() {
    local arm="$1"
    local gpu="$2"
    local pid_file="${PID_DIR}/${arm}.pid"
    local log_file="${LOG_DIR}/train_${arm}.log"
    local complete_marker="${ROUTE_ROOT}/PAIR_${arm}_COMPLETE.ok"

    if [[ -s "${complete_marker}" ]]; then
        echo "PAIR_ALREADY_COMPLETE arm=${arm} marker=${complete_marker}"
        return 0
    fi
    if [[ -s "${pid_file}" ]]; then
        local old_pid
        old_pid=$(cat "${pid_file}")
        if kill -0 "${old_pid}" 2>/dev/null; then
            echo "Pair arm is already running: arm=${arm} pid=${old_pid}" >&2
            return 1
        fi
    fi

    nohup env \
        S2_UHOST_ENV_FILE="${S2_UHOST_ENV_FILE}" \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        S2_PAIR_ARM="${arm}" \
        PYTHONUNBUFFERED=1 \
        bash "${SCRIPT_DIR}/06_train_met_aug_pair_h20_uhost.slurm" \
        > "${log_file}" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "${pid}" > "${pid_file}"
    sleep 3
    if ! kill -0 "${pid}" 2>/dev/null; then
        tail -100 "${log_file}" >&2
        return 1
    fi
    echo "PAIR_LAUNCHED arm=${arm} physical_gpu=${gpu} pid=${pid} log=${log_file}"
}

# Start the control first so its one-time custom-trainer copy completes before
# Route A reads the same isolated site-packages directory.
launch_arm control "${CONTROL_GPU}"
sleep 30
launch_arm route_a "${ROUTE_A_GPU}"

echo "S2_MET_AUG_PAIR_LAUNCH_PASS"
