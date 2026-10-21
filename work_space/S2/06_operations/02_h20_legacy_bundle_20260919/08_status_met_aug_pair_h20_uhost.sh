#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/s2_uhost_common.sh"

PID_DIR="${ROUTE_ROOT}/pids"
LOG_DIR="${ROUTE_ROOT}/logs"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader
df -h "${S2_DATA_ROOT%/*}"

for arm in control route_a; do
    pid_file="${PID_DIR}/${arm}.pid"
    log_file="${LOG_DIR}/train_${arm}.log"
    state=not_started
    pid=none
    if [[ -s "${pid_file}" ]]; then
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            state=running
        else
            state=stopped
        fi
    fi
    echo "arm=${arm} state=${state} pid=${pid}"
    if [[ -s "${log_file}" ]]; then
        tail -n 400 "${log_file}" \
            | grep -E 'Epoch [0-9]+|Epoch time:|train_loss|val_loss|MET_AUG_|Traceback|CUDA out of memory|nan|inf' \
            | tail -20 || true
    fi
done
