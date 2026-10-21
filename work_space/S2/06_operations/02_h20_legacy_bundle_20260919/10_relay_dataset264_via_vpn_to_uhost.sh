#!/usr/bin/env bash

set -euo pipefail

S2_UHOST="${S2_UHOST:-s2-h20}"
ECNU_HOST="${ECNU_HOST:-59.78.189.132}"
ECNU_PORT="${ECNU_PORT:-2323}"
ECNU_USER="${ECNU_USER:-zqchen}"
ECNU_PROJECT_ROOT="${ECNU_PROJECT_ROOT:-/public/home/zqchen/projects/ECNU_EYU_data}"
REMOTE_FORWARD_PORT="${REMOTE_FORWARD_PORT:-22323}"
UHOST_DATA_ROOT="${UHOST_DATA_ROOT:-/root/brats2026/data/s2_dataset264}"
DATASET_NAME=Dataset264_BraTS2026_MET_Completion

if ! nc -z -w 8 "${ECNU_HOST}" "${ECNU_PORT}" 2>/dev/null; then
    echo "ECNU is unreachable. Connect Cisco AnyConnect to vpn-ct.ecnu.edu.cn first." >&2
    exit 1
fi
if ! ssh "${S2_UHOST}" "command -v rsync >/dev/null"; then
    echo "rsync is missing on UHost; install it only after explicit approval." >&2
    exit 1
fi

ssh -N -T \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=6 \
    -R "127.0.0.1:${REMOTE_FORWARD_PORT}:${ECNU_HOST}:${ECNU_PORT}" \
    "${S2_UHOST}" &
TUNNEL_PID=$!
cleanup() {
    kill "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in {1..20}; do
    if ssh "${S2_UHOST}" "timeout 2 bash -c '</dev/tcp/127.0.0.1/${REMOTE_FORWARD_PORT}'" 2>/dev/null; then
        break
    fi
    sleep 1
done
if ! ssh "${S2_UHOST}" "timeout 2 bash -c '</dev/tcp/127.0.0.1/${REMOTE_FORWARD_PORT}'" 2>/dev/null; then
    echo "ECNU reverse tunnel did not become ready." >&2
    exit 1
fi

ECNU_STORAGE_ROOT="${ECNU_PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency"
REMOTE_RECEIVER="/root/brats2026/ECNU_EYU_data/work_space/S2/slurm/dataset264_rsync_receiver.sh"

# Agent forwarding keeps private keys on the local Mac. If ECNU requires a
# password, OpenSSH prompts interactively; the script never stores it.
ssh -t -A "${S2_UHOST}" \
    bash "${REMOTE_RECEIVER}" \
    "${ECNU_USER}" "${REMOTE_FORWARD_PORT}" "${ECNU_STORAGE_ROOT}" \
    "${UHOST_DATA_ROOT}" "${DATASET_NAME}"
