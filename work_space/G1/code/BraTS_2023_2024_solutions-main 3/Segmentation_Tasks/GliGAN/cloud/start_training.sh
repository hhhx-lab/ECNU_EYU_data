#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODALITY="${1:-}"
CUDA_DEVICE="${2:-0}"
case "${MODALITY}" in
  t1c|t1n|t2w|t2f) ;;
  *) echo "Usage: $0 {t1c|t1n|t2w|t2f} [cuda_device]" >&2; exit 2 ;;
esac

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
RUN_ROOT="${RUN_ROOT:-/root/brats2026/runs/g1_diffusion_v3}"
PID_DIR="${RUN_ROOT}/pids"
LOG_DIR="${RUN_ROOT}/logs"
PID_FILE="${PID_DIR}/${MODALITY}.pid"
LOG_FILE="${LOG_DIR}/train_${MODALITY}.log"
WATCHDOG_PID_FILE="${PID_DIR}/${MODALITY}.watchdog.pid"
TRAIN_MAX_RUNTIME_SECONDS="${TRAIN_MAX_RUNTIME_SECONDS:-0}"
TRAIN_STOP_GRACE_SECONDS="${TRAIN_STOP_GRACE_SECONDS:-1800}"
mkdir -p "${PID_DIR}" "${LOG_DIR}"

if ! [[ "${TRAIN_MAX_RUNTIME_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "TRAIN_MAX_RUNTIME_SECONDS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${TRAIN_STOP_GRACE_SECONDS}" =~ ^[0-9]+$ ]] || (( TRAIN_STOP_GRACE_SECONDS < 1 )); then
  echo "TRAIN_STOP_GRACE_SECONDS must be a positive integer" >&2
  exit 2
fi

if [[ -s "${PID_FILE}" ]]; then
  OLD_PID=$(cat "${PID_FILE}")
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Training is already running: modality=${MODALITY} pid=${OLD_PID}"
    exit 3
  fi
fi

nohup env \
  ENV_FILE="${ENV_FILE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  PYTHONUNBUFFERED=1 \
  "${SCRIPT_DIR}/train_modality.sh" "${MODALITY}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" > "${PID_FILE}"
sleep 2
if ! kill -0 "${PID}" 2>/dev/null; then
  tail -80 "${LOG_FILE}" >&2
  exit 1
fi

if (( TRAIN_MAX_RUNTIME_SECONDS > 0 )); then
  WATCHDOG_LOG="${LOG_DIR}/watchdog_${MODALITY}.log"
  nohup env \
    ENV_FILE="${ENV_FILE}" \
    TRAIN_STOP_GRACE_SECONDS="${TRAIN_STOP_GRACE_SECONDS}" \
    "${SCRIPT_DIR}/deadline_watchdog.sh" \
    "${MODALITY}" "${PID}" "${TRAIN_MAX_RUNTIME_SECONDS}" \
    > "${WATCHDOG_LOG}" 2>&1 < /dev/null &
  WATCHDOG_PID=$!
  printf '%s\n' "${WATCHDOG_PID}" > "${WATCHDOG_PID_FILE}"
  echo "TRAINING_WATCHDOG modality=${MODALITY} pid=${WATCHDOG_PID} deadline_seconds=${TRAIN_MAX_RUNTIME_SECONDS}"
fi

echo "TRAINING_LAUNCHED modality=${MODALITY} gpu=${CUDA_DEVICE} pid=${PID} log=${LOG_FILE}"
