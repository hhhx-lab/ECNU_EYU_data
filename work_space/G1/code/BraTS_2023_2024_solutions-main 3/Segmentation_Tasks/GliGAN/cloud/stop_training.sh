#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODALITY="${1:-}"
case "${MODALITY}" in
  t1c|t1n|t2w|t2f) ;;
  *) echo "Usage: $0 {t1c|t1n|t2w|t2f}" >&2; exit 2 ;;
esac

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
RUN_ROOT="${RUN_ROOT:-/root/brats2026/runs/g1_diffusion_v3}"
PID_FILE="${RUN_ROOT}/pids/${MODALITY}.pid"
test -s "${PID_FILE}"
PID=$(cat "${PID_FILE}")
GRACE_SECONDS="${TRAIN_STOP_GRACE_SECONDS:-1800}"
if ! [[ "${GRACE_SECONDS}" =~ ^[0-9]+$ ]] || (( GRACE_SECONDS < 1 )); then
  echo "TRAIN_STOP_GRACE_SECONDS must be a positive integer" >&2
  exit 2
fi

if ! kill -0 "${PID}" 2>/dev/null; then
  echo "Process is not running: modality=${MODALITY} pid=${PID}"
  exit 0
fi

kill -TERM "${PID}"
echo "Sent SIGTERM; waiting for the current step and atomic checkpoint: pid=${PID}"
for ((SECOND = 1; SECOND <= GRACE_SECONDS; SECOND++)); do
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "TRAINING_STOPPED_SAFELY modality=${MODALITY}"
    exit 0
  fi
  sleep 1
done
echo "Process still exists after ${GRACE_SECONDS} seconds. Do not use SIGKILL before checking the log." >&2
exit 1
