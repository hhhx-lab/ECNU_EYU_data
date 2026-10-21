#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODALITY="${1:-}"
EXPECTED_PID="${2:-}"
DELAY_SECONDS="${3:-}"
case "${MODALITY}" in
  t1c|t1n|t2w|t2f) ;;
  *) echo "Usage: $0 {t1c|t1n|t2w|t2f} expected_pid delay_seconds" >&2; exit 2 ;;
esac
if ! [[ "${EXPECTED_PID}" =~ ^[0-9]+$ && "${DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "expected_pid and delay_seconds must be integers" >&2
  exit 2
fi

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
RUN_ROOT="${RUN_ROOT:-/root/brats2026/runs/g1_diffusion_v3}"
PID_FILE="${RUN_ROOT}/pids/${MODALITY}.pid"

echo "WATCHDOG_ARMED modality=${MODALITY} expected_pid=${EXPECTED_PID} delay=${DELAY_SECONDS}"
sleep "${DELAY_SECONDS}"

if [[ ! -s "${PID_FILE}" ]] || [[ "$(cat "${PID_FILE}")" != "${EXPECTED_PID}" ]]; then
  echo "WATCHDOG_STALE modality=${MODALITY}; pid file changed"
  exit 0
fi
if ! kill -0 "${EXPECTED_PID}" 2>/dev/null; then
  echo "WATCHDOG_NOOP modality=${MODALITY}; process already stopped"
  exit 0
fi

echo "WATCHDOG_DEADLINE_REACHED modality=${MODALITY}; requesting atomic checkpoint"
exec "${SCRIPT_DIR}/stop_training.sh" "${MODALITY}"
