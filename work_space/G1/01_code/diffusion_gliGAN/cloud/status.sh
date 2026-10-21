#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
RUN_ROOT="${RUN_ROOT:-/root/brats2026/runs/g1_diffusion_v3}"
LOGDIR="${LOGDIR:-brats2026_diffusion_v3_edm_zscore}"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
for MODALITY in t1c t1n t2w t2f; do
  PID_FILE="${RUN_ROOT}/pids/${MODALITY}.pid"
  PID=""
  STATE="not_started"
  if [[ -s "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
      STATE="running"
    else
      STATE="stopped"
    fi
  fi
  WEIGHTS_DIR="${RUN_ROOT}/checkpoints/${LOGDIR}/${MODALITY}/weights"
  STEP=""
  if [[ -d "${WEIGHTS_DIR}" ]]; then
    STEP=$(find "${WEIGHTS_DIR}" -maxdepth 1 -type f -name 'diffusion_*.pt' \
      | sed -nE 's|.*/diffusion_([0-9]+)\.pt$|\1|p' | sort -n | tail -1)
  fi
  echo "modality=${MODALITY} state=${STATE} pid=${PID:-none} latest_step=${STEP:-none}"
done
