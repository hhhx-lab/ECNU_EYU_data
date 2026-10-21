#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT}"
: "${GENERATION_PID:?set GENERATION_PID}"
: "${PYTHON_BIN:?set PYTHON_BIN}"

GENERATION_ROOT="${GENERATION_ROOT:-${RUN_ROOT}/eval/150000_full94_a800_20260721}"
GENERATION_LOG="${GENERATION_LOG:-${RUN_ROOT}/logs/evaluate_150000_full94_a800_20260721.log}"
SELECTION_JSON="${SELECTION_JSON:-${RUN_ROOT}/splits/full_eval_150000_v2/full_eval_cohort_summary.json}"
CHECKPOINT_INVENTORY="${CHECKPOINT_INVENTORY:-${RUN_ROOT}/splits/checkpoint_inventory.csv}"
QC_SCRIPT="${QC_SCRIPT:-${RUN_ROOT}/g2/code/g2_diffusion_checkpoint_qc.py}"
QC_OUTPUT="${QC_OUTPUT:-${RUN_ROOT}/g2/qc/diffusion_checkpoint_full94_150000_a800_20260721}"

echo "WATCH_START=$(date -Is)"
while kill -0 "${GENERATION_PID}" 2>/dev/null; do
  sleep 60
done
echo "GENERATION_PROCESS_EXITED=$(date -Is)"

test -s "${GENERATION_ROOT}/metrics.json"
test -s "${GENERATION_ROOT}/generation_manifest.csv"
if grep -qiE 'Traceback|CUDA out of memory|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "${GENERATION_LOG}"; then
  echo "GENERATION_ERROR_PATTERN_FOUND" >&2
  exit 21
fi
test ! -e "${QC_OUTPUT}"

"${PYTHON_BIN}" -u "${QC_SCRIPT}" \
  --generation-manifest "${GENERATION_ROOT}/generation_manifest.csv" \
  --selection-json "${SELECTION_JSON}" \
  --checkpoint-inventory "${CHECKPOINT_INVENTORY}" \
  --output-root "${QC_OUTPUT}" \
  --expected-cases 94

echo "G2_QC_COMPLETE=$(date -Is)"
