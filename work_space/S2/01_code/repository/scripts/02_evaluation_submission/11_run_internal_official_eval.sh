#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 ENV_DIR EVAL_ROOT [EXPECTED_COUNT]" >&2
  exit 2
fi

ENV_DIR=$1
EVAL_ROOT=$2
EXPECTED_COUNT=${3:-103}
PRED_ROOT="${EVAL_ROOT}/prediction"
REF_ROOT="${EVAL_ROOT}/reference"
SUMMARY_JSON="${EVAL_ROOT}/panoptica_evaluation_summary.json"
METRICS_CSV="${EVAL_ROOT}/leaderboard_metrics.csv"
EVALUATE_LOG="${EVAL_ROOT}/brats_evaluate.log"
PARSE_LOG="${EVAL_ROOT}/brats_parse_metrics.log"

PYTHON="${ENV_DIR}/bin/python"
EVALUATE="${ENV_DIR}/bin/brats-evaluate"
PARSE="${ENV_DIR}/bin/brats-parse-metrics"

for executable in "${PYTHON}" "${EVALUATE}" "${PARSE}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Missing executable: ${executable}" >&2
    exit 3
  fi
done

for directory in "${PRED_ROOT}" "${REF_ROOT}"; do
  if [[ ! -d "${directory}" ]]; then
    echo "Missing evaluation directory: ${directory}" >&2
    exit 4
  fi
done

if [[ -e "${SUMMARY_JSON}" || -e "${METRICS_CSV}" ]]; then
  echo "Refusing to overwrite existing evaluation outputs in ${EVAL_ROOT}" >&2
  exit 5
fi

pred_count=$(find "${PRED_ROOT}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
ref_count=$(find "${REF_ROOT}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l | tr -d ' ')
if [[ "${pred_count}" != "${EXPECTED_COUNT}" || "${ref_count}" != "${EXPECTED_COUNT}" ]]; then
  echo "Evaluation count mismatch: pred=${pred_count} ref=${ref_count} expected=${EXPECTED_COUNT}" >&2
  exit 6
fi

"${PYTHON}" - "${PRED_ROOT}" "${REF_ROOT}" "${EXPECTED_COUNT}" <<'PY'
import sys
from pathlib import Path

prediction_root = Path(sys.argv[1])
reference_root = Path(sys.argv[2])
expected_count = int(sys.argv[3])
prediction_names = {path.name for path in prediction_root.glob("*.nii.gz")}
reference_names = {path.name for path in reference_root.glob("*.nii.gz")}
if prediction_names != reference_names:
    raise SystemExit(
        "Prediction/reference filename mismatch: "
        f"missing={sorted(reference_names - prediction_names)[:10]} "
        f"extra={sorted(prediction_names - reference_names)[:10]}"
    )
if len(prediction_names) != expected_count:
    raise SystemExit(
        f"Unique filename count mismatch: actual={len(prediction_names)} expected={expected_count}"
    )
PY

{
  printf 'config=mets\n'
  printf 'vol_threshold=27\n'
  printf 'overlap_threshold=0.2\n'
  printf 'expected_count=%s\n' "${EXPECTED_COUNT}"
  printf 'prediction_root=%s\n' "${PRED_ROOT}"
  printf 'reference_root=%s\n' "${REF_ROOT}"
  printf 'summary_json=%s\n' "${SUMMARY_JSON}"
  printf 'metrics_csv=%s\n' "${METRICS_CSV}"
} > "${EVAL_ROOT}/evaluation_contract.txt"

{
  "${PYTHON}" --version
  "${PYTHON}" -c 'import importlib.metadata as m; print("BraTS-evaluation=" + m.version("BraTS-evaluation")); print("panoptica=" + m.version("panoptica"))'
  sha256sum "${PYTHON}" "${EVALUATE}" "${PARSE}"
} > "${EVAL_ROOT}/evaluation_environment.txt"

"${EVALUATE}" \
  --config mets \
  --ref_path "${REF_ROOT}" \
  --pred_path "${PRED_ROOT}" \
  --summary_json "${SUMMARY_JSON}" \
  2>&1 | tee "${EVALUATE_LOG}"

"${PYTHON}" - "${SUMMARY_JSON}" "${EXPECTED_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected_count = int(sys.argv[2])
data = json.loads(summary_path.read_text(encoding="utf-8"))
missings = data.get("missings")
metrics = data.get("metrics")
if missings != []:
    raise SystemExit(f"Evaluator reported missing cases: {missings}")
if not isinstance(metrics, list) or len(metrics) != expected_count:
    raise SystemExit(
        f"Evaluator metric count mismatch: actual={len(metrics) if isinstance(metrics, list) else 'invalid'} "
        f"expected={expected_count}"
    )
errors = [item for item in metrics if isinstance(item, dict) and "error" in item]
if errors:
    raise SystemExit(f"Evaluator reported subject errors: {errors[:3]}")
PY

"${PARSE}" mets \
  --json_path "${SUMMARY_JSON}" \
  --vol_threshold 27 \
  --overlap_threshold 0.2 \
  --output_csv_path "${METRICS_CSV}" \
  2>&1 | tee "${PARSE_LOG}"

"${PYTHON}" - "${METRICS_CSV}" "${EXPECTED_COUNT}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
expected_count = int(sys.argv[2])
with csv_path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    columns = set(reader.fieldnames or [])

required_columns = {"subject_id"}
for region in ("et", "rc", "tc", "wt"):
    required_columns.update(
        {
            f"lesionwise_dsc_mean_{region}",
            f"lesionwise_nsd_mean_{region}",
            f"small_instance_tp_{region}",
            f"small_instance_fn_{region}",
            f"small_instance_fp_{region}",
            f"small_instance_f1_{region}",
        }
    )
missing_columns = sorted(required_columns - columns)
if missing_columns:
    raise SystemExit(f"Missing required metric columns: {missing_columns}")

expected_rows = expected_count + 3
if len(rows) != expected_rows:
    raise SystemExit(f"CSV row count mismatch: actual={len(rows)} expected={expected_rows}")
summary_ids = [row["subject_id"] for row in rows[-3:]]
if summary_ids != ["mean", "std", "median"]:
    raise SystemExit(f"Unexpected summary rows: {summary_ids}")
subject_ids = [row["subject_id"] for row in rows[:-3]]
if len(subject_ids) != len(set(subject_ids)):
    raise SystemExit("Duplicate subject IDs in metrics CSV")
PY

{
  printf 'status=pass\n'
  printf 'completed_at_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  sha256sum "${SUMMARY_JSON}" "${METRICS_CSV}"
} > "${EVAL_ROOT}/EVALUATION_COMPLETE.ok"

echo "S2_INTERNAL_OFFICIAL_EVAL_PASS root=${EVAL_ROOT} cases=${EXPECTED_COUNT}"
