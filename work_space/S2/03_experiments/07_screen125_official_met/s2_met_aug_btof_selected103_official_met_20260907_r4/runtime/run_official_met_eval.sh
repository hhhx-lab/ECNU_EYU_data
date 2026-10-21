#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 ENV_DIR PRED_ROOT REF_ROOT OUTPUT_ROOT EXPECTED_COUNT" >&2
  exit 2
fi

ENV_DIR=$1
PRED_ROOT=$2
REF_ROOT=$3
OUTPUT_ROOT=$4
EXPECTED_COUNT=$5
PYTHON="${ENV_DIR}/bin/python"
EVALUATE="${ENV_DIR}/bin/brats-evaluate"
PARSE="${ENV_DIR}/bin/brats-parse-metrics"

for executable in "${PYTHON}" "${EVALUATE}" "${PARSE}"; do
  [[ -x "${executable}" ]] || { echo "Missing executable: ${executable}" >&2; exit 3; }
done
for directory in "${PRED_ROOT}" "${REF_ROOT}"; do
  [[ -d "${directory}" ]] || { echo "Missing NIfTI directory: ${directory}" >&2; exit 4; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Exclusive evaluation root exists: ${OUTPUT_ROOT}" >&2; exit 5; }
mkdir -p "${OUTPUT_ROOT}"

SUMMARY_JSON="${OUTPUT_ROOT}/panoptica_evaluation_summary.json"
METRICS_CSV="${OUTPUT_ROOT}/leaderboard_metrics.csv"
EVALUATE_LOG="${OUTPUT_ROOT}/brats_evaluate.log"
PARSE_LOG="${OUTPUT_ROOT}/brats_parse_metrics.log"

"${PYTHON}" - "${PRED_ROOT}" "${REF_ROOT}" "${EXPECTED_COUNT}" <<'PY'
import sys
from pathlib import Path

pred = Path(sys.argv[1])
ref = Path(sys.argv[2])
expected = int(sys.argv[3])
pred_names = {path.name for path in pred.glob("*.nii.gz")}
ref_names = {path.name for path in ref.glob("*.nii.gz")}
if len(pred_names) != expected or len(ref_names) != expected:
    raise SystemExit(f"count mismatch: pred={len(pred_names)} ref={len(ref_names)} expected={expected}")
if pred_names != ref_names:
    raise SystemExit(
        f"filename mismatch: missing={sorted(ref_names-pred_names)[:10]} extra={sorted(pred_names-ref_names)[:10]}"
    )
PY

{
  printf 'artifact_status=experimental_unvalidated\n'
  printf 'operator_approved=false\n'
  printf 'formal_gate_status=not_run_not_passed\n'
  printf 'config=mets\n'
  printf 'vol_threshold=27\n'
  printf 'overlap_threshold=0.2\n'
  printf 'expected_count=%s\n' "${EXPECTED_COUNT}"
  printf 'prediction_root=%s\n' "${PRED_ROOT}"
  printf 'reference_root=%s\n' "${REF_ROOT}"
} > "${OUTPUT_ROOT}/evaluation_contract.txt"

{
  "${PYTHON}" --version
  "${PYTHON}" - <<'PY'
import importlib.metadata as m

expected = {"BraTS-evaluation": "0.0.8", "panoptica": "2.1.0", "numpy": "1.26.4"}
observed = {name: m.version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"evaluation environment drift: {observed}")
for name, version in observed.items():
    print(f"{name}={version}")
PY
  sha256sum "${PYTHON}" "${EVALUATE}" "${PARSE}"
} > "${OUTPUT_ROOT}/evaluation_environment.txt"

"${EVALUATE}" \
  --config mets \
  --ref_path "${REF_ROOT}" \
  --pred_path "${PRED_ROOT}" \
  --summary_json "${SUMMARY_JSON}" \
  > >(tee "${EVALUATE_LOG}") 2>&1

"${PYTHON}" - "${SUMMARY_JSON}" "${EXPECTED_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("missings") != []:
    raise SystemExit(f"evaluator missing cases: {data.get('missings')}")
metrics = data.get("metrics")
if not isinstance(metrics, list) or len(metrics) != expected:
    raise SystemExit(f"metric count mismatch: {len(metrics) if isinstance(metrics, list) else 'invalid'}")
errors = [item for item in metrics if isinstance(item, dict) and "error" in item]
if errors:
    raise SystemExit(f"subject errors: {errors[:3]}")
PY

"${PARSE}" mets \
  --json_path "${SUMMARY_JSON}" \
  --vol_threshold 27 \
  --overlap_threshold 0.2 \
  --output_csv_path "${METRICS_CSV}" \
  > >(tee "${PARSE_LOG}") 2>&1

"${PYTHON}" - "${METRICS_CSV}" "${EXPECTED_COUNT}" "${OUTPUT_ROOT}" <<'PY'
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

csv_path = Path(sys.argv[1])
expected = int(sys.argv[2])
root = Path(sys.argv[3])
with csv_path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fields = reader.fieldnames or []
if "subject_id" not in fields:
    raise SystemExit("subject_id column missing")
if len(rows) != expected + 3:
    raise SystemExit(f"CSV row count {len(rows)} != {expected + 3}")
if [row["subject_id"] for row in rows[-3:]] != ["mean", "std", "median"]:
    raise SystemExit("summary row order drift")
ids = [row["subject_id"] for row in rows[:-3]]
if len(ids) != len(set(ids)) == expected:
    raise SystemExit("duplicate or missing subject IDs")
required = set()
for region in ("et", "rc", "tc", "wt"):
    required.update(
        {
            f"all_instance_f1_{region}",
            f"large_instance_f1_{region}",
            f"small_instance_f1_{region}",
            f"lesionwise_dsc_mean_{region}",
            f"lesionwise_hd95_mean_{region}",
            f"lesionwise_nsd_mean_{region}",
        }
    )
missing = sorted(required - set(fields))
if missing:
    raise SystemExit(f"required metrics missing: {missing}")
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
summary = root / "panoptica_evaluation_summary.json"
payload = {
    "schema_version": 1,
    "status": "pass",
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "case_count": expected,
    "csv_data_row_count": expected,
    "csv_summary_rows": ["mean", "std", "median"],
    "metric_column_count": len(fields) - 1,
    "summary_json_sha256": sha(summary),
    "metrics_csv_sha256": sha(csv_path),
}
with (root / "EVALUATION_VALIDATION.json").open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
(root / "EVALUATION_COMPLETE.ok").write_text(
    "status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY
