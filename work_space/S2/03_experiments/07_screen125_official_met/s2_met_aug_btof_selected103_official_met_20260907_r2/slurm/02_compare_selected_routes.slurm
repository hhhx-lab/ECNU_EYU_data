#!/usr/bin/env bash
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/logs/%x_%j.err

set -euo pipefail
umask 027

ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
FAILED="${ROOT}/evidence/PAIRED_COMPARISON_FAILED.json"
VALIDATION="${ROOT}/evidence/PAIRED_COMPARISON_VALIDATION.json"
COMPLETE="${ROOT}/evidence/PAIRED_COMPARISON_COMPLETE.ok"
OUT="${ROOT}/cohort/fixed103/paired_comparison"

write_failure() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" FAILED="$FAILED" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY' || true
import json,os
from datetime import datetime,timezone
from pathlib import Path
p=Path(os.environ["FAILED"])
payload={"schema_version":1,"status":"failed","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","job_id":os.environ.get("SLURM_JOB_ID"),"exit_code":int(os.environ["RC"]),"failed_at_utc":datetime.now(timezone.utc).isoformat()}
try:
    with p.open("x",encoding="utf-8") as h:
        json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
except FileExistsError:
    pass
PY
  exit "$rc"
}
trap write_failure ERR INT TERM

cd "$ROOT"
test -f evidence/REMOTE_STATIC_PREFLIGHT.json
test ! -e "$FAILED"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
test ! -e "$OUT"
sha256sum -c STATIC_SHA256SUMS.txt
for route in B C40 D E_SNFH F; do
  test -f "evidence/${route}_EVALUATION_COMPLETE.ok"
  test -f "evidence/${route}_EVALUATION_VALIDATION.json"
  test -f "cohort/fixed103/models/${route}/evaluation/leaderboard_metrics.csv"
done

export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" "${ROOT}/runtime/compare_paired_met.py" \
  --input "B=${ROOT}/cohort/fixed103/models/B/evaluation/leaderboard_metrics.csv" \
  --input "C40=${ROOT}/cohort/fixed103/models/C40/evaluation/leaderboard_metrics.csv" \
  --input "D=${ROOT}/cohort/fixed103/models/D/evaluation/leaderboard_metrics.csv" \
  --input "E_SNFH=${ROOT}/cohort/fixed103/models/E_SNFH/evaluation/leaderboard_metrics.csv" \
  --input "F=${ROOT}/cohort/fixed103/models/F/evaluation/leaderboard_metrics.csv" \
  --output-root "$OUT" --expected-count 103 \
  --bootstrap 20000 --seed 20260831 \
  --scope screen125_selected_checkpoint_fixed103_official_met_exploratory

ROOT="$ROOT" OUT="$OUT" VALIDATION="$VALIDATION" COMPLETE="$COMPLETE" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY'
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
root=Path(os.environ["ROOT"]); out=Path(os.environ["OUT"])
paired=json.loads((out/"PAIRED_COMPARISON.json").read_text())
assert paired["status"]=="pass" and paired["case_count"]==103
assert list(paired["methods"])==["B","C40","D","E_SNFH","F"]
assert len(paired["pairs"])==10
assert paired["bootstrap"]=={"replicates":20000,"seed":20260831}
routes={}
for route in ("B","C40","D","E_SNFH","F"):
    val=json.loads((root/f"evidence/{route}_EVALUATION_VALIDATION.json").read_text())
    assert val["status"]=="pass" and val["case_count"]==103
    routes[route]={"selected_epoch":val["selected_epoch"],"checkpoint_sha256":val["checkpoint_sha256"],"metrics_csv_sha256":val["metrics_csv_sha256"]}
payload={"schema_version":1,"status":"pass","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","job_id":os.environ.get("SLURM_JOB_ID"),"completed_at_utc":datetime.now(timezone.utc).isoformat(),"case_count":103,"method_count":5,"pair_count":10,"bootstrap_replicates":20000,"bootstrap_seed":20260831,"routes":routes,"paired_comparison_sha256":sha(out/"PAIRED_COMPARISON.json"),"interpretation":"exploratory because the same fixed103 cohort was used for checkpoint selection and reporting","locked_test_used":False}
with Path(os.environ["VALIDATION"]).open("x",encoding="utf-8") as h:
    json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
with Path(os.environ["COMPLETE"]).open("x",encoding="ascii") as h:
    h.write("status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n")
print(json.dumps(payload,sort_keys=True))
PY
trap - ERR INT TERM
