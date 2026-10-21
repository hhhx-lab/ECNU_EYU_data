#!/usr/bin/env bash
#SBATCH --job-name=s2t7mf1r3
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r3_evaluation_only/logs/table7_eval_only_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r3_evaluation_only/logs/table7_eval_only_%j.err

set -euo pipefail
umask 027

PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r3_evaluation_only"
R2_ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100"
REFERENCE="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_raw/Dataset264_BraTS2026_MET_Completion/labelsTs"
TEST_LIST="${PROJECT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository/data/splits/completion_warmstart/test_locked.txt"
MAPPING_CSV="${PROJECT}/work_space/G2/results/manifests/nnunet_case_mapping_master.csv"
EVAL_ENV=/public/home/zqchen/.conda/envs/brats_eval
EVAL_PY="${EVAL_ENV}/bin/python"
FAILED="${ROOT}/evidence/TABLE7_EVALUATION_ONLY_FAILED.json"
STARTED="${ROOT}/evidence/TABLE7_EVALUATION_ONLY_STARTED.json"
MAPPING_AUDIT="${ROOT}/evidence/CASE_ID_MAPPING_AND_SYMLINK_AUDIT.json"
VALIDATION="${ROOT}/evidence/TABLE7_DISJOINT_MICROF1_VALIDATION.json"
COMPLETE="${ROOT}/evidence/TABLE7_DISJOINT_MICROF1_COMPLETE.ok"
STAGE=bootstrap

fail_stage() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" STAGE="$STAGE" FAILED="$FAILED" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$EVAL_PY" - <<'PY' || true
import json, os
from datetime import datetime, timezone
from pathlib import Path
payload = {
    "schema_version": 1,
    "status": "failed",
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "evaluation_only": True,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "stage": os.environ["STAGE"],
    "exit_code": int(os.environ["RC"]),
    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
}
try:
    with Path(os.environ["FAILED"]).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
except FileExistsError:
    pass
PY
  exit "$rc"
}
trap fail_stage ERR INT TERM

cd "$ROOT"
test -f evidence/REMOTE_STATIC_PREFLIGHT.json
test -f evidence/TABLE7_EVALUATION_ONLY_LAUNCH.json
test ! -e "$FAILED"
test ! -e "$STARTED"
test ! -e "$MAPPING_AUDIT"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
test ! -e views
test ! -e official_reproduction
test ! -e disjoint_microf1
sha256sum -c STATIC_SHA256SUMS.txt

module purge
module load apps/envs/miniconda3/25.5.1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

STAGE=started_evidence
STARTED="$STARTED" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$EVAL_PY" - <<'PY'
import json, os, socket
from datetime import datetime, timezone
from pathlib import Path
payload = {
    "schema_version": 1,
    "status": "pass",
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "evaluation_only": True,
    "training": False,
    "inference": False,
    "gpu_requested": False,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "hostname": socket.gethostname(),
    "case_count": 104,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
}
with Path(os.environ["STARTED"]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
json.loads(Path(os.environ["STARTED"]).read_text(encoding="utf-8"))
print(json.dumps(payload, sort_keys=True))
PY

STAGE=build_mapped_views
"$EVAL_PY" "${ROOT}/runtime/build_case_views.py" \
  --test-list "$TEST_LIST" \
  --mapping-csv "$MAPPING_CSV" \
  --r2-root "$R2_ROOT" \
  --reference-root "$REFERENCE" \
  --view-root "${ROOT}/views" \
  --audit-json "$MAPPING_AUDIT" \
  --expected-count 104 \
  --expected-test-sha256 d763fb4d1eb798a46670079a0e148ba5ce0e0cc15de174f93effb1c741cb114b \
  --expected-mapping-sha256 b1ff36ea927a3d5c8d4aa976ce5533d320e16c7801a9a30db22e8ecbb6d64672 \
  --expected-pair-sha256 78dd2710043171dd1de75043ff5d9f7be7ff9a226b71191a11c17a6ec493ffaf \
  --expected-checkpoint-sha256 4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267 \
  > "${ROOT}/logs/build_case_views_${SLURM_JOB_ID}.log" 2>&1

STAGE=official_evaluation
mkdir "${ROOT}/official_reproduction"
{
  "$EVAL_PY" --version
  "$EVAL_PY" -c 'import importlib.metadata as m; print("BraTS-evaluation="+m.version("BraTS-evaluation")); print("panoptica="+m.version("panoptica")); print("numpy="+m.version("numpy")); print("scipy="+m.version("scipy")); print("nibabel="+m.version("nibabel"))'
  sha256sum "$EVAL_ENV/bin/brats-evaluate" "$EVAL_ENV/bin/brats-parse-metrics"
} > "${ROOT}/official_reproduction/evaluation_environment.txt"

"$EVAL_ENV/bin/brats-evaluate" \
  --config mets \
  --ref_path "${ROOT}/views/reference" \
  --pred_path "${ROOT}/views/prediction" \
  --summary_json "${ROOT}/official_reproduction/panoptica_evaluation_summary.json" \
  > "${ROOT}/logs/official_evaluate_${SLURM_JOB_ID}.log" 2>&1

for threshold in 27 50; do
  "$EVAL_ENV/bin/brats-parse-metrics" mets \
    --json_path "${ROOT}/official_reproduction/panoptica_evaluation_summary.json" \
    --vol_threshold "$threshold" \
    --overlap_threshold 0.2 \
    --output_csv_path "${ROOT}/official_reproduction/leaderboard_metrics_threshold${threshold}.csv" \
    > "${ROOT}/logs/official_parse_t${threshold}_${SLURM_JOB_ID}.log" 2>&1
done

STAGE=disjoint_microf1
"$EVAL_PY" "${ROOT}/runtime/disjoint_microf1.py" \
  --prediction-root "${ROOT}/views/prediction" \
  --reference-root "${ROOT}/views/reference" \
  --output-root "${ROOT}/disjoint_microf1" \
  --expected-count 104 \
  --overlap-threshold 0.2 \
  --thresholds 27,50 \
  > "${ROOT}/logs/disjoint_microf1_${SLURM_JOB_ID}.log" 2>&1

STAGE=final_validation
"$EVAL_PY" "${ROOT}/runtime/validate_run.py" \
  --root "$ROOT" \
  --mode final \
  --job-id "${SLURM_JOB_ID}" \
  > "${ROOT}/logs/final_validation_${SLURM_JOB_ID}.log" 2>&1

STAGE=complete_evidence
COMPLETE="$COMPLETE" VALIDATION="$VALIDATION" SLURM_JOB_ID="${SLURM_JOB_ID}" "$EVAL_PY" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
validation = Path(os.environ["VALIDATION"])
payload = {
    "status": "pass",
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "evaluation_only": True,
    "job_id": os.environ["SLURM_JOB_ID"],
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "validation_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
}
with Path(os.environ["COMPLETE"]).open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
json.loads(Path(os.environ["COMPLETE"]).read_text(encoding="utf-8"))
PY

trap - ERR INT TERM
echo "TABLE7_EVALUATION_ONLY_R3_PASS job_id=${SLURM_JOB_ID}"
