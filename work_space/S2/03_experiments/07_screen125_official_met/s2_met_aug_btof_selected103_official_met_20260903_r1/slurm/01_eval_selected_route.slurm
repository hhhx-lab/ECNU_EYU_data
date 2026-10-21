#!/usr/bin/env bash
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1/logs/%x_%j.err

set -euo pipefail
umask 027

ROUTE="${1:?route token is required}"
case "$ROUTE" in
  B|C40|D|E_SNFH|F) ;;
  *) echo "unsupported route: $ROUTE" >&2; exit 2 ;;
esac

PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1"
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
EVAL_ENV=/public/home/zqchen/.conda/envs/brats_eval
S2_REPOSITORY="${PROJECT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
PLAN="${ROOT}/SELECTED103_OFFICIAL_MET_EXECUTION_PLAN_20260903_R1.json"
INPUT="${ROOT}/cohort/fixed103/input"
REFERENCE="${ROOT}/cohort/fixed103/reference"
MODEL_ROOT="${ROOT}/model_snapshots/${ROUTE}"
OUT="${ROOT}/cohort/fixed103/models/${ROUTE}"
STARTED="${ROOT}/evidence/${ROUTE}_EVALUATION_STARTED.json"
FAILED="${ROOT}/evidence/${ROUTE}_EVALUATION_FAILED.json"
VALIDATION="${ROOT}/evidence/${ROUTE}_EVALUATION_VALIDATION.json"
COMPLETE="${ROOT}/evidence/${ROUTE}_EVALUATION_COMPLETE.ok"

read -r EXPECTED_EPOCH EXPECTED_SHA < <(
  "$PYTHON" - "$PLAN" "$ROUTE" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))["routes"][sys.argv[2]]
print(int(p["checkpoint_epoch"]),p["checkpoint_sha256"])
PY
)

write_failure() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" FAILED="$FAILED" ROUTE="$ROUTE" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY' || true
import json,os
from datetime import datetime,timezone
from pathlib import Path
p=Path(os.environ["FAILED"])
payload={"schema_version":1,"status":"failed","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","route":os.environ["ROUTE"],"job_id":os.environ.get("SLURM_JOB_ID"),"exit_code":int(os.environ["RC"]),"failed_at_utc":datetime.now(timezone.utc).isoformat()}
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
test ! -e "$STARTED"
test ! -e "$FAILED"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
test ! -e "$OUT"
test -d "$INPUT"
test -d "$REFERENCE"
test -f "$MODEL_ROOT/fold_0/checkpoint_final.pth"
sha256sum -c STATIC_SHA256SUMS.txt
test "$(sha256sum "$MODEL_ROOT/fold_0/checkpoint_final.pth" | awk '{print $1}')" = "$EXPECTED_SHA"

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/runtime:${S2_REPOSITORY}:${PYTHONPATH:-}"
export nnUNet_raw="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_preprocessed"
export nnUNet_results="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_results"
export nnUNet_extTrainer="${ROOT}/runtime/custom_nnunet"
export nnUNet_compile=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

DEVICE_NAME=$("$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
name=torch.cuda.get_device_name(0)
assert "A100" in name,name
print(name)
PY
)

ROUTE="$ROUTE" EXPECTED_EPOCH="$EXPECTED_EPOCH" EXPECTED_SHA="$EXPECTED_SHA" STARTED="$STARTED" DEVICE_NAME="$DEVICE_NAME" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY'
import json,os
from datetime import datetime,timezone
from pathlib import Path
payload={"schema_version":1,"status":"started","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","route":os.environ["ROUTE"],"selected_epoch":int(os.environ["EXPECTED_EPOCH"]),"checkpoint_sha256":os.environ["EXPECTED_SHA"],"case_count":103,"device_name":os.environ["DEVICE_NAME"],"single_gpu":True,"job_id":os.environ.get("SLURM_JOB_ID"),"started_at_utc":datetime.now(timezone.utc).isoformat()}
with Path(os.environ["STARTED"]).open("x",encoding="utf-8") as h:
    json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
PY

mkdir -p "$OUT"
"$PYTHON" "${ROOT}/runtime/frozen_s2_inference.py" +  --input "$INPUT" --output "${OUT}/prediction" +  --model-root "$MODEL_ROOT" --audit-json "${OUT}/inference_audit.json" +  --expected-count 103 --expected-checkpoint-sha256 "$EXPECTED_SHA" +  --model-name "$ROUTE" +  > "${ROOT}/logs/${ROUTE}_inference_${SLURM_JOB_ID}.log" 2>&1

"${ROOT}/runtime/run_official_met_eval.sh" "$EVAL_ENV" +  "${OUT}/prediction" "$REFERENCE" "${OUT}/evaluation" 103 +  > "${ROOT}/logs/${ROUTE}_official_met_${SLURM_JOB_ID}.log" 2>&1

ROUTE="$ROUTE" OUT="$OUT" VALIDATION="$VALIDATION" COMPLETE="$COMPLETE" EXPECTED_EPOCH="$EXPECTED_EPOCH" EXPECTED_SHA="$EXPECTED_SHA" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY'
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
out=Path(os.environ["OUT"])
inf=json.loads((out/"inference_audit.json").read_text())
ev=json.loads((out/"evaluation/EVALUATION_VALIDATION.json").read_text())
assert inf["status"]=="pass" and inf["case_count"]==103 and inf["prediction_count"]==103
assert inf["checkpoint_sha256"]==os.environ["EXPECTED_SHA"]
assert inf["trainer_name_from_checkpoint"]=="nnUNetTrainerBraTS2026RCMetAugScreen125FocalCompletionFineTune"
assert ev["status"]=="pass" and ev["case_count"]==103
assert len(list((out/"prediction").glob("*.nii.gz")))==103
payload={"schema_version":1,"status":"pass","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","route":os.environ["ROUTE"],"selected_epoch":int(os.environ["EXPECTED_EPOCH"]),"checkpoint_sha256":os.environ["EXPECTED_SHA"],"case_count":103,"prediction_count":103,"metrics_csv_sha256":sha(out/"evaluation/leaderboard_metrics.csv"),"job_id":os.environ.get("SLURM_JOB_ID"),"completed_at_utc":datetime.now(timezone.utc).isoformat(),"locked_test_used":False}
with Path(os.environ["VALIDATION"]).open("x",encoding="utf-8") as h:
    json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
with Path(os.environ["COMPLETE"]).open("x",encoding="ascii") as h:
    h.write("status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n")
print(json.dumps(payload,sort_keys=True))
PY
trap - ERR INT TERM
