#!/usr/bin/env bash
#SBATCH --job-name=s2r4103p1
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/logs/fixed103_paired_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/logs/fixed103_paired_%j.err

set -euo pipefail
umask 027

PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1"
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
EVAL_ENV=/public/home/zqchen/.conda/envs/brats_eval
S2_REPOSITORY="${PROJECT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
STAGE="${ROOT}/cohorts/fixed103"
FAILED="${ROOT}/evidence/FIXED103_PAIRED_FAILED.json"
VALIDATION="${ROOT}/evidence/FIXED103_PAIRED_VALIDATION.json"
COMPLETE="${ROOT}/evidence/FIXED103_PAIRED_COMPLETE.ok"
GATE="${ROOT}/evidence/FIXED103_GATE_FROZEN.json"
SYNTH="${PROJECT}/work_space/G1/results/experimental_unvalidated_single_model_eval_20260812_r4/ensemble_r4/synthesized"
MODEL="${PROJECT}/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/data/nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__nnUNetPlans__3d_fullres"
CHECKPOINT_SHA=b8b2d13a6268231f73d43c37cb097d1be3daeec654b9bba6ddd24f410cf7b27e

fail_stage() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" FAILED="$FAILED" STAGE_NAME=fixed103_real_vs_synthetic SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY' || true
import json,os
from datetime import datetime,timezone
from pathlib import Path
p=Path(os.environ["FAILED"]); d={"status":"failed","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","stage":os.environ["STAGE_NAME"],"job_id":os.environ.get("SLURM_JOB_ID"),"exit_code":int(os.environ["RC"]),"failed_at_utc":datetime.now(timezone.utc).isoformat()}
try:
    with p.open("x",encoding="utf-8") as f: json.dump(d,f,indent=2,sort_keys=True); f.write("\n")
except FileExistsError: pass
PY
  exit "$rc"
}
trap fail_stage ERR INT TERM

cd "$ROOT"
test -f evidence/VAL27_FOUR_MODELS_COMPLETE.ok
test -f evidence/VAL27_FOUR_MODELS_VALIDATION.json
test ! -e "$STAGE"
test ! -e "$FAILED"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
test ! -e "$GATE"
sha256sum -c STATIC_SHA256SUMS.txt

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

"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert "A100" in torch.cuda.get_device_name(0)
print(torch.cuda.get_device_name(0))
PY

mkdir -p "$STAGE"
for route in real synthetic; do
  route_root="${STAGE}/${route}"
  if [[ "$route" == real ]]; then
    "$PYTHON" "${ROOT}/runtime/materialize_s2_inputs.py" \
      --manifest "${ROOT}/manifests/FIXED103_REAL_SYNTHETIC_CASES.csv" \
      --source-root /public/home/zqchen/data --route real \
      --input-root "${route_root}/input" --reference-root "${route_root}/reference" \
      --audit-json "${route_root}/input_audit.json" --audit-csv "${route_root}/input_audit.csv" \
      --expected-count 103
  else
    "$PYTHON" "${ROOT}/runtime/materialize_s2_inputs.py" \
      --manifest "${ROOT}/manifests/FIXED103_REAL_SYNTHETIC_CASES.csv" \
      --source-root /public/home/zqchen/data --synthetic-root "$SYNTH" --route synthetic \
      --input-root "${route_root}/input" --reference-root "${route_root}/reference" \
      --audit-json "${route_root}/input_audit.json" --audit-csv "${route_root}/input_audit.csv" \
      --expected-count 103
  fi
  "$PYTHON" "${ROOT}/runtime/frozen_s2_inference.py" \
    --input "${route_root}/input" --output "${route_root}/prediction" \
    --model-root "$MODEL" --audit-json "${route_root}/inference_audit.json" \
    --expected-count 103 --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
    --model-name "FixV3_${route}_T2W" \
    > "${ROOT}/logs/fixed103_${route}_inference_${SLURM_JOB_ID}.log" 2>&1
  "${ROOT}/runtime/run_official_met_eval.sh" "$EVAL_ENV" \
    "${route_root}/prediction" "${route_root}/reference" "${route_root}/evaluation" 103 \
    > "${ROOT}/logs/fixed103_${route}_evaluation_${SLURM_JOB_ID}.log" 2>&1
done

"$PYTHON" "${ROOT}/runtime/compare_paired_met.py" \
  --input "real_T2W=${STAGE}/real/evaluation/leaderboard_metrics.csv" \
  --input "r4_ensemble_T2W=${STAGE}/synthetic/evaluation/leaderboard_metrics.csv" \
  --output-root "${STAGE}/paired_comparison" --expected-count 103 \
  --bootstrap 20000 --seed 20260824 --scope fixed103_authentic_vs_r4_synthesized_t2w_same_fixv3

ROOT="$ROOT" STAGE="$STAGE" VALIDATION="$VALIDATION" GATE="$GATE" CHECKPOINT_SHA="$CHECKPOINT_SHA" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY'
import csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ["ROOT"]); stage=Path(os.environ["STAGE"]); validation=Path(os.environ["VALIDATION"]); gate=Path(os.environ["GATE"])
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
route_data={}
audits={}
for route in ("real","synthetic"):
 rr=stage/route
 inp=json.loads((rr/"input_audit.json").read_text()); inf=json.loads((rr/"inference_audit.json").read_text()); ev=json.loads((rr/"evaluation"/"EVALUATION_VALIDATION.json").read_text())
 assert inp["status"]==inf["status"]==ev["status"]=="pass"
 assert inp["case_count"]==inf["case_count"]==ev["case_count"]==103
 assert inf["checkpoint_sha256"]==os.environ["CHECKPOINT_SHA"]
 assert len(list((rr/"prediction").glob("*.nii.gz")))==103
 rows=list(csv.DictReader((rr/"input_audit.csv").open()))
 audits[route]={r["source_case_id"]:r for r in rows}
 route_data[route]={"input_audit_sha256":sha(rr/"input_audit.json"),"metrics_sha256":sha(rr/"evaluation"/"leaderboard_metrics.csv")}
assert set(audits["real"])==set(audits["synthetic"])
for case_id in audits["real"]:
 a=audits["real"][case_id]; b=audits["synthetic"][case_id]
 for key in ("t1n_sha256","t1c_sha256","t2f_sha256","seg_sha256"):
  assert a[key]==b[key], (case_id,key)
paired=json.loads((stage/"paired_comparison"/"PAIRED_COMPARISON.json").read_text())
assert paired["status"]=="pass" and paired["case_count"]==103 and len(paired["pairs"])==1
payload={"schema_version":1,"status":"pass","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","stage":"fixed103_real_vs_synthetic","job_id":os.environ.get("SLURM_JOB_ID"),"completed_at_utc":datetime.now(timezone.utc).isoformat(),"case_count":103,"checkpoint_sha256":os.environ["CHECKPOINT_SHA"],"same_checkpoint":True,"same_case_ids":True,"same_reference_labels":True,"same_non_t2w_channels":True,"routes":route_data,"paired_comparison_sha256":sha(stage/"paired_comparison"/"PAIRED_COMPARISON.json")}
with validation.open("x",encoding="utf-8") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
plan=root/"ENDPOINT_VALIDATION_EXECUTION_PLAN_20260824_R1.json"
gate_payload={"schema_version":1,"status":"frozen_for_locked_test","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","frozen_at_utc":datetime.now(timezone.utc).isoformat(),"checkpoint_sha256":os.environ["CHECKPOINT_SHA"],"execution_plan_sha256":sha(plan),"val27_validation_sha256":sha(root/"evidence/VAL27_FOUR_MODELS_VALIDATION.json"),"fixed103_validation_sha256":sha(validation),"tuning_after_freeze_allowed":False,"test26_unlocked":True}
with gate.open("x",encoding="utf-8") as f: json.dump(gate_payload,f,indent=2,sort_keys=True); f.write("\n")
print(json.dumps(payload,sort_keys=True))
PY
printf 'status=pass\nstage=fixed103_real_vs_synthetic\nartifact_status=experimental_unvalidated\noperator_approved=false\n' > "$COMPLETE"
trap - ERR INT TERM

