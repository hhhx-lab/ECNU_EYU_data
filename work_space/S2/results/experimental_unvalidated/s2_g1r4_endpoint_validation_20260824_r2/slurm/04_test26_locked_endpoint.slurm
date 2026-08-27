#!/usr/bin/env bash
#SBATCH --job-name=s2r4t26e2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/logs/test26_endpoint_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/logs/test26_endpoint_%j.err

set -euo pipefail
umask 027

PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2"
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
EVAL_ENV=/public/home/zqchen/.conda/envs/brats_eval
G1_ASSETS="${PROJECT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3"
G1_RUNTIME="${ROOT}/runtime/g1_r4_frozen"
S2_REPOSITORY="${PROJECT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
STAGE="${ROOT}/cohorts/test26"
FAILED="${ROOT}/evidence/TEST26_LOCKED_ENDPOINT_FAILED.json"
VALIDATION="${ROOT}/evidence/TEST26_LOCKED_ENDPOINT_VALIDATION.json"
COMPLETE="${ROOT}/evidence/TEST26_LOCKED_ENDPOINT_COMPLETE.ok"
PIPELINE_COMPLETE="${ROOT}/ENDPOINT_VALIDATION_PIPELINE_COMPLETE.ok"
MODEL="${PROJECT}/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/data/nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__nnUNetPlans__3d_fullres"
CHECKPOINT_SHA=b8b2d13a6268231f73d43c37cb097d1be3daeec654b9bba6ddd24f410cf7b27e
VAE="${G1_ASSETS}/training/vae_finetuned/run_3034060/best_model.pt"
ENCDEC="${G1_ASSETS}/training/endec/endec_3056244/check_points/model_37035.pt"
BBDM="${G1_ASSETS}/training/bbdm/bbdm_3056245/check_points/model_111105.pt"

fail_stage() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" FAILED="$FAILED" STAGE_NAME=test26_locked_endpoint SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY' || true
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
test -f evidence/FIXED103_PAIRED_COMPLETE.ok
test -f evidence/FIXED103_PAIRED_VALIDATION.json
test -f evidence/FIXED103_GATE_FROZEN.json
test ! -e "$STAGE"
test ! -e "$FAILED"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
test ! -e "$PIPELINE_COMPLETE"
sha256sum -c STATIC_SHA256SUMS.txt

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/runtime:${G1_RUNTIME}:${S2_REPOSITORY}:${PYTHONPATH:-}"
export G1_VAE_WEIGHTS="$VAE"
export G1_ENCDEC_CHECKPOINT="$ENCDEC"
export G1_BBDM_CHECKPOINT="$BBDM"
export G1_BBDM_S=0.01
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
"$PYTHON" "${ROOT}/runtime/synthesize_missing_t2w_r4.py" \
  --manifest "${ROOT}/manifests/TEST26_LOCKED_MISSING_T2W_CASES.csv" \
  --source-root /public/home/zqchen/data \
  --output-root "${STAGE}/synthesis_r4/synthesized" \
  --spatial-audit "${STAGE}/synthesis_r4/spatial_audit.csv" \
  --geometry-audit-json "${STAGE}/synthesis_r4/geometry_audit.json" \
  --geometry-audit-csv "${STAGE}/synthesis_r4/geometry_audit.csv" \
  --run-json "${STAGE}/synthesis_r4/synthesis_run.json" \
  --expected-count 26 --cohort test26_locked \
  --vae-path "$VAE" --encdec-path "$ENCDEC" --bbdm-path "$BBDM" \
  --vae-sha256 004dbaa00669685b1f24fbaf6044d5f592f4a7ae6156a482be34235724175294 \
  --encdec-sha256 aed9bc96d6cb285b60d092fd48f572c60f76dba2d654dba4e60754ad55c0c558 \
  --bbdm-sha256 969eef82241a9500d843bb38b4fcf3230eb0b55ecfb1876aa83c65b59ec05541

"$PYTHON" "${ROOT}/runtime/materialize_s2_inputs.py" \
  --manifest "${ROOT}/manifests/TEST26_LOCKED_MISSING_T2W_CASES.csv" \
  --source-root /public/home/zqchen/data --synthetic-root "${STAGE}/synthesis_r4/synthesized" --route synthetic \
  --input-root "${STAGE}/input" --reference-root "${STAGE}/reference" \
  --audit-json "${STAGE}/input_audit.json" --audit-csv "${STAGE}/input_audit.csv" \
  --expected-count 26

"$PYTHON" "${ROOT}/runtime/frozen_s2_inference.py" \
  --input "${STAGE}/input" --output "${STAGE}/prediction" \
  --model-root "$MODEL" --audit-json "${STAGE}/inference_audit.json" \
  --expected-count 26 --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --model-name FixV3_locked_test26 \
  > "${ROOT}/logs/test26_inference_${SLURM_JOB_ID}.log" 2>&1
"${ROOT}/runtime/run_official_met_eval.sh" "$EVAL_ENV" \
  "${STAGE}/prediction" "${STAGE}/reference" "${STAGE}/evaluation" 26 \
  > "${ROOT}/logs/test26_evaluation_${SLURM_JOB_ID}.log" 2>&1

ROOT="$ROOT" STAGE="$STAGE" VALIDATION="$VALIDATION" CHECKPOINT_SHA="$CHECKPOINT_SHA" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY'
import csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
root=Path(os.environ["ROOT"]); stage=Path(os.environ["STAGE"]); target=Path(os.environ["VALIDATION"])
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
synth=json.loads((stage/"synthesis_r4/synthesis_run.json").read_text()); geom=json.loads((stage/"synthesis_r4/geometry_audit.json").read_text()); inp=json.loads((stage/"input_audit.json").read_text()); inf=json.loads((stage/"inference_audit.json").read_text()); ev=json.loads((stage/"evaluation/EVALUATION_VALIDATION.json").read_text()); gate=json.loads((root/"evidence/FIXED103_GATE_FROZEN.json").read_text())
assert all(item["status"]=="pass" for item in (synth,geom,inp,inf,ev))
assert synth["case_count"]==geom["case_count"]==inp["case_count"]==inf["case_count"]==ev["case_count"]==26
assert synth["source_t2w_read"] is False and geom["repaired_count"]==0 and inp["geometry_mismatch_count"]==0
assert inf["checkpoint_sha256"]==gate["checkpoint_sha256"]==os.environ["CHECKPOINT_SHA"]
assert len(list((stage/"synthesis_r4/synthesized").glob("*.nii.gz")))==26
assert len(list((stage/"prediction").glob("*.nii.gz")))==26
payload={"schema_version":1,"status":"pass","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","stage":"test26_locked_endpoint","job_id":os.environ.get("SLURM_JOB_ID"),"completed_at_utc":datetime.now(timezone.utc).isoformat(),"case_count":26,"checkpoint_sha256":os.environ["CHECKPOINT_SHA"],"source_t2w_read_count":0,"geometry_mismatch_count":0,"geometry_repaired_count":0,"synthesis_run_sha256":sha(stage/"synthesis_r4/synthesis_run.json"),"input_audit_sha256":sha(stage/"input_audit.json"),"inference_audit_sha256":sha(stage/"inference_audit.json"),"metrics_sha256":sha(stage/"evaluation/leaderboard_metrics.csv"),"frozen_gate_sha256":sha(root/"evidence/FIXED103_GATE_FROZEN.json")}
with target.open("x",encoding="utf-8") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
print(json.dumps(payload,sort_keys=True))
PY
printf 'status=pass\nstage=test26_locked_endpoint\nartifact_status=experimental_unvalidated\noperator_approved=false\n' > "$COMPLETE"
printf 'status=pass\nartifact_status=experimental_unvalidated\noperator_approved=false\nformal_gate_status=not_run_not_passed\n' > "$PIPELINE_COMPLETE"
trap - ERR INT TERM
