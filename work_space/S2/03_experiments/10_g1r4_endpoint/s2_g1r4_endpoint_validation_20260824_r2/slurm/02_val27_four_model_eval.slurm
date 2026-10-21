#!/usr/bin/env bash
#SBATCH --job-name=s2r4v27m2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/logs/val27_models_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/logs/val27_models_%j.err

set -euo pipefail
umask 027

PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2"
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
EVAL_ENV=/public/home/zqchen/.conda/envs/brats_eval
S2_REPOSITORY="${PROJECT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
STAGE="${ROOT}/cohorts/val27/segmentation"
FAILED="${ROOT}/evidence/VAL27_FOUR_MODELS_FAILED.json"
VALIDATION="${ROOT}/evidence/VAL27_FOUR_MODELS_VALIDATION.json"
COMPLETE="${ROOT}/evidence/VAL27_FOUR_MODELS_COMPLETE.ok"
INPUT="${STAGE}/input_synthetic"
REFERENCE="${STAGE}/reference"
SYNTH="${ROOT}/cohorts/val27/synthesis_r4/synthesized"

fail_stage() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" FAILED="$FAILED" STAGE_NAME=val27_four_models SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY' || true
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
test -f evidence/VAL27_SYNTHESIS_COMPLETE.ok
test -f evidence/VAL27_SYNTHESIS_VALIDATION.json
test ! -e "$STAGE"
test ! -e "$FAILED"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
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
"$PYTHON" "${ROOT}/runtime/materialize_s2_inputs.py" \
  --manifest "${ROOT}/manifests/VAL27_MISSING_T2W_CASES.csv" \
  --source-root /public/home/zqchen/data --synthetic-root "$SYNTH" --route synthetic \
  --input-root "$INPUT" --reference-root "$REFERENCE" \
  --audit-json "${STAGE}/input_audit.json" --audit-csv "${STAGE}/input_audit.csv" \
  --expected-count 27

declare -A MODEL_ROOTS=(
  [R]="${ROOT}/model_snapshots/R"
  [B]="${ROOT}/model_snapshots/B"
  [E]="${PROJECT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCFocalCompletionFineTune__nnUNetPlans__3d_fullres"
  [FixV3]="${PROJECT}/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/data/nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__nnUNetPlans__3d_fullres"
)
declare -A MODEL_SHA=(
  [R]=bec0beead1b6ab6beda5e10c9c92e192af833fa82f794277d50f81d277a922a1
  [B]=78eccc59f9217a529cafdd522733de9a1578f0e96d8765ee7c48731027824db5
  [E]=4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267
  [FixV3]=b8b2d13a6268231f73d43c37cb097d1be3daeec654b9bba6ddd24f410cf7b27e
)

for model in R B E FixV3; do
  out="${STAGE}/models/${model}"
  mkdir -p "$out"
  "$PYTHON" "${ROOT}/runtime/frozen_s2_inference.py" \
    --input "$INPUT" --output "${out}/prediction" \
    --model-root "${MODEL_ROOTS[$model]}" \
    --audit-json "${out}/inference_audit.json" --expected-count 27 \
    --expected-checkpoint-sha256 "${MODEL_SHA[$model]}" --model-name "$model" \
    > "${ROOT}/logs/val27_${model}_inference_${SLURM_JOB_ID}.log" 2>&1
  "${ROOT}/runtime/run_official_met_eval.sh" "$EVAL_ENV" \
    "${out}/prediction" "$REFERENCE" "${out}/evaluation" 27 \
    > "${ROOT}/logs/val27_${model}_evaluation_${SLURM_JOB_ID}.log" 2>&1
done

"$PYTHON" "${ROOT}/runtime/compare_paired_met.py" \
  --input "R=${STAGE}/models/R/evaluation/leaderboard_metrics.csv" \
  --input "B=${STAGE}/models/B/evaluation/leaderboard_metrics.csv" \
  --input "E=${STAGE}/models/E/evaluation/leaderboard_metrics.csv" \
  --input "FixV3=${STAGE}/models/FixV3/evaluation/leaderboard_metrics.csv" \
  --output-root "${STAGE}/paired_comparison" --expected-count 27 \
  --bootstrap 20000 --seed 20260824 --scope val27_missing_t2w_four_model_endpoint

ROOT="$ROOT" STAGE="$STAGE" VALIDATION="$VALIDATION" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY'
import csv,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
stage=Path(os.environ["STAGE"]); target=Path(os.environ["VALIDATION"])
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
assert json.loads((stage/"input_audit.json").read_text())["case_count"]==27
models={}
for name in ("R","B","E","FixV3"):
 root=stage/"models"/name
 inf=json.loads((root/"inference_audit.json").read_text()); ev=json.loads((root/"evaluation"/"EVALUATION_VALIDATION.json").read_text())
 assert inf["status"]=="pass" and inf["prediction_count"]==27
 assert ev["status"]=="pass" and ev["case_count"]==27
 assert len(list((root/"prediction").glob("*.nii.gz")))==27
 models[name]={"checkpoint_sha256":inf["checkpoint_sha256"],"metrics_sha256":sha(root/"evaluation"/"leaderboard_metrics.csv")}
paired=json.loads((stage/"paired_comparison"/"PAIRED_COMPARISON.json").read_text())
assert paired["status"]=="pass" and paired["case_count"]==27 and len(paired["pairs"])==6
payload={"schema_version":1,"status":"pass","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","stage":"val27_four_models","job_id":os.environ.get("SLURM_JOB_ID"),"completed_at_utc":datetime.now(timezone.utc).isoformat(),"case_count":27,"models":models,"paired_comparison_sha256":sha(stage/"paired_comparison"/"PAIRED_COMPARISON.json"),"locked_test_used":False}
with target.open("x",encoding="utf-8") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
print(json.dumps(payload,sort_keys=True))
PY
printf 'status=pass\nstage=val27_four_models\nartifact_status=experimental_unvalidated\noperator_approved=false\n' > "$COMPLETE"
trap - ERR INT TERM
