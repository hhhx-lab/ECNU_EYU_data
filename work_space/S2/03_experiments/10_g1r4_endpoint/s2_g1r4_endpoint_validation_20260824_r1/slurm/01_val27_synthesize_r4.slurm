#!/usr/bin/env bash
#SBATCH --job-name=s2r4v27s1
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/logs/val27_synthesis_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/logs/val27_synthesis_%j.err

set -euo pipefail
umask 027

PROJECT=/public/home/zqchen/projects/ECNU_EYU_data
ROOT="${PROJECT}/work_space/S2/results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1"
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
G1="${PROJECT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3"
STAGE="${ROOT}/cohorts/val27/synthesis_r4"
FAILED="${ROOT}/evidence/VAL27_SYNTHESIS_FAILED.json"
VALIDATION="${ROOT}/evidence/VAL27_SYNTHESIS_VALIDATION.json"
COMPLETE="${ROOT}/evidence/VAL27_SYNTHESIS_COMPLETE.ok"
VAE="${G1}/training/vae_finetuned/run_3034060/best_model.pt"
ENCDEC="${G1}/training/endec/endec_3056244/check_points/model_37035.pt"
BBDM="${G1}/training/bbdm/bbdm_3056245/check_points/model_111105.pt"

fail_stage() {
  rc=$?
  trap - ERR INT TERM
  RC="$rc" FAILED="$FAILED" STAGE_NAME=val27_r4_synthesis SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY' || true
import json, os
from datetime import datetime, timezone
from pathlib import Path
path=Path(os.environ["FAILED"])
payload={"status":"failed","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","stage":os.environ["STAGE_NAME"],"job_id":os.environ.get("SLURM_JOB_ID"),"exit_code":int(os.environ["RC"]),"failed_at_utc":datetime.now(timezone.utc).isoformat()}
try:
    with path.open("x",encoding="utf-8") as handle:
        json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
except FileExistsError:
    pass
PY
  exit "$rc"
}
trap fail_stage ERR INT TERM

cd "$ROOT"
test -f ENDPOINT_VALIDATION_EXECUTION_PLAN_20260824_R1.json
test -f evidence/REMOTE_STATIC_PREFLIGHT.json
test ! -e "$STAGE"
test ! -e "$FAILED"
test ! -e "$VALIDATION"
test ! -e "$COMPLETE"
sha256sum -c STATIC_SHA256SUMS.txt

module purge
module load apps/envs/miniconda3/25.5.1
module load compiler/cuda/12.1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${G1}:${PYTHONPATH:-}"
export G1_VAE_WEIGHTS="$VAE"
export G1_ENCDEC_CHECKPOINT="$ENCDEC"
export G1_BBDM_CHECKPOINT="$BBDM"
export G1_BBDM_S=0.01

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
    raise SystemExit(f"expected one CUDA device, got {torch.cuda.device_count()}")
if "A100" not in torch.cuda.get_device_name(0):
    raise SystemExit(f"expected A100, got {torch.cuda.get_device_name(0)}")
print(torch.cuda.get_device_name(0))
PY

mkdir -p "$(dirname "$STAGE")"
"$PYTHON" "${ROOT}/runtime/synthesize_missing_t2w_r4.py" \
  --manifest "${ROOT}/manifests/VAL27_MISSING_T2W_CASES.csv" \
  --source-root /public/home/zqchen/data \
  --output-root "${STAGE}/synthesized" \
  --spatial-audit "${STAGE}/spatial_audit.csv" \
  --geometry-audit-json "${STAGE}/geometry_audit.json" \
  --geometry-audit-csv "${STAGE}/geometry_audit.csv" \
  --run-json "${STAGE}/synthesis_run.json" \
  --expected-count 27 --cohort val27 \
  --vae-path "$VAE" --encdec-path "$ENCDEC" --bbdm-path "$BBDM" \
  --vae-sha256 004dbaa00669685b1f24fbaf6044d5f592f4a7ae6156a482be34235724175294 \
  --encdec-sha256 aed9bc96d6cb285b60d092fd48f572c60f76dba2d654dba4e60754ad55c0c558 \
  --bbdm-sha256 969eef82241a9500d843bb38b4fcf3230eb0b55ecfb1876aa83c65b59ec05541

ROOT="$ROOT" STAGE="$STAGE" VALIDATION="$VALIDATION" SLURM_JOB_ID="${SLURM_JOB_ID:-}" "$PYTHON" - <<'PY'
import csv, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
root=Path(os.environ["ROOT"]); stage=Path(os.environ["STAGE"]); target=Path(os.environ["VALIDATION"])
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
run=json.loads((stage/"synthesis_run.json").read_text()); geom=json.loads((stage/"geometry_audit.json").read_text())
spatial=list(csv.DictReader((stage/"spatial_audit.csv").open()))
outputs=sorted((stage/"synthesized").glob("*.nii.gz"))
assert run["status"]=="pass" and run["case_count"]==27 and run["source_t2w_read"] is False
assert run["spatial_preprocessing"]=="foreground_centered_isotropic_resample_v1"
assert geom["status"]=="pass" and geom["case_count"]==27 and geom["geometry_mismatch_before_count"]==0 and geom["repaired_count"]==0
assert len(spatial)==27 and all(int(r["foreground_outside_voxel_count"])==0 and int(r["lesion_outside_voxel_count"])==0 for r in spatial)
assert len(outputs)==27
payload={"schema_version":1,"status":"pass","artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","stage":"val27_r4_synthesis","job_id":os.environ.get("SLURM_JOB_ID"),"completed_at_utc":datetime.now(timezone.utc).isoformat(),"case_count":27,"nifti_count":27,"spatial_row_count":27,"foreground_outside_count":0,"lesion_outside_count":0,"geometry_mismatch_count":0,"geometry_repaired_count":0,"source_t2w_read_count":0,"synthesis_run_sha256":sha(stage/"synthesis_run.json"),"spatial_audit_sha256":sha(stage/"spatial_audit.csv"),"geometry_audit_sha256":sha(stage/"geometry_audit.json")}
with target.open("x",encoding="utf-8") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n")
print(json.dumps(payload,sort_keys=True))
PY
printf 'status=pass\nstage=val27_r4_synthesis\nartifact_status=experimental_unvalidated\noperator_approved=false\n' > "$COMPLETE"
trap - ERR INT TERM

