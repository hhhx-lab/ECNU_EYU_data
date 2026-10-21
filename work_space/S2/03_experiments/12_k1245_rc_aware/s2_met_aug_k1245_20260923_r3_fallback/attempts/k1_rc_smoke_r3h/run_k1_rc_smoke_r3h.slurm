#!/bin/bash
#SBATCH --job-name=s2k1rcsmkr3h
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r3_fallback/attempts/k1_rc_smoke_r3h/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r3_fallback/attempts/k1_rc_smoke_r3h/logs/%x_%j.err
set -euo pipefail
umask 077
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r3_fallback/attempts/k1_rc_smoke_r3h
R2=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260923_r2
PARENT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
REPO="$ROOT/runtime/repository"
FAILED="$ROOT/evidence/K1_SMOKE_R3H_FAILED.json"
STARTED="$ROOT/evidence/K1_SMOKE_R3H_STARTED.json"
VALIDATION="$ROOT/evidence/K1_SMOKE_R3H_VALIDATION.json"
fail() { rc=$?; trap - ERR; if [[ ! -e "$FAILED" ]]; then "$PYTHON" - "$FAILED" "$rc" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt,json,os,sys
p,rc,j=sys.argv[1:]; x={"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","status":"failed","return_code":int(rc),"job_id":j,"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')}
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n')
PY
fi; exit "$rc"; }
trap fail ERR
[[ -f "$ROOT/evidence/K1_SMOKE_R3H_STATIC_PREFLIGHT.json" ]]
[[ ! -e "$STARTED" && ! -e "$FAILED" && ! -e "$VALIDATION" ]]
[[ ! -e "$ROOT/smoke/k1_rc24_r3h" ]]
module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - "$STARTED" "${SLURM_JOB_ID:-unknown}" "$ROOT/K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3H.json" <<'PY'
import datetime as dt,hashlib,json,os,sys
p,j,plan=sys.argv[1:]; x={"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","status":"started","job_id":j,"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z'),"plan_sha256":hashlib.sha256(open(plan,'rb').read()).hexdigest(),"F_STRICT_PREREG_GATE":False}
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n')
PY
"$PYTHON" -B "$ROOT/runtime/run_k1_rc_smoke.py" --smoke-plan "$ROOT/K1_RC_SMOKE_EXECUTION_PLAN_20260923_R3H.json" --pool-root "$R2" --preprocessed-dir "$PARENT/data/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres" --valid-mask-manifest "$PARENT/route_rebuild_20260812_r1/valid_masks/valid_mask_manifest.json" --legacy-manifest "$PARENT/route_rebuild_20260812_r1/component_manifest.json" --calibration "$PARENT/calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json" --g1-code-dir "$PARENT/runtime/g1_archive/BraTS_2023_2024_solutions-main_3/Segmentation_Tasks/GliGAN" --g1-checkpoint-root "$PARENT/frozen/g1" --g1-checkpoint-selection "$PARENT/frozen/g1/checkpoint_selection.json" --g2-parent-gate "$PARENT/frozen/g1/g2_diffusion_qc_gate.json" --device cuda
"$PYTHON" - "$ROOT" "$VALIDATION" <<'PY'
import datetime as dt,json,os,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]); report=json.loads((root/'smoke/k1_rc24_r3h/AUTOMATIC_VALIDATION.json').read_text())
if report.get('automatic_status')!='pass' or report.get('automatic_pass_count')!=24 or report.get('F_STRICT_PREREG_GATE') is not False: raise SystemExit('K1 RC smoke contract failed')
report.update({"status":"hold_for_manual_review","manual_review_required":24,"training_authorization":False,"k5_authorization":False,"validation_timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')})
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as f: json.dump(report,f,indent=2,sort_keys=True); f.write('\n')
PY
