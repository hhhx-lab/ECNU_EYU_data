#!/bin/bash
# Independent K4 175-epoch training contract; submitted only after user authorization.
#SBATCH --job-name=s2k4train175r6
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260929_k4_175_r6/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260929_k4_175_r6/logs/%x_%j.err

set -euo pipefail
umask 077
ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260929_k4_175_r6
PARENT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
REPO="$ROOT/runtime/repository"
DISCOVERY="$ROOT/runtime/repository"
SOURCE=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency
RESULTS="$ROOT/results/K4"
FOLD="$RESULTS/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175__nnUNetPlans__3d_fullres/fold_0"
EVIDENCE="$ROOT/evidence"
STARTED="$EVIDENCE/K4_175_TRAINING_STARTED.json"
FAILED="$EVIDENCE/K4_175_TRAINING_FAILED.json"
VALIDATION="$EVIDENCE/K4_175_TRAINING_VALIDATION.json"
COMPLETE="$EVIDENCE/K4_175_TRAINING_COMPLETE.ok"
TRAINER=nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175
fail() { rc=$?; trap - ERR; if [[ ! -e "$FAILED" && ! -e "$COMPLETE" ]]; then "$PYTHON" - "$FAILED" "$rc" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt,json,os,sys
p,rc,j=sys.argv[1:]; x={"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","status":"failed","return_code":int(rc),"job_id":j,"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')}
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n')
PY
fi; exit "$rc"; }
trap fail ERR
[[ ! -e "$STARTED" && ! -e "$FAILED" && ! -e "$VALIDATION" && ! -e "$COMPLETE" ]]
[[ -f "$ROOT/evidence/REMOTE_STATIC_PREFLIGHT.json" && -f "$ROOT/evidence/POOL_BUILD_COMPLETE.ok" ]]
[[ -f "$ROOT/config/K1245_ROUTE_K4.json" && -f "$ROOT/pools/small8/component_manifest.json" ]]
module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:$PATH"
export MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$DISCOVERY:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_extTrainer="$DISCOVERY"
export nnUNet_raw="$SOURCE/nnUNet_raw"
export nnUNet_preprocessed="$PARENT/data/nnUNet_preprocessed"
export nnUNet_results="$RESULTS"
export BRATS_S2_REPO_DIR="$REPO"
export BRATS_SPLIT_DIR="$PARENT/inputs/splits"
export S2_EXPERIMENT_MODE="met_aug_k1245_20260929_k4_175_r6_K4"
export S2_CONTINUE=0 S2_SKIP_COMPLETED=0 S2_MET_AUG_ENABLE=1
export S2_MET_AUG_COMPONENT_MANIFEST="$ROOT/pools/small8/component_manifest.json"
export S2_MET_AUG_ROUTE_CONFIG="$ROOT/config/K1245_ROUTE_K4.json"
export S2_MET_AUG_VALID_MASK_MANIFEST="$PARENT/route_rebuild_20260812_r1/valid_masks/valid_mask_manifest.json"
export S2_MET_AUG_FIX_V3_CALIBRATION="$PARENT/calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
export S2_MET_AUG_ORIGINAL_E_CHECKPOINT="$PARENT/frozen/original_e/checkpoint_final.pth"
export S2_MET_AUG_G1_CODE_DIR="$PARENT/runtime/g1_archive/BraTS_2023_2024_solutions-main_3/Segmentation_Tasks/GliGAN"
export S2_MET_AUG_G1_CHECKPOINT_ROOT="$PARENT/frozen/g1"
export S2_MET_AUG_G1_CHECKPOINT_SELECTION="$PARENT/frozen/g1/checkpoint_selection.json"
export S2_MET_AUG_G2_QC_GATE="$PARENT/frozen/g1/g2_diffusion_qc_gate.json"
export S2_MET_AUG_AUDIT_PATH="$FOLD/met_aug_events.jsonl"
export S2_MET_AUG_K1245_PLAN="$ROOT/K1245_K4_175_EXECUTION_PLAN_20260929_R6.json"
export S2_K1245_COMPONENT_MANIFEST="$ROOT/pools/small8/component_manifest.json"
export S2_K1245_CONFIG="$ROOT/config/K1245_ROUTE_K4.json"
export S2_K1245_PLAN="$ROOT/K1245_K4_175_EXECUTION_PLAN_20260929_R6.json"
export S2_K1245_SMOKE_AUTHORIZATION="$ROOT/evidence/K4_175_TRAINING_AUTHORIZATION.json"
export S2_K1245_LEGACY_COMPONENT_MANIFEST="$PARENT/route_rebuild_20260812_r1/component_pool/component_manifest.json"
export S2_K1245_EPOCHS=175 S2_K1245_INITIAL_LR=0.001 S2_K1245_SAVE_EVERY=25
export S2_PAIRED_TRAINING_SEED=20260724
export S2_COMPLETION_EPOCHS=175 S2_COMPLETION_INITIAL_LR=0.001 S2_COMPLETION_SAVE_EVERY=25
export S2_SCREEN_EPOCHS=175 S2_SCREEN_INITIAL_LR=0.001 S2_SCREEN_SAVE_EVERY=25
export S2_FOCAL_GAMMA=2.0 CUBLAS_WORKSPACE_CONFIG=:4096:8 nnUNet_compile=0 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON" - "$STARTED" "${SLURM_JOB_ID:-unknown}" "$ROOT/K1245_K4_175_EXECUTION_PLAN_20260929_R6.json" "$REPO/custom_nnunet/$TRAINER.py" "$ROOT/config/K1245_ROUTE_K4.json" "$ROOT/pools/small8/component_manifest.json" "$S2_MET_AUG_FIX_V3_CALIBRATION" <<'PY'
import datetime as dt,hashlib,json,os,sys
from pathlib import Path
p,j,plan,trainer,cfg,manifest,calibration=sys.argv[1:]; import torch
if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0): raise SystemExit('requires one A100')
assert json.loads(Path(cfg).read_text())['route_id'] == 'K4'
assert json.loads(Path(manifest).read_text())['component_count'] >= 24
calibration_sha=hashlib.sha256(Path(calibration).read_bytes()).hexdigest()
assert calibration_sha == '68a5d8a6c501804b7a47d626e43a58ff71b781154c0cc955a9c5a5d5b786d37f'
assert json.loads(Path(cfg).read_text())['legacy_fix_v3_calibration_sha256'] == calibration_sha
payload={"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","status":"started","job_id":j,"route":"K4","epochs":175,"checkpoint_epochs":[25,50,75,100,125,150,175],"trainer":Path(trainer).name,"trainer_sha256":hashlib.sha256(Path(trainer).read_bytes()).hexdigest(),"config_sha256":hashlib.sha256(Path(cfg).read_bytes()).hexdigest(),"manifest_sha256":hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),"plan_sha256":hashlib.sha256(Path(plan).read_bytes()).hexdigest(),"calibration_path":calibration,"calibration_sha256":calibration_sha,"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')}
fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
PY
nnUNetv2_train 264 3d_fullres 0 -tr "$TRAINER" -num_gpus 1 -pretrained_weights "$PARENT/frozen/original_e/checkpoint_final.pth"
"$PYTHON" - "$FOLD" "$VALIDATION" "$COMPLETE" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt,hashlib,json,os,sys
from pathlib import Path
fold,v,c,j=sys.argv[1:]; fold=Path(fold); rows=[]
for e in (25,50,75,100,125,150,175):
 p=fold/f'checkpoint_epoch_{e:04d}.pth'
 if not p.is_file(): raise SystemExit(f'missing {p}')
 import torch
 x=torch.load(p,map_location='cpu',weights_only=False); log=x.get('logging',{}); ema=log.get('ema_fg_dice',[])
 if int(x.get('current_epoch',-1))!=e or len(ema)<e: raise SystemExit(f'checkpoint contract failed {p}')
 rows.append({'epoch':e,'ema_fg_dice':float(ema[e-1]),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
best=sorted(rows,key=lambda x:(-x['ema_fg_dice'],x['epoch']))[0]
payload={'artifact_status':'experimental_unvalidated','operator_approved':False,'formal_gate_status':'not_run_not_passed','status':'pass','job_id':j,'completed_epochs':175,'candidate_checkpoints':rows,'best_checkpoint':best,'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00','Z')}
fd=os.open(v,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
fd=os.open(c,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as f: f.write('complete\n')
PY
