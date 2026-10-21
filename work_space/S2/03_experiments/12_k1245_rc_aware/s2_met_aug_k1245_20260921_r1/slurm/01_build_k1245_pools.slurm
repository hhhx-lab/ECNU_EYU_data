#!/bin/bash
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --job-name=s2kpoolr1
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260921_r1/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260921_r1/logs/%x_%j.err

set -euo pipefail
umask 077

ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260921_r1
PARENT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
REPO="${ROOT}/runtime/repository"
SOURCE=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency
STARTED="${ROOT}/evidence/POOL_BUILD_STARTED.json"
FAILED="${ROOT}/evidence/POOL_BUILD_FAILED.json"
COMPLETE="${ROOT}/evidence/POOL_BUILD_COMPLETE.ok"

write_failure() {
  rc=$?
  trap - ERR
  if [[ ! -e "${FAILED}" && ! -e "${COMPLETE}" ]]; then
    "${PYTHON}" - "${FAILED}" "${rc}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt, json, os, sys
path, rc, job = sys.argv[1:]
payload={"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","status":"failed","return_code":int(rc),"job_id":job,"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z")}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w") as h: json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
PY
  fi
  exit "${rc}"
}
trap write_failure ERR

[[ -f "${ROOT}/evidence/REMOTE_STATIC_PREFLIGHT.json" ]] || { echo preflight_missing >&2; exit 2; }
[[ ! -e "${STARTED}" && ! -e "${FAILED}" && ! -e "${COMPLETE}" && ! -e "${ROOT}/pools" ]] || { echo immutable_output_exists >&2; exit 2; }
module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" - "${STARTED}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt, json, os, sys
path, job = sys.argv[1:]
payload={"artifact_status":"experimental_unvalidated","operator_approved":False,"formal_gate_status":"not_run_not_passed","status":"started","job_id":job,"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z")}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,"w") as h: json.dump(payload,h,indent=2,sort_keys=True); h.write("\n")
PY

"${PYTHON}" -B "${REPO}/scripts/09_k1245/60_build_k1245_component_pools.py" \
  --dataset-dir "${SOURCE}/nnUNet_raw/Dataset264_BraTS2026_MET_Completion" \
  --preprocessed-dir "${PARENT}/data/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres" \
  --train-file "${PARENT}/inputs/splits/train_fixed.txt" \
  --mapping-csv "${PARENT}/inputs/mapping/nnunet_case_mapping_master.csv" \
  --preprocessed-coordinate-audit "${PARENT}/route_rebuild_20260812_r1/preprocessed_coordinate_contract.json" \
  --output-root "${ROOT}/pools" \
  --plan "${ROOT}/K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.json"

"${PYTHON}" -B "${REPO}/scripts/09_k1245/61_prepare_k1245_configs.py" \
  --root "${ROOT}" \
  --legacy-route-config "${PARENT}/calibration_rebuild_20260819_r10/FIX_V3_ROUTE_CONFIG_SCHEMA5.json" \
  --legacy-component-manifest "${PARENT}/route_rebuild_20260812_r1/component_pool/component_manifest.json"

"${PYTHON}" - "${ROOT}" "${COMPLETE}" <<'PY'
import json, os, sys
from pathlib import Path
from custom_nnunet.met_aug_k1245 import K1245Manifest
root, complete = Path(sys.argv[1]), Path(sys.argv[2])
evidence=json.loads((root/'pools/K1245_POOL_AND_EMPIRICAL_VALIDATION.json').read_text())
for token in ('rc27','small8'): K1245Manifest.load(root/f'pools/{token}/component_manifest.json')
if evidence['rc_component_count'] < 24: raise SystemExit('insufficient RC donors')
fd=os.open(complete,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
with os.fdopen(fd,'w') as h: h.write('complete\n')
PY
