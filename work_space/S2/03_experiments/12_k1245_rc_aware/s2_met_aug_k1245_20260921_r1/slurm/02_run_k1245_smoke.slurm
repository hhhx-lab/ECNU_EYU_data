#!/bin/bash
#SBATCH --job-name=s2k1245smkr1
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260921_r1/logs/%x_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260921_r1/logs/%x_%j.err

set -euo pipefail
umask 077

ROOT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260921_r1
PARENT=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1
PYTHON=/public/home/zqchen/.conda/envs/segmamba/bin/python
REPO="${ROOT}/runtime/repository"
SOURCE=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency
PREPROCESSED="${PARENT}/data/nnUNet_preprocessed/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres"
VALID_MASKS="${PARENT}/route_rebuild_20260812_r1/valid_masks/valid_mask_manifest.json"
LEGACY_MANIFEST="${PARENT}/route_rebuild_20260812_r1/component_pool/component_manifest.json"
CALIBRATION="${PARENT}/calibration_rebuild_20260819_r10/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
G1_CODE="${PARENT}/runtime/g1_archive/BraTS_2023_2024_solutions-main_3/Segmentation_Tasks/GliGAN"
G1_ROOT="${PARENT}/frozen/g1"
G1_SELECTION="${G1_ROOT}/checkpoint_selection.json"
G2_GATE="${G1_ROOT}/g2_diffusion_qc_gate.json"

FAILED="${ROOT}/evidence/K1245_SMOKE_FAILED.json"
STARTED="${ROOT}/evidence/K1245_SMOKE_STARTED.json"
VALIDATION="${ROOT}/evidence/K1245_SMOKE_VALIDATION.json"
COMPLETE="${ROOT}/evidence/K1245_SMOKE_COMPLETE.ok"

fail() {
  rc=$?
  trap - ERR
  if [[ ! -e "${FAILED}" && ! -e "${COMPLETE}" ]]; then
    "${PYTHON}" - "${FAILED}" "${rc}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt, json, os, sys
path, rc, job = sys.argv[1:]
payload = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "formal_gate_status": "not_run_not_passed",
    "status": "failed",
    "return_code": int(rc),
    "job_id": job,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  fi
  exit "${rc}"
}
trap fail ERR

[[ -f "${ROOT}/evidence/POOL_BUILD_COMPLETE.ok" ]] || { echo pool_incomplete >&2; exit 2; }
[[ ! -e "${ROOT}/evidence/POOL_BUILD_FAILED.json" ]] || { echo pool_failed >&2; exit 2; }
[[ -f "${ROOT}/evidence/REMOTE_STATIC_PREFLIGHT.json" ]] || { echo preflight_missing >&2; exit 2; }
[[ -f "${ROOT}/evidence/K1245_SMOKE_STATIC_PREFLIGHT.json" ]] || { echo smoke_preflight_missing >&2; exit 2; }
[[ -f "${ROOT}/config/K1245_ROUTE_K2.json" && -f "${ROOT}/config/K1245_ROUTE_K4.json" ]] || { echo configs_missing >&2; exit 2; }
[[ ! -e "${STARTED}" && ! -e "${FAILED}" && ! -e "${VALIDATION}" && ! -e "${COMPLETE}" ]] || { echo smoke_evidence_exists >&2; exit 2; }

module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" - "${STARTED}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt, json, os, sys
path, job = sys.argv[1:]
payload = {"artifact_status":"experimental_unvalidated", "operator_approved":False,
           "formal_gate_status":"not_run_not_passed", "status":"started",
           "job_id":job, "timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z")}
fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o400)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY

"${PYTHON}" -B "${REPO}/scripts/09_k1245/62_run_k1245_smoke.py" \
  --root "${ROOT}" \
  --preprocessed-dir "${PREPROCESSED}" \
  --valid-mask-manifest "${VALID_MASKS}" \
  --legacy-manifest "${LEGACY_MANIFEST}" \
  --calibration "${CALIBRATION}" \
  --g1-code-dir "${G1_CODE}" \
  --g1-checkpoint-root "${G1_ROOT}" \
  --g1-checkpoint-selection "${G1_SELECTION}" \
  --g2-parent-gate "${G2_GATE}" \
  --device cuda

"${PYTHON}" - "${ROOT}" "${VALIDATION}" "${COMPLETE}" <<'PY'
import datetime as dt, json, os, sys
from pathlib import Path
root, validation, complete = map(Path, sys.argv[1:])
report = json.loads((root / "evidence/K1245_SMOKE_AUTOMATIC.json").read_text())
if report.get("status") != "hold_for_manual_review" or report.get("automatic_status") != "pass":
    raise SystemExit("automatic smoke did not pass")
for kind, required in (("rc24", 24), ("small24", 24)):
    item = report["reports"][kind]
    if item.get("automatic_pass_count") != required:
        raise SystemExit(f"{kind} automatic count mismatch")
payload = dict(report)
payload.update({"status":"hold_for_manual_review", "manual_review_required":True,
                "validation_timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00","Z")})
fd=os.open(validation, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o400)
with os.fdopen(fd,"w") as handle: json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY
# Deliberately no COMPLETE marker: manual acceptance is a preregistered gate.
