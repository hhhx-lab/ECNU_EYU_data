#!/bin/bash
#SBATCH --job-name=s2f3smoke
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/full200_smoke_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/full200_smoke_%j.err

set -euo pipefail
umask 077

ROOT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1"
REPO="${ROOT}/runtime/repository"
PYTHON="/public/home/zqchen/.conda/envs/segmamba/bin/python"
SOURCE="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency"
ROUTE_ROOT="${ROOT}/route_rebuild_20260812_r1"
CAL_ROOT="${ROOT}/calibration_rebuild_20260812_r1"
SMOKE_ROOT="${ROOT}/smoke_20260817_r1"
FAILED="${ROOT}/evidence/FULL200_SMOKE_FAILED.json"
VALIDATION="${ROOT}/evidence/FULL200_SMOKE_VALIDATION.json"
COMPLETE="${ROOT}/evidence/FULL200_SMOKE_COMPLETE.ok"
G1_CODE="${ROOT}/runtime/g1_archive/BraTS_2023_2024_solutions-main_3/Segmentation_Tasks/GliGAN"

write_failure() {
    local rc="$?"
    trap - ERR
    if [[ ! -e "${FAILED}" && ! -e "${COMPLETE}" ]]; then
        "${PYTHON}" - "${FAILED}" "${rc}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import json
import os
import sys

path, rc, job_id = sys.argv[1:]
payload = {
    "artifact_status": "experimental_unvalidated",
    "job_id": job_id,
    "return_code": int(rc),
    "status": "failed",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
PY
    fi
    exit "${rc}"
}
trap write_failure ERR

[[ -f "${CAL_ROOT}/CALIBRATION_REBUILD_COMPLETE.ok" ]] || { echo "Calibration rebuild is incomplete" >&2; exit 2; }
[[ -f "${CAL_ROOT}/CALIBRATION_REBUILD_VALIDATION.json" ]] || { echo "Calibration validation is missing" >&2; exit 2; }
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "Smoke output already exists" >&2; exit 2; }
[[ ! -e "${FAILED}" && ! -e "${VALIDATION}" && ! -e "${COMPLETE}" ]] || { echo "Smoke terminal evidence already exists" >&2; exit 2; }

module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:${PATH}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"${PYTHON}" "${REPO}/scripts/20_run_met_aug_training_smoke.py" \
    --nnunet-raw "${SOURCE}/nnUNet_raw" \
    --nnunet-preprocessed "${ROOT}/data/nnUNet_preprocessed" \
    --split-dir "${ROOT}/inputs/splits" \
    --pretrained-weights "${ROOT}/frozen/original_e/checkpoint_final.pth" \
    --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
    --route-config "${CAL_ROOT}/FIX_V3_ROUTE_CONFIG_SCHEMA5.json" \
    --valid-mask-manifest "${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json" \
    --route-approval "${CAL_ROOT}/FIX_V3_EMERGENCY_DECISION.json" \
    --fix-v3-calibration "${CAL_ROOT}/FIX_V2_CALIBRATION_ECNU_FROZEN.json" \
    --fix-v2-failure-audit "${ROOT}/frozen/fix_v2/QC_HOLDOUT_RUN_ATTEMPT_02_FAILED.json" \
    --g1-code-dir "${G1_CODE}" \
    --g1-checkpoint-root "${ROOT}/frozen/g1" \
    --g1-checkpoint-selection "${ROOT}/frozen/g1/checkpoint_selection.json" \
    --g2-parent-gate "${ROOT}/frozen/g1/g2_diffusion_qc_gate.json" \
    --output-dir "${SMOKE_ROOT}" \
    --dataset 264 \
    --configuration 3d_fullres \
    --fold 0 \
    --plans nnUNetPlans \
    --trainer nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune \
    --min-steps 4 \
    --max-steps 12 \
    --min-committed-events 1 \
    --min-gpu-memory-gib 30

"${PYTHON}" - "${SMOKE_ROOT}/met_aug_training_smoke_report.json" "${VALIDATION}" "${COMPLETE}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import hashlib
import json
import math
import os
import sys

report_path, validation_path, marker_path, job_id = sys.argv[1:]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("status") != "pass":
    raise SystemExit("smoke report did not pass")
steps = report.get("steps", [])
if len(steps) < 4 or len(steps) > 12:
    raise SystemExit("smoke step count violates contract")
if int(report.get("audit", {}).get("committed_events", 0)) < 1:
    raise SystemExit("smoke has no COMMITTED event")
if any(not math.isfinite(float(step.get("loss", float("nan")))) for step in steps):
    raise SystemExit("smoke contains non-finite loss")
if report.get("checkpoint_files") or report.get("validation_executed") or report.get("checkpoint_saved"):
    raise SystemExit("smoke escaped its no-checkpoint/no-validation contract")
payload = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "status": "pass",
    "job_id": job_id,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "report": report_path,
    "report_sha256": hashlib.sha256(open(report_path, "rb").read()).hexdigest(),
    "steps": len(steps),
    "committed_events": int(report["audit"]["committed_events"]),
    "device_name": report["device"]["name"],
    "peak_reserved_gib": max(float(step["cuda"]["peak_reserved_gib"]) for step in steps),
    "estimated_200_epochs_hours": float(report["timing"]["estimated_200_epochs_hours"]),
}
fd = os.open(validation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write("experimental_unvalidated A100 smoke pass\n")
PY

trap - ERR
echo "S2_FIX_V3_FULL200_SMOKE_PASS"
