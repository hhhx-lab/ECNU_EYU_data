#!/bin/bash
#SBATCH --job-name=s2f3_1mm
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/preprocess_true1mm_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/preprocess_true1mm_%j.err

set -euo pipefail
umask 077

ROOT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1"
SOURCE="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency"
DATASET="Dataset264_BraTS2026_MET_Completion"
RAW_DATASET="${SOURCE}/nnUNet_raw/${DATASET}"
PREPROCESSED_DATASET="${ROOT}/data/nnUNet_preprocessed/${DATASET}"
PREPROCESSED_CONFIG="${PREPROCESSED_DATASET}/nnUNetPlans_3d_fullres"
REPOSITORY="${ROOT}/runtime/repository"
PYTHON="/public/home/zqchen/.conda/envs/segmamba/bin/python"
ORIGINAL_E="${ROOT}/frozen/original_e/checkpoint_final.pth"
FAILED="${ROOT}/evidence/PREPROCESS_TRUE1MM_FAILED.json"
COMPLETE="${ROOT}/evidence/PREPROCESS_TRUE1MM_COMPLETE.ok"

write_failure() {
    local rc="$?"
    trap - ERR
    if [[ ! -e "${FAILED}" && ! -e "${COMPLETE}" ]]; then
        "${PYTHON}" - "${FAILED}" "${rc}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "artifact_status": "experimental_unvalidated",
    "job_id": sys.argv[3],
    "return_code": int(sys.argv[2]),
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

[[ -d "${ROOT}" ]] || { echo "Missing exclusive experiment root: ${ROOT}" >&2; exit 2; }
[[ -f "${ROOT}/EXPERIMENT_CONTRACT.json" ]] || { echo "Missing experiment contract" >&2; exit 2; }
[[ ! -e "${FAILED}" ]] || { echo "Failure evidence already exists: ${FAILED}" >&2; exit 2; }
[[ ! -e "${COMPLETE}" ]] || { echo "Preprocessing already completed: ${COMPLETE}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Missing project Python: ${PYTHON}" >&2; exit 2; }
[[ -f "${ORIGINAL_E}" ]] || { echo "Missing frozen original E: ${ORIGINAL_E}" >&2; exit 2; }
[[ -f "${PREPROCESSED_DATASET}/dataset.json" ]] || { echo "Missing staged dataset.json" >&2; exit 2; }
[[ -f "${PREPROCESSED_DATASET}/dataset_fingerprint.json" ]] || { echo "Missing staged dataset fingerprint" >&2; exit 2; }
[[ -f "${PREPROCESSED_DATASET}/nnUNetPlans.json" ]] || { echo "Missing staged plans" >&2; exit 2; }
[[ -f "${REPOSITORY}/scripts/21_validate_preprocessed_coordinate_contract.py" ]] || {
    echo "Missing coordinate validator" >&2
    exit 2
}

module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:${PATH}"
export PYTHONPATH="${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_raw="${SOURCE}/nnUNet_raw"
export nnUNet_preprocessed="${ROOT}/data/nnUNet_preprocessed"
export nnUNet_results="${ROOT}/data/nnUNet_results"

"${PYTHON}" - "${ORIGINAL_E}" "${PREPROCESSED_DATASET}/nnUNetPlans.json" <<'PY'
import hashlib
import json
import sys

import torch

checkpoint_path, plans_path = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint.get("current_epoch") != 200:
    raise SystemExit("original E checkpoint is not the frozen epoch-200 artifact")
if checkpoint.get("trainer_name") != "nnUNetTrainerBraTS2026RCFocalCompletionFineTune":
    raise SystemExit("original E trainer identity drifted")
checkpoint_plans = checkpoint.get("init_args", {}).get("plans")
disk_plans = json.loads(open(plans_path, encoding="utf-8").read())
canonical = lambda value: hashlib.sha256(
    json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
if canonical(checkpoint_plans) != canonical(disk_plans):
    raise SystemExit("staged plans do not match frozen original E")
spacing = disk_plans["configurations"]["3d_fullres"]["spacing"]
if [float(value) for value in spacing] != [1.0, 1.0, 1.0]:
    raise SystemExit(f"3d_fullres spacing is not true 1 mm: {spacing}")
print("S2_FIX_V3_FULL200_PLANS_PREFLIGHT_PASS")
PY

[[ "$(sha256sum "${ORIGINAL_E}" | awk '{print $1}')" == "4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267" ]] || {
    echo "Frozen original E SHA256 mismatch" >&2
    exit 2
}
[[ "$(find -L "${RAW_DATASET}/imagesTr" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)" -eq 4552 ]] || {
    echo "Expected 4552 readable training modality files" >&2
    exit 2
}
[[ "$(find -L "${RAW_DATASET}/labelsTr" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)" -eq 1138 ]] || {
    echo "Expected 1138 readable training labels" >&2
    exit 2
}

nnUNetv2_preprocess \
    -d 264 \
    -plans_name nnUNetPlans \
    -c 3d_fullres \
    -np 16 \
    --no_pbar

[[ "$(find "${PREPROCESSED_CONFIG}" -maxdepth 1 -type f -name '*.pkl' | wc -l)" -eq 1138 ]] || {
    echo "Expected 1138 pkl files" >&2
    exit 2
}
[[ "$(find "${PREPROCESSED_CONFIG}" -maxdepth 1 -type f -name '*.b2nd' ! -name '*_seg.b2nd' | wc -l)" -eq 1138 ]] || {
    echo "Expected 1138 image b2nd files" >&2
    exit 2
}
[[ "$(find "${PREPROCESSED_CONFIG}" -maxdepth 1 -type f -name '*_seg.b2nd' | wc -l)" -eq 1138 ]] || {
    echo "Expected 1138 segmentation b2nd files" >&2
    exit 2
}

"${PYTHON}" "${REPOSITORY}/scripts/21_validate_preprocessed_coordinate_contract.py" \
    --dataset-dir "${RAW_DATASET}" \
    --preprocessed-dir "${PREPROCESSED_CONFIG}" \
    --output "${ROOT}/evidence/preprocessed_coordinate_contract.json" \
    --required-spacing-mm 1 1 1

"${PYTHON}" - "${ROOT}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
job_id = sys.argv[2]
audit = root / "evidence" / "preprocessed_coordinate_contract.json"
payload = json.loads(audit.read_text(encoding="utf-8"))
if payload.get("status") != "pass" or payload.get("case_count") != 1138:
    raise SystemExit("coordinate audit did not establish the 1138-case contract")
validation = {
    "artifact_status": "experimental_unvalidated",
    "case_count": 1138,
    "configuration_spacing_mm": [1.0, 1.0, 1.0],
    "coordinate_audit": str(audit),
    "coordinate_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
    "job_id": job_id,
    "status": "pass",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
}
target = root / "evidence" / "PREPROCESS_TRUE1MM_VALIDATION.json"
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(validation, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
marker = root / "evidence" / "PREPROCESS_TRUE1MM_COMPLETE.ok"
fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write("experimental_unvalidated true-1mm preprocessing pass\n")
PY

trap - ERR
echo "S2_FIX_V3_FULL200_TRUE1MM_PREPROCESS_PASS"
