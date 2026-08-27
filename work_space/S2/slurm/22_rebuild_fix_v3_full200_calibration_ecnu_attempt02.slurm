#!/bin/bash
#SBATCH --job-name=s2f3cal2
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/calibration_rebuild_attempt02_%j.out
#SBATCH --error=/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1/logs/calibration_rebuild_attempt02_%j.err

set -euo pipefail
umask 077

ROOT="/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1"
REPO="${ROOT}/runtime/repository"
PYTHON="/public/home/zqchen/.conda/envs/segmamba/bin/python"
DATASET="Dataset264_BraTS2026_MET_Completion"
PREPROCESSED="${ROOT}/data/nnUNet_preprocessed/${DATASET}/nnUNetPlans_3d_fullres"
ROUTE_ROOT="${ROOT}/route_rebuild_20260812_r1"
CAL_ROOT="${ROOT}/calibration_rebuild_20260817_r2"
HISTORY="${ROOT}/frozen/calibration_history"
MANIFEST="${ROUTE_ROOT}/component_pool/component_manifest.json"
VALID_MASK="${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json"
PARTITION="${CAL_ROOT}/FIX_V2_PARTITION_AUDIT.json"
REFERENCE="${CAL_ROOT}/REFERENCE_CDF.json"
REFERENCE_VALIDATION="${CAL_ROOT}/REFERENCE_CDF_VALIDATION.json"
MEASUREMENT_DIR="${CAL_ROOT}/measurement_configs"
SELECTION_DIR="${CAL_ROOT}/selection_derivation"
CALIBRATION="${CAL_ROOT}/FIX_V2_CALIBRATION_ECNU_FROZEN.json"
FIX_V3_CONFIG="${CAL_ROOT}/FIX_V3_ROUTE_CONFIG_SCHEMA5.json"
EMERGENCY_DECISION="${CAL_ROOT}/FIX_V3_EMERGENCY_DECISION.json"
FAILED="${CAL_ROOT}/CALIBRATION_REBUILD_FAILED.json"
COMPLETE="${CAL_ROOT}/CALIBRATION_REBUILD_COMPLETE.ok"

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

[[ -d "${ROOT}" && -d "${CAL_ROOT}" ]] || { echo "Missing experiment/calibration root" >&2; exit 2; }
[[ -f "${ROOT}/evidence/PREPROCESS_TRUE1MM_COMPLETE.ok" ]] || { echo "true-1mm preprocessing is incomplete" >&2; exit 2; }
[[ -f "${ROUTE_ROOT}/PREPARE_21_12_13_14_COMPLETE.ok" ]] || { echo "route rebuild is incomplete" >&2; exit 2; }
[[ ! -e "${FAILED}" && ! -e "${COMPLETE}" ]] || { echo "Calibration terminal evidence already exists" >&2; exit 2; }
[[ "$(find "${CAL_ROOT}" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')" == "0" ]] || {
    echo "Calibration root is not empty; refusing overwrite" >&2
    exit 2
}

for path in \
    "${MANIFEST}" \
    "${VALID_MASK}" \
    "${ROOT}/EXPERIMENT_CONTRACT.json" \
    "${ROOT}/frozen/original_e/checkpoint_final.pth" \
    "${ROOT}/frozen/fix_v2/QC_HOLDOUT_RUN_ATTEMPT_02_FAILED.json" \
    "${HISTORY}/DEVELOPMENT_RUN_ATTEMPT_01_VALIDATION.json" \
    "${HISTORY}/DEVELOPMENT_RUN_ATTEMPT_01_FILES.sha256" \
    "${HISTORY}/DEVELOPMENT_BLINDED_REVIEW_DECISIONS_ATTEMPT_02.csv" \
    "${HISTORY}/DEVELOPMENT_BLINDED_REVIEW_ATTEMPT_02_VALIDATION.json"; do
    [[ -f "${path}" ]] || { echo "Missing required frozen input: ${path}" >&2; exit 2; }
done

module purge
module load apps/envs/miniconda3/25.5.1
export PATH="/public/home/zqchen/.conda/envs/segmamba/bin:${PATH}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

"${PYTHON}" "${REPO}/scripts/29_make_met_aug_fix_v2_partitions.py" \
    --component-manifest "${MANIFEST}" \
    --output "${PARTITION}" \
    --seed 20260728 \
    --reference-fraction 0.70 \
    --development-fraction 0.15

"${PYTHON}" "${REPO}/scripts/33_extract_met_aug_fix_v2_reference.py" \
    --component-manifest "${MANIFEST}" \
    --partition-audit "${PARTITION}" \
    --valid-mask-manifest "${VALID_MASK}" \
    --preprocessed-dir "${PREPROCESSED}" \
    --output "${REFERENCE}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

"${PYTHON}" "${REPO}/scripts/37_validate_met_aug_fix_v2_reference.py" \
    --component-manifest "${MANIFEST}" \
    --partition-audit "${PARTITION}" \
    --valid-mask-manifest "${VALID_MASK}" \
    --preprocessed-dir "${PREPROCESSED}" \
    --reference-cdf "${REFERENCE}" \
    --output "${REFERENCE_VALIDATION}"

"${PYTHON}" "${REPO}/scripts/34_make_met_aug_fix_v2_measurement_configs.py" \
    --component-manifest "${MANIFEST}" \
    --partition-audit "${PARTITION}" \
    --reference-cdf "${REFERENCE}" \
    --reference-validation "${REFERENCE_VALIDATION}" \
    --output-dir "${MEASUREMENT_DIR}" \
    --seed 20260725

"${PYTHON}" "${REPO}/scripts/38_select_and_derive_met_aug_fix_v2.py" \
    --component-manifest "${MANIFEST}" \
    --reference-cdf "${REFERENCE}" \
    --reference-validation "${REFERENCE_VALIDATION}" \
    --measurement-config-index "${MEASUREMENT_DIR}/MEASUREMENT_CONFIG_INDEX.json" \
    --development-output "${ROOT}/frozen/development_attempt_01" \
    --development-validation "${HISTORY}/DEVELOPMENT_RUN_ATTEMPT_01_VALIDATION.json" \
    --development-files-manifest "${HISTORY}/DEVELOPMENT_RUN_ATTEMPT_01_FILES.sha256" \
    --review-decisions "${HISTORY}/DEVELOPMENT_BLINDED_REVIEW_DECISIONS_ATTEMPT_02.csv" \
    --review-validation "${HISTORY}/DEVELOPMENT_BLINDED_REVIEW_ATTEMPT_02_VALIDATION.json" \
    --output-dir "${SELECTION_DIR}" \
    --seed 20260725

read -r PATIENT_GROUP_COUNT COMPONENT_COUNT < <(
    "${PYTHON}" - "${PARTITION}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(payload["patient_group_count"]), int(payload["component_count"]))
PY
)

"${PYTHON}" "${REPO}/scripts/30_freeze_met_aug_fix_v2_calibration.py" \
    --draft "${SELECTION_DIR}/FIX_V2_CALIBRATION_CANDIDATE_BEFORE_QC.json" \
    --partition-audit "${PARTITION}" \
    --reference-cdf "${REFERENCE}" \
    --patient-group-count "${PATIENT_GROUP_COUNT}" \
    --component-count "${COMPONENT_COUNT}" \
    --boundary-policy label_only_qc_v1 \
    --output "${CALIBRATION}"

"${PYTHON}" - \
    "${ROOT}" "${MANIFEST}" "${VALID_MASK}" "${PARTITION}" "${REFERENCE}" \
    "${REFERENCE_VALIDATION}" "${MEASUREMENT_DIR}/MEASUREMENT_CONFIG_INDEX.json" \
    "${SELECTION_DIR}/SELECTION_AND_DERIVATION_VALIDATION.json" "${CALIBRATION}" \
    "${FIX_V3_CONFIG}" "${EMERGENCY_DECISION}" "${SLURM_JOB_ID:-unknown}" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

from custom_nnunet.met_aug_core import (
    ComponentManifest,
    RouteConfig,
    canonical_json_sha256,
    make_fix_v3_route_a_config,
    sha256_file,
)
from custom_nnunet.met_aug_fix_v2 import FixV2Calibration
from custom_nnunet.met_aug_fix_v3_emergency import (
    make_fix_v3_emergency_decision,
    validate_fix_v3_emergency_decision,
)

(
    root_value,
    manifest_value,
    valid_mask_value,
    partition_value,
    reference_value,
    reference_validation_value,
    measurement_index_value,
    selection_validation_value,
    calibration_value,
    config_value,
    decision_value,
    job_id,
) = sys.argv[1:]
root = Path(root_value)
manifest_path = Path(manifest_value)
valid_mask_path = Path(valid_mask_value)
partition_path = Path(partition_value)
reference_path = Path(reference_value)
reference_validation_path = Path(reference_validation_value)
measurement_index_path = Path(measurement_index_value)
selection_validation_path = Path(selection_validation_value)
calibration_path = Path(calibration_value)
config_path = Path(config_value)
decision_path = Path(decision_value)
cal_root = calibration_path.parent

def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def write_exclusive(path, payload, mode=0o400):
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded)

manifest = ComponentManifest.load(manifest_path)
calibration = FixV2Calibration.load(calibration_path, expected_policy="label_only_qc_v1")
config = make_fix_v3_route_a_config(
    manifest,
    boundary_policy=calibration.boundary_policy,
    calibration_sha256=calibration.sha256,
    seed=20260725,
)
write_exclusive(config_path, config)
RouteConfig.load(config_path, manifest)

original_e = root / "frozen/original_e/checkpoint_final.pth"
fix_v2_failure = root / "frozen/fix_v2/QC_HOLDOUT_RUN_ATTEMPT_02_FAILED.json"
decision_text = make_fix_v3_emergency_decision(
    component_manifest=manifest_path,
    route_config=config_path,
    valid_mask_manifest=valid_mask_path,
    calibration=calibration_path,
    original_e=original_e,
    fix_v2_failure=fix_v2_failure,
)
fd = os.open(decision_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write(decision_text)
validate_fix_v3_emergency_decision(
    decision_path,
    component_manifest=manifest_path,
    route_config=config_path,
    valid_mask_manifest=valid_mask_path,
    calibration=calibration_path,
    original_e=original_e,
    fix_v2_failure=fix_v2_failure,
)

contract = read_json(root / "EXPERIMENT_CONTRACT.json")
development_validation_path = root / "frozen/calibration_history/DEVELOPMENT_RUN_ATTEMPT_01_VALIDATION.json"
development_validation = read_json(development_validation_path)
development_rows_path = root / "frozen/development_attempt_01/development_measurements.jsonl"
development_rows = [
    json.loads(line)
    for line in development_rows_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
manifest_component_ids = {record.component_id for record in manifest.records}
development_donor_ids = {str(row["donor_component_id"]) for row in development_rows}
missing_donors = sorted(development_donor_ids - manifest_component_ids)
if missing_donors:
    raise RuntimeError(f"rebuilt manifest misses historical Development donors: {missing_donors[:10]}")

selection_validation = read_json(selection_validation_path)
if selection_validation.get("status") != "pass":
    raise RuntimeError("selection/derivation validation did not pass")
if int(selection_validation.get("strict_replay_pass_count", -1)) != 46:
    raise RuntimeError("strict replay pass count drifted")
if int(selection_validation.get("strict_replay_reject_count", -1)) != 2:
    raise RuntimeError("strict replay reject count drifted")

reference_validation = read_json(reference_validation_path)
measurement_index = read_json(measurement_index_path)
partition = read_json(partition_path)
if reference_validation.get("status") != "pass":
    raise RuntimeError("Reference validation did not pass")
if measurement_index.get("status") != "measurement_only_not_gate_eligible":
    raise RuntimeError("measurement index role drifted")
if partition.get("status") != "pass":
    raise RuntimeError("partition audit did not pass")

relocation = {
    "artifact_status": "experimental_unvalidated",
    "status": "pass",
    "audit_scope": "deterministic_rebuild_and_historical_development_replay_compatibility",
    "old_absolute_path_artifacts_reused": False,
    "old_calibration_reused": False,
    "old_sha_anchors_preserved": contract["frozen_expected_sha256"],
    "historical_component_manifest_identity_sha256": development_validation["component_manifest_identity_sha256"],
    "rebuilt_component_manifest_identity_sha256": manifest.identity_sha256,
    "identity_change_expected_due_to_rebuild": development_validation["component_manifest_identity_sha256"] != manifest.identity_sha256,
    "fixed_train_sha256": sha256_file(root / "inputs/splits/train_fixed.txt"),
    "fixed_validation_sha256": sha256_file(root / "inputs/splits/val_fixed.txt"),
    "development_row_count": len(development_rows),
    "development_unique_donor_count": len(development_donor_ids),
    "development_missing_donor_count": len(missing_donors),
    "strict_replay_pass_count": selection_validation["strict_replay_pass_count"],
    "strict_replay_reject_count": selection_validation["strict_replay_reject_count"],
    "semantic_equivalence_limit": "Compatibility is established for the frozen split and all historical Development replay donors; old and rebuilt file SHA values are intentionally not claimed equal.",
}
relocation["audit_sha256"] = canonical_json_sha256(relocation, exclude=("audit_sha256",))
relocation_path = cal_root / "RELOCATION_SEMANTIC_EQUIVALENCE_AUDIT.json"
write_exclusive(relocation_path, relocation)

validation = {
    "artifact_status": "experimental_unvalidated",
    "operator_approved": False,
    "status": "pass",
    "job_id": job_id,
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "component_count": len(manifest.records),
    "target_case_count": len(manifest.target_groups),
    "patient_group_count": int(partition["patient_group_count"]),
    "reference_validation_status": reference_validation["status"],
    "development_replay_status": selection_validation["status"],
    "fix_v3_schema_version": int(config["schema_version"]),
    "formal_gate_status": "not_run_not_passed",
    "inputs": {
        "component_manifest_sha256": sha256_file(manifest_path),
        "valid_mask_manifest_sha256": sha256_file(valid_mask_path),
        "partition_sha256": sha256_file(partition_path),
        "reference_cdf_sha256": sha256_file(reference_path),
        "reference_validation_sha256": sha256_file(reference_validation_path),
        "measurement_index_sha256": sha256_file(measurement_index_path),
        "selection_validation_sha256": sha256_file(selection_validation_path),
        "calibration_sha256": sha256_file(calibration_path),
        "fix_v3_config_sha256": sha256_file(config_path),
        "emergency_decision_sha256": sha256_file(decision_path),
        "relocation_audit_sha256": sha256_file(relocation_path),
    },
}
validation["validation_audit_sha256"] = canonical_json_sha256(
    validation, exclude=("validation_audit_sha256",)
)
write_exclusive(cal_root / "CALIBRATION_REBUILD_VALIDATION.json", validation)
PY

(
    cd "${CAL_ROOT}"
    find . -type f \
        ! -name 'CALIBRATION_REBUILD_SHA256SUMS.txt' \
        ! -name 'CALIBRATION_REBUILD_COMPLETE.ok' \
        ! -name 'CALIBRATION_REBUILD_FAILED.json' \
        -print0 | sort -z | xargs -0 sha256sum > CALIBRATION_REBUILD_SHA256SUMS.txt
    chmod 400 CALIBRATION_REBUILD_SHA256SUMS.txt
)

"${PYTHON}" - "${COMPLETE}" <<'PY'
import os
import sys

fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    handle.write("experimental_unvalidated ECNU calibration rebuild pass\n")
PY

trap - ERR
echo "S2_FIX_V3_FULL200_CALIBRATION_REBUILD_PASS"
