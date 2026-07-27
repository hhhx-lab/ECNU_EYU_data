#!/bin/bash
#SBATCH --job-name=s2_cache_gate
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=work_space/S2/logs/s2_cache_gate_%j.out
#SBATCH --error=work_space/S2/logs/s2_cache_gate_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/${USER}/projects/ECNU_EYU_data}"
S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
S2_STORAGE_ROOT="${S2_STORAGE_ROOT:-${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency}"
DATASET_NAME=Dataset264_BraTS2026_MET_Completion
DATASET_DIR="${S2_STORAGE_ROOT}/nnUNet_raw/${DATASET_NAME}"
PREPROCESSED_ROOT="${S2_STORAGE_ROOT}/nnUNet_preprocessed/${DATASET_NAME}"
PREPROCESSED_DIR="${PREPROCESSED_ROOT}/nnUNetPlans_3d_fullres"
SPLIT_DIR="${S2_REPO}/data/splits/completion_warmstart"
BASELINE_CHECKPOINT="${S2_PRETRAINED_WEIGHTS:-${PROJECT_ROOT}/work_space/S2/results/checkpoint_final.pth}"
OUTPUT_AUDIT="${SPLIT_DIR}/fixed_split_cache_audit.json"

required=(
    "${S2_REPO}/scripts/09_clone_baseline_plans.py"
    "${S2_REPO}/scripts/05_validate_fixed_split_cache.py"
    "${BASELINE_CHECKPOINT}"
    "${DATASET_DIR}/dataset.json"
    "${SPLIT_DIR}/train_fixed.txt"
    "${SPLIT_DIR}/val_fixed.txt"
    "${SPLIT_DIR}/test_locked.txt"
)
for path in "${required[@]}"; do
    [[ -s "${path}" ]] || { echo "Missing required artifact: ${path}" >&2; exit 1; }
done
[[ -d "${PREPROCESSED_DIR}" ]] || { echo "Missing preprocessed cache: ${PREPROCESSED_DIR}" >&2; exit 1; }

module purge
module load apps/envs/miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${S2_CONDA_ENV:-segmamba}"

python "${S2_REPO}/scripts/09_clone_baseline_plans.py" \
    --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
    --source-dataset-json "${DATASET_DIR}/dataset.json" \
    --target-preprocessed-dir "${PREPROCESSED_ROOT}" \
    --target-dataset-name "${DATASET_NAME}" \
    --expected-num-training 1138

python "${S2_REPO}/scripts/05_validate_fixed_split_cache.py" \
    --train-file "${SPLIT_DIR}/train_fixed.txt" \
    --val-file "${SPLIT_DIR}/val_fixed.txt" \
    --dataset-dir "${DATASET_DIR}" \
    --preprocessed-dir "${PREPROCESSED_DIR}" \
    --output-json "${OUTPUT_AUDIT}"

python - "${DATASET_DIR}/dataset.json" "${SPLIT_DIR}" "${OUTPUT_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
split_dir = Path(sys.argv[2])
audit_path = Path(sys.argv[3])
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
stats = dataset.get("g2_materialization_stats", {})
expected_stats = {
    "included_cases": 1242,
    "completion_replacements": 212,
}
for key, expected in expected_stats.items():
    actual = stats.get(key)
    if actual != expected:
        raise SystemExit(f"Dataset264 materialization mismatch: {key}={actual}, expected={expected}")

expected_split = {
    "train_fixed.txt": 1035,
    "val_fixed.txt": 103,
    "test_locked.txt": 104,
}
for filename, expected in expected_split.items():
    values = [
        line.strip()
        for line in (split_dir / filename).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != expected or len(values) != len(set(values)):
        raise SystemExit(
            f"Split mismatch: {filename} count={len(values)} unique={len(set(values))} expected={expected}"
        )

audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit != {
    "status": "pass",
    "train_count": 1035,
    "validation_count": 103,
    "dataset_count": 1138,
    "preprocessed_count": 1138,
    "cache_format": "b2nd",
    "id_sets_equal": True,
}:
    raise SystemExit(f"Unexpected cache audit: {audit}")
print("COMPLETION_EXISTING_CACHE_PASS included_cases=1242 completion_paths_overridden=212 split=1035/103/104")
PY

AVAILABLE_KIB=$(df -Pk /public | awk 'NR == 2 {print $4}')
MINIMUM_KIB=$((1024 * 1024 * 1024))
if (( AVAILABLE_KIB < MINIMUM_KIB )); then
    echo "/public has less than 1 TiB free: ${AVAILABLE_KIB} KiB" >&2
    exit 1
fi

printf 'job_id=%s\ndataset=%s\ncache_audit=%s\n' \
    "${SLURM_JOB_ID}" "${DATASET_NAME}" "${OUTPUT_AUDIT}" \
    > "${S2_STORAGE_ROOT}/S2_COMPLETION_PREP_PASS.ok"
echo "S2_COMPLETION_PREP_PASS dataset=${DATASET_NAME} storage=${S2_STORAGE_ROOT} reused_cache=true"
