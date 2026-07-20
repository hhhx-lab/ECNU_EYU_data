#!/bin/bash
#SBATCH --job-name=s2_comp_prep
#SBATCH --partition=cpu_192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=work_space/S2/logs/s2_completion_prepare_%j.out
#SBATCH --error=work_space/S2/logs/s2_completion_prepare_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/${USER}/projects/ECNU_EYU_data}"
S2_HPC_USER_ROOT="${S2_HPC_USER_ROOT:-/hpc_stor/data/${USER}}"
S2_STORAGE_ROOT="${S2_STORAGE_ROOT:-${S2_HPC_USER_ROOT}/ECNU_EYU_data/work_space/S2/data/ecnu_completion}"
STORAGE_POLICY="${PROJECT_ROOT}/work_space/S2/slurm/s2_storage_policy.sh"
S2_REPO="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
G2_RESULTS="${PROJECT_ROOT}/work_space/G2/results"
G2_MATERIALIZER="${PROJECT_ROOT}/work_space/G2/code/g2_materialize_nnunet_dataset.py"
COMPLETION_ROOT="${COMPLETION_ROOT:-${PROJECT_ROOT}/work_space/G1/code/brats2025-latent-ensemble-synthesis-main-v3/data/output/run_3104668}"
ACCEPTED_MANIFEST="${ACCEPTED_MANIFEST:-${G2_RESULTS}/manifests/synthetic_accepted_manifest_run_3104668.csv}"
REAL_MAPPING="${REAL_MAPPING:-${G2_RESULTS}/manifests/nnunet_case_mapping_master.csv}"
REAL_DATA_ROOT="${REAL_DATA_ROOT:-/public/home/${USER}/data}"
MASTER_SPLIT="${MASTER_SPLIT:-${G2_RESULTS}/splits/splits_master_train_val_test.json}"
FAKE_T2W_CASES="${FAKE_T2W_CASES:-${G2_RESULTS}/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${PROJECT_ROOT}/work_space/S2/results/checkpoint_final.pth}"

[[ -f "${STORAGE_POLICY}" ]] || { echo "Missing storage policy: ${STORAGE_POLICY}" >&2; exit 1; }
source "${STORAGE_POLICY}"
s2_configure_storage_policy
trap s2_stop_storage_watchdog EXIT
trap 's2_stop_storage_watchdog; exit 130' INT
trap 's2_stop_storage_watchdog; exit 143' TERM

export nnUNet_raw="${S2_STORAGE_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${S2_STORAGE_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${S2_STORAGE_ROOT}/nnUNet_results"
export BRATS_SPLIT_DIR="${S2_REPO}/data/splits/completion_warmstart"
DATASET_NAME="Dataset264_BraTS2026_MET_Completion"
DATASET_DIR="${nnUNet_raw}/${DATASET_NAME}"

required_files=(
    "${G2_MATERIALIZER}"
    "${ACCEPTED_MANIFEST}"
    "${REAL_MAPPING}"
    "${MASTER_SPLIT}"
    "${FAKE_T2W_CASES}"
    "${BASELINE_CHECKPOINT}"
    "${S2_REPO}/scripts/08_prepare_completion_online_split.py"
    "${S2_REPO}/scripts/09_clone_baseline_plans.py"
    "${S2_REPO}/scripts/05_validate_fixed_split_cache.py"
)
for path in "${required_files[@]}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
[[ -d "${COMPLETION_ROOT}" ]] || { echo "Missing completion root: ${COMPLETION_ROOT}" >&2; exit 1; }
[[ -d "${REAL_DATA_ROOT}" ]] || { echo "Missing real data root: ${REAL_DATA_ROOT}" >&2; exit 1; }
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" "${BRATS_SPLIT_DIR}"
s2_write_public_emergency_marker
s2_start_storage_watchdog

module purge
module load apps/envs/miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${S2_CONDA_ENV:-segmamba}"

python - <<'PY'
import importlib.metadata
import nibabel
import torch

print("python/torch:", torch.__version__)
print("nnunetv2:", importlib.metadata.version("nnunetv2"))
print("nibabel:", nibabel.__version__)
PY

python "${G2_MATERIALIZER}" \
    --results-root "${G2_RESULTS}" \
    --real-mapping "${REAL_MAPPING}" \
    --real-data-root "${REAL_DATA_ROOT}" \
    --master-split "${MASTER_SPLIT}" \
    --synthetic-accepted-manifest "${ACCEPTED_MANIFEST}" \
    --completion-root "${COMPLETION_ROOT}" \
    --fake-t2w-cases "${FAKE_T2W_CASES}" \
    --output-root "${nnUNet_raw}" \
    --case-folder-root "${S2_STORAGE_ROOT}/completion_case_folders" \
    --dataset-id 264 \
    --dataset-name BraTS2026_MET_Completion \
    --dataset-profile completion \
    --mode symlink \
    --allow-incomplete-completion \
    --clean-output

python "${S2_REPO}/scripts/08_prepare_completion_online_split.py" \
    --dataset-dir "${DATASET_DIR}" \
    --output-split-dir "${BRATS_SPLIT_DIR}" \
    --expected-train 1035 \
    --expected-val 103 \
    --expected-test 104 \
    --expected-completions 212

nnUNetv2_extract_fingerprint \
    -d 264 \
    --verify_dataset_integrity \
    -np 16

python "${S2_REPO}/scripts/09_clone_baseline_plans.py" \
    --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
    --source-dataset-json "${DATASET_DIR}/dataset.json" \
    --target-preprocessed-dir "${nnUNet_preprocessed}/${DATASET_NAME}" \
    --target-dataset-name "${DATASET_NAME}" \
    --expected-num-training 1138

s2_check_active_storage_free_space
nnUNetv2_preprocess \
    -d 264 \
    -plans_name nnUNetPlans \
    -c 3d_fullres \
    -np 16

python "${S2_REPO}/scripts/05_validate_fixed_split_cache.py" \
    --train-file "${BRATS_SPLIT_DIR}/train_fixed.txt" \
    --val-file "${BRATS_SPLIT_DIR}/val_fixed.txt" \
    --dataset-dir "${DATASET_DIR}" \
    --preprocessed-dir "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_3d_fullres" \
    --output-json "${BRATS_SPLIT_DIR}/fixed_split_cache_audit.json"

s2_check_active_storage_free_space
echo "S2_COMPLETION_PREP_PASS dataset=${DATASET_DIR} storage=${S2_STORAGE_ROOT} mode=${S2_STORAGE_MODE}"
