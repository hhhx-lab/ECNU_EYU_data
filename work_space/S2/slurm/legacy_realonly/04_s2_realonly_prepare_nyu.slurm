#!/bin/bash
#SBATCH --job-name=s2_ro_prep
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/s2_realonly_prepare_%j.out
#SBATCH --error=logs/s2_realonly_prepare_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

# Prepare the current S2 dataset or recover the historical Dataset260 baseline.
# This job intentionally requests no GPU.

set -e

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
export PROJ
S2_REPO="${PROJ}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
G2_RESULTS="${PROJ}/work_space/G2/results"
G2_RAW_INTAKE_SCRIPT="${PROJ}/work_space/G2/code/g2_build_realonly_from_raw.py"
RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJ}/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training}"
LOG_DIR="${PROJ}/logs"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2026_s2}"
S2_EXPERIMENT_MODE="${S2_EXPERIMENT_MODE:-current}"
G2_REFRESH_REALONLY="${G2_REFRESH_REALONLY:-0}"
S2_RUN_PREPROCESS="${S2_RUN_PREPROCESS:-auto}"
S2_PRESERVE_FIXED_EXCLUSIONS="${S2_PRESERVE_FIXED_EXCLUSIONS:-1}"

case "${S2_EXPERIMENT_MODE}" in
    current)
        DEFAULT_S2_REUSE=0
        DEFAULT_S2_DATASET_ID=263
        DEFAULT_S2_DATASET_NAME=Dataset263_BraTS2026_MET_RealOnly_Current
        DEFAULT_S2_MAPPING_CSV="${G2_RESULTS}/manifests/nnunet_case_mapping_realonly.csv"
        DEFAULT_S2_SPLIT_JSON="${G2_RESULTS}/splits/splits_final_train_val_test.json"
        DEFAULT_S2_TRAIN_COUNT=823
        DEFAULT_S2_VAL_COUNT=103
        DEFAULT_S2_TEST_COUNT=104
        ;;
    legacy)
        DEFAULT_S2_REUSE=1
        DEFAULT_S2_DATASET_ID=260
        DEFAULT_S2_DATASET_NAME=Dataset260_BraTS2026_MET_RealOnly
        DEFAULT_S2_MAPPING_CSV="${G2_RESULTS}/manifests/nnunet_case_mapping_master.csv"
        DEFAULT_S2_SPLIT_JSON="${G2_RESULTS}/splits/splits_final_train_val_test.json"
        DEFAULT_S2_TRAIN_COUNT=828
        DEFAULT_S2_VAL_COUNT=207
        DEFAULT_S2_TEST_COUNT=259
        ;;
    *)
        echo "S2_EXPERIMENT_MODE must be current or legacy, got: ${S2_EXPERIMENT_MODE}" >&2
        exit 2
        ;;
esac

S2_REUSE_EXISTING_FIXED_SPLIT="${S2_REUSE_EXISTING_FIXED_SPLIT:-${DEFAULT_S2_REUSE}}"

export nnUNet_raw="${nnUNet_raw:-${PROJ}/work_space/S2/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJ}/work_space/S2/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJ}/work_space/S2/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S2_REPO}/data/splits/${S2_EXPERIMENT_MODE}}"
export BRATS_NNUNET_MAPPING_CSV="${BRATS_NNUNET_MAPPING_CSV:-${DEFAULT_S2_MAPPING_CSV}}"
export BRATS_NNUNET_TRAINVAL_MAPPING_CSV="${BRATS_NNUNET_TRAINVAL_MAPPING_CSV:-${BRATS_SPLIT_DIR}/nnunet_case_mapping_realonly_train_val.csv}"
export BRATS_SPLIT_JSON="${BRATS_SPLIT_JSON:-${DEFAULT_S2_SPLIT_JSON}}"
export S2_DATASET_ID="${S2_DATASET_ID:-${DEFAULT_S2_DATASET_ID}}"
export S2_DATASET_NAME="${S2_DATASET_NAME:-${DEFAULT_S2_DATASET_NAME}}"
export S2_EXPECTED_TRAIN_COUNT="${S2_EXPECTED_TRAIN_COUNT:-${DEFAULT_S2_TRAIN_COUNT}}"
export S2_EXPECTED_VAL_COUNT="${S2_EXPECTED_VAL_COUNT:-${DEFAULT_S2_VAL_COUNT}}"
export S2_EXPECTED_TEST_COUNT="${S2_EXPECTED_TEST_COUNT:-${DEFAULT_S2_TEST_COUNT}}"
export NNUNET_DATASET_DIR="${NNUNET_DATASET_DIR:-${nnUNet_raw}/${S2_DATASET_NAME}}"
S2_TRAINER="${S2_TRAINER:-nnUNetTrainerBraTS2026RC}"
S2_CONFIGURATION="${S2_CONFIGURATION:-3d_fullres}"
S2_PREPROCESSED_DATA_IDENTIFIER="${S2_PREPROCESSED_DATA_IDENTIFIER:-nnUNetPlans_3d_fullres}"
S2_RESULT_BASE="${nnUNet_results}/${S2_DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__${S2_CONFIGURATION}"
S2_EXISTING_VALIDATION_DIR="${S2_EXISTING_VALIDATION_DIR:-${S2_RESULT_BASE}/fold_0/validation}"

echo "===== S2 Preparation Job ====="
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Date         : $(date)"
echo "Project      : ${PROJ}"
echo "Mode         : ${S2_EXPERIMENT_MODE}"
echo "Raw data     : ${RAW_DATA_DIR}"
echo "Mapping CSV  : ${BRATS_NNUNET_MAPPING_CSV}"
echo "Split JSON   : ${BRATS_SPLIT_JSON}"
echo "Fixed split  : ${BRATS_SPLIT_DIR}"
echo "Dataset      : ${NNUNET_DATASET_DIR}"
echo "Validation   : ${S2_EXISTING_VALIDATION_DIR}"
echo "Preprocess   : ${S2_RUN_PREPROCESS}"
echo "Expected     : ${S2_EXPECTED_TRAIN_COUNT}/${S2_EXPECTED_VAL_COUNT}/${S2_EXPECTED_TEST_COUNT}"
echo "=============================="

if [[ "${S2_RUN_PREPROCESS}" != "auto" && "${S2_RUN_PREPROCESS}" != "0" && "${S2_RUN_PREPROCESS}" != "1" ]]; then
    echo "S2_RUN_PREPROCESS must be auto, 0, or 1, got: ${S2_RUN_PREPROCESS}" >&2
    exit 2
fi
if [[ "${G2_REFRESH_REALONLY}" != "0" && "${G2_REFRESH_REALONLY}" != "1" ]]; then
    echo "G2_REFRESH_REALONLY must be 0 or 1, got: ${G2_REFRESH_REALONLY}" >&2
    exit 2
fi
if [[ "${S2_PRESERVE_FIXED_EXCLUSIONS}" != "0" && "${S2_PRESERVE_FIXED_EXCLUSIONS}" != "1" ]]; then
    echo "S2_PRESERVE_FIXED_EXCLUSIONS must be 0 or 1, got: ${S2_PRESERVE_FIXED_EXCLUSIONS}" >&2
    exit 2
fi
if [[ "${S2_REUSE_EXISTING_FIXED_SPLIT}" != "0" && "${S2_REUSE_EXISTING_FIXED_SPLIT}" != "1" ]]; then
    echo "S2_REUSE_EXISTING_FIXED_SPLIT must be 0 or 1, got: ${S2_REUSE_EXISTING_FIXED_SPLIT}" >&2
    exit 2
fi
if [[ "${S2_DATASET_NAME}" != Dataset${S2_DATASET_ID}_* ]]; then
    echo "S2_DATASET_NAME must start with Dataset${S2_DATASET_ID}_, got: ${S2_DATASET_NAME}" >&2
    exit 2
fi
if [[ "${S2_DATASET_ID}" != "${DEFAULT_S2_DATASET_ID}" || "${S2_DATASET_NAME}" != "${DEFAULT_S2_DATASET_NAME}" ]]; then
    echo "${S2_EXPERIMENT_MODE} mode is locked to dataset ${DEFAULT_S2_DATASET_ID}/${DEFAULT_S2_DATASET_NAME}." >&2
    echo "Got ${S2_DATASET_ID}/${S2_DATASET_NAME}; refusing to cross-contaminate current and legacy artifacts." >&2
    exit 2
fi
if [[ "${S2_EXPERIMENT_MODE}" == "current" && "${S2_REUSE_EXISTING_FIXED_SPLIT}" != "0" ]]; then
    echo "Current S2 mode must use S2_REUSE_EXISTING_FIXED_SPLIT=0." >&2
    exit 2
fi
if [[ "${S2_EXPERIMENT_MODE}" == "legacy" && "${S2_REUSE_EXISTING_FIXED_SPLIT}" != "1" ]]; then
    echo "Legacy S2 mode must use S2_REUSE_EXISTING_FIXED_SPLIT=1." >&2
    exit 2
fi
if [[ "${S2_EXPERIMENT_MODE}" == "legacy" && "${S2_PRESERVE_FIXED_EXCLUSIONS}" != "1" ]]; then
    echo "Legacy S2 mode requires S2_PRESERVE_FIXED_EXCLUSIONS=1." >&2
    exit 2
fi
if [[ "${G2_REFRESH_REALONLY}" == "1" && "${S2_REUSE_EXISTING_FIXED_SPLIT}" == "1" ]]; then
    echo "G2_REFRESH_REALONLY=1 cannot be combined with S2_REUSE_EXISTING_FIXED_SPLIT=1." >&2
    echo "Use the existing artifacts for the completed baseline, or create a separately named new experiment." >&2
    exit 2
fi
if [[ ! -d "${S2_REPO}" ]]; then
    echo "Missing S2 repository: ${S2_REPO}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_SH}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${BRATS_SPLIT_DIR}" "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${S2_REPO}"

if [[ "${G2_REFRESH_REALONLY}" == "1" ]]; then
    echo "===== Refresh G2 real-only mapping/split ====="
    echo "WARNING: rebuilding the mapping can change nnU-Net case identities."
    echo "Do not refresh when reusing the completed fixed-split experiment."
    if [[ ! -d "${RAW_DATA_DIR}" ]]; then
        echo "Missing raw data directory: ${RAW_DATA_DIR}" >&2
        exit 1
    fi
    if [[ ! -f "${G2_RAW_INTAKE_SCRIPT}" ]]; then
        echo "Missing G2 raw intake script: ${G2_RAW_INTAKE_SCRIPT}" >&2
        exit 1
    fi
    G2_ARGS=(
        --project-root "${PROJ}"
        --data-root "${RAW_DATA_DIR}"
        --results-root "${G2_RESULTS}"
        --fail-if-no-valid-cases
    )
    python "${G2_RAW_INTAKE_SCRIPT}" "${G2_ARGS[@]}"
fi

if [[ ! -f "${BRATS_NNUNET_MAPPING_CSV}" || ! -f "${BRATS_SPLIT_JSON}" ]]; then
    echo "The ${S2_EXPERIMENT_MODE} mapping or split JSON is missing:" >&2
    echo "  ${BRATS_NNUNET_MAPPING_CSV}" >&2
    echo "  ${BRATS_SPLIT_JSON}" >&2
    echo "Restore the required G2 artifacts or refresh them only in current mode." >&2
    exit 1
fi

echo "===== Build the single fixed split ====="
FIXED_SPLIT_ARGS=(
    --split-json "${BRATS_SPLIT_JSON}"
    --mapping-csv "${BRATS_NNUNET_MAPPING_CSV}"
    --output-dir "${BRATS_SPLIT_DIR}"
    --trainval-mapping-csv "${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}"
    --expected-train-count "${S2_EXPECTED_TRAIN_COUNT}"
    --expected-val-count "${S2_EXPECTED_VAL_COUNT}"
    --expected-test-count "${S2_EXPECTED_TEST_COUNT}"
)
if [[ "${S2_EXPERIMENT_MODE}" == "current" ]]; then
    FIXED_SPLIT_ARGS+=(--require-patient-group-disjoint)
fi
if [[ "${S2_REUSE_EXISTING_FIXED_SPLIT}" == "1" ]]; then
    # Recover the authoritative old ID space from Dataset260 and the actual
    # fold_0 validation outputs. Do not trust files overwritten by the failed
    # five-fold preparation job.
    FIXED_SPLIT_ARGS+=(--reuse-existing)
    FIXED_SPLIT_ARGS+=(--existing-dataset-dir "${NNUNET_DATASET_DIR}")
    FIXED_SPLIT_ARGS+=(--existing-validation-dir "${S2_EXISTING_VALIDATION_DIR}")
    FIXED_SPLIT_ARGS+=(--baseline-excluded-source-id BraTS-MET-01094-002)
    FIXED_SPLIT_ARGS+=(--baseline-excluded-source-id BraTS-MET-01184-002)
fi
python scripts/04_build_fixed_split.py "${FIXED_SPLIT_ARGS[@]}"

PREPROCESSED_DATASET="${nnUNet_preprocessed}/${S2_DATASET_NAME}"
PREPROCESS_EXISTS=0
if [[ -f "${PREPROCESSED_DATASET}/nnUNetPlans.json" && -d "${PREPROCESSED_DATASET}/${S2_PREPROCESSED_DATA_IDENTIFIER}" ]]; then
    PREPROCESS_EXISTS=1
fi

if [[ "${S2_RUN_PREPROCESS}" == "1" || ("${S2_RUN_PREPROCESS}" == "auto" && "${PREPROCESS_EXISTS}" == "0") ]]; then
    echo "===== Materialize ${S2_DATASET_NAME} train+validation pool ====="
    python scripts/01_convert_to_nnunet.py \
        --mapping-csv "${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}" \
        --dst "${NNUNET_DATASET_DIR}" \
        --mode symlink \
        --clean

    echo "===== nnU-Net plan and preprocess ====="
    nnUNetv2_plan_and_preprocess \
        -d "${S2_DATASET_ID}" \
        -c "${S2_CONFIGURATION}" \
        --verify_dataset_integrity \
        --no_pbar
elif [[ "${PREPROCESS_EXISTS}" == "1" ]]; then
    echo "Reusing existing preprocessing for ${S2_DATASET_NAME}."
    echo "The raw dataset is not rematerialized, preserving case identities."
else
    echo "S2_RUN_PREPROCESS=0 but the preprocessed dataset is incomplete: ${PREPROCESSED_DATASET}" >&2
    exit 1
fi

echo "===== Audit fixed split against raw dataset and preprocessed cache ====="
python scripts/05_validate_fixed_split_cache.py \
    --train-file "${BRATS_SPLIT_DIR}/train_fixed.txt" \
    --val-file "${BRATS_SPLIT_DIR}/val_fixed.txt" \
    --dataset-dir "${NNUNET_DATASET_DIR}" \
    --preprocessed-dir "${PREPROCESSED_DATASET}/${S2_PREPROCESSED_DATA_IDENTIFIER}" \
    --output-json "${BRATS_SPLIT_DIR}/fixed_split_cache_audit.json"

echo "===== PREPARATION DONE: $(date) ====="
echo "Fixed split summary: ${BRATS_SPLIT_DIR}/fixed_split_summary.json"
echo "Cache audit       : ${BRATS_SPLIT_DIR}/fixed_split_cache_audit.json"
echo "Next: submit 04_s2_realonly_nyu.slurm once, without --array."
