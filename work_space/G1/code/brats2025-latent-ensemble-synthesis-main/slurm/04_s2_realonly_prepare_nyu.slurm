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

# Prepare the shared Dataset260 and deterministic five-fold split exactly once.
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
G2_REFRESH_REALONLY="${G2_REFRESH_REALONLY:-0}"
S2_RUN_PREPROCESS="${S2_RUN_PREPROCESS:-auto}"
S2_PRESERVE_FOLD0_EXCLUSIONS="${S2_PRESERVE_FOLD0_EXCLUSIONS:-1}"
S2_CV_SEED="${S2_CV_SEED:-BraTS2026-S2-realonly-five-fold-v1}"

export nnUNet_raw="${nnUNet_raw:-${PROJ}/work_space/S2/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJ}/work_space/S2/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJ}/work_space/S2/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S2_REPO}/data/splits}"
export BRATS_NNUNET_MAPPING_CSV="${BRATS_NNUNET_MAPPING_CSV:-${G2_RESULTS}/manifests/nnunet_case_mapping_realonly.csv}"
export BRATS_NNUNET_TRAINVAL_MAPPING_CSV="${BRATS_NNUNET_TRAINVAL_MAPPING_CSV:-${BRATS_SPLIT_DIR}/nnunet_case_mapping_realonly_train_val.csv}"
export BRATS_SPLIT_JSON="${BRATS_SPLIT_JSON:-${G2_RESULTS}/splits/splits_final_train_val_test.json}"
export S2_DATASET_ID="${S2_DATASET_ID:-260}"
export S2_DATASET_NAME="${S2_DATASET_NAME:-Dataset260_BraTS2026_MET_RealOnly}"
export NNUNET_DATASET_DIR="${NNUNET_DATASET_DIR:-${nnUNet_raw}/${S2_DATASET_NAME}}"

mkdir -p "${LOG_DIR}" "${BRATS_SPLIT_DIR}" "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "===== S2 Preparation Job ====="
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Date         : $(date)"
echo "Project      : ${PROJ}"
echo "Raw data     : ${RAW_DATA_DIR}"
echo "Mapping CSV  : ${BRATS_NNUNET_MAPPING_CSV}"
echo "Split JSON   : ${BRATS_SPLIT_JSON}"
echo "CV split dir : ${BRATS_SPLIT_DIR}"
echo "Dataset      : ${NNUNET_DATASET_DIR}"
echo "Preprocess   : ${S2_RUN_PREPROCESS}"
echo "=============================="

if [[ ! -d "${S2_REPO}" ]]; then
    echo "Missing S2 repository: ${S2_REPO}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_SH}" >&2
    exit 1
fi
if [[ "${S2_RUN_PREPROCESS}" != "auto" && "${S2_RUN_PREPROCESS}" != "0" && "${S2_RUN_PREPROCESS}" != "1" ]]; then
    echo "S2_RUN_PREPROCESS must be auto, 0, or 1, got: ${S2_RUN_PREPROCESS}" >&2
    exit 2
fi
if [[ "${G2_REFRESH_REALONLY}" != "0" && "${G2_REFRESH_REALONLY}" != "1" ]]; then
    echo "G2_REFRESH_REALONLY must be 0 or 1, got: ${G2_REFRESH_REALONLY}" >&2
    exit 2
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${S2_REPO}"

if [[ "${G2_REFRESH_REALONLY}" == "1" ]]; then
    echo "===== Refresh G2 real-only mapping/split ====="
    echo "WARNING: rebuilding the mapping can change nnU-Net case identities."
    echo "Do not refresh when continuing the completed fold 0 experiment."
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
    if [[ "${S2_PRESERVE_FOLD0_EXCLUSIONS}" == "1" ]]; then
        # The completed fold 0 excluded these two raw masks. Keep every CV fold
        # on the same 1035-case pool. A future corrected-label rerun must retrain
        # fold 0 as a new experiment instead of mixing data pools.
        G2_ARGS+=(--exclude-ids BraTS-MET-01094-002)
        G2_ARGS+=(--exclude-ids BraTS-MET-01184-002)
    fi
    python "${G2_RAW_INTAKE_SCRIPT}" "${G2_ARGS[@]}"
fi

if [[ ! -f "${BRATS_NNUNET_MAPPING_CSV}" || ! -f "${BRATS_SPLIT_JSON}" ]]; then
    echo "The fold-0 mapping or split JSON is missing:" >&2
    echo "  ${BRATS_NNUNET_MAPPING_CSV}" >&2
    echo "  ${BRATS_SPLIT_JSON}" >&2
    echo "Restore the artifacts used by fold 0. For a new experiment only, set G2_REFRESH_REALONLY=1." >&2
    exit 1
fi

echo "===== Build deterministic five-fold splits ====="
python scripts/04_build_five_fold_splits.py \
    --split-json "${BRATS_SPLIT_JSON}" \
    --mapping-csv "${BRATS_NNUNET_MAPPING_CSV}" \
    --output-dir "${BRATS_SPLIT_DIR}" \
    --trainval-mapping-csv "${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}" \
    --seed "${S2_CV_SEED}"

PREPROCESSED_DATASET="${nnUNet_preprocessed}/${S2_DATASET_NAME}"
PREPROCESS_EXISTS=0
if [[ -f "${PREPROCESSED_DATASET}/nnUNetPlans.json" && -d "${PREPROCESSED_DATASET}/nnUNetPlans_3d_fullres" ]]; then
    PREPROCESS_EXISTS=1
fi

if [[ "${S2_RUN_PREPROCESS}" == "1" || ("${S2_RUN_PREPROCESS}" == "auto" && "${PREPROCESS_EXISTS}" == "0") ]]; then
    echo "===== Materialize shared Dataset260 CV pool ====="
    python scripts/01_convert_to_nnunet.py \
        --mapping-csv "${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}" \
        --dst "${NNUNET_DATASET_DIR}" \
        --mode symlink \
        --clean

    echo "===== nnU-Net plan and preprocess ====="
    nnUNetv2_plan_and_preprocess -d "${S2_DATASET_ID}" --verify_dataset_integrity
elif [[ "${PREPROCESS_EXISTS}" == "1" ]]; then
    echo "Reusing the Dataset260 preprocessing created for completed fold 0."
    echo "The raw dataset is not rematerialized, preserving fold-0 case identities."
else
    echo "S2_RUN_PREPROCESS=0 but preprocessed Dataset260 is incomplete: ${PREPROCESSED_DATASET}" >&2
    exit 1
fi

echo "===== PREPARATION DONE: $(date) ====="
echo "Five-fold summary: ${BRATS_SPLIT_DIR}/five_fold_summary.json"
echo "Next: submit 04_s2_realonly_nyu.slurm as an array job."
