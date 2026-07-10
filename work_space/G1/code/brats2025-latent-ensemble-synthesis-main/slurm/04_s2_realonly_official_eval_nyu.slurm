#!/bin/bash
#SBATCH --job-name=s2_ro_metrics
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/s2_realonly_official_%j.out
#SBATCH --error=logs/s2_realonly_official_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

# Evaluate the combined out-of-fold predictions with BraTS_evaluation.
# This job intentionally requests no GPU and must run after all five folds.

set -e

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
S2_REPO="${PROJ}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats_eval}"

export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJ}/work_space/S2/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJ}/work_space/S2/data/nnunet_results}"
BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S2_REPO}/data/splits}"
BRATS_NNUNET_TRAINVAL_MAPPING_CSV="${BRATS_NNUNET_TRAINVAL_MAPPING_CSV:-${BRATS_SPLIT_DIR}/nnunet_case_mapping_realonly_train_val.csv}"
S2_DATASET_NAME="${S2_DATASET_NAME:-Dataset260_BraTS2026_MET_RealOnly}"
S2_TRAINER="${S2_TRAINER:-nnUNetTrainerBraTS2026RC}"
S2_CONFIGURATION="${S2_CONFIGURATION:-3d_fullres}"
OFFICIAL_RUN_ID="${OFFICIAL_RUN_ID:-realonly_5fold_cv}"
VOL_THRESHOLD="${VOL_THRESHOLD:-27}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.2}"

RESULT_BASE="${nnUNet_results}/${S2_DATASET_NAME}/${S2_TRAINER}__nnUNetPlans__${S2_CONFIGURATION}"
REFERENCE_ALL="${nnUNet_preprocessed}/${S2_DATASET_NAME}/gt_segmentations"
OUTPUT_ROOT="${PROJ}/work_space/S2/results/${OFFICIAL_RUN_ID}"
REFERENCE_VIEW="${OUTPUT_ROOT}/reference"
PREDICTION_VIEW="${OUTPUT_ROOT}/prediction"
SUMMARY_JSON="${OUTPUT_ROOT}/panoptica_evaluation_summary.json"
METRICS_CSV="${OUTPUT_ROOT}/leaderboard_metrics.csv"

echo "===== S2 Official-Compatible Evaluation ====="
echo "Job ID       : ${SLURM_JOB_ID:-local}"
echo "Date         : $(date)"
echo "Result base  : ${RESULT_BASE}"
echo "Reference    : ${REFERENCE_ALL}"
echo "Output       : ${OUTPUT_ROOT}"
echo "Vol threshold: ${VOL_THRESHOLD}"
echo "Overlap thr. : ${OVERLAP_THRESHOLD}"
echo "============================================="

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_SH}" >&2
    exit 1
fi
if [[ ! -d "${REFERENCE_ALL}" ]]; then
    echo "Missing nnU-Net ground-truth directory: ${REFERENCE_ALL}" >&2
    exit 1
fi
if [[ ! -f "${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}" ]]; then
    echo "Missing nnU-Net/source ID mapping: ${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

for command_name in brats-evaluate brats-parse-metrics; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing command in ${CONDA_ENV}: ${command_name}" >&2
        echo "Install the BraTS-evaluation package in the dedicated evaluation environment." >&2
        exit 1
    fi
done

mkdir -p "${REFERENCE_VIEW}" "${PREDICTION_VIEW}"
find "${REFERENCE_VIEW}" -mindepth 1 -maxdepth 1 -type l -delete
find "${PREDICTION_VIEW}" -mindepth 1 -maxdepth 1 -type l -delete
rm -f "${SUMMARY_JSON}" "${METRICS_CSV}"

MAPPING_TSV="${OUTPUT_ROOT}/nnunet_to_source_id.tsv"
python - "${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}" "${MAPPING_TSV}" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
with source.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
required = {"nnunet_case_id", "source_case_id"}
if not rows or not required.issubset(rows[0]):
    raise SystemExit(f"Mapping lacks required columns {sorted(required)}: {source}")
nn_ids = [row["nnunet_case_id"].strip() for row in rows]
source_ids = [row["source_case_id"].strip() for row in rows]
if len(nn_ids) != len(set(nn_ids)) or len(source_ids) != len(set(source_ids)):
    raise SystemExit("Mapping contains duplicate nnU-Net or source case IDs")
if any("\t" in value or "\n" in value for value in nn_ids + source_ids):
    raise SystemExit("Mapping case IDs must not contain tabs or newlines")
target.write_text(
    "".join(f"{nn_id}\t{source_id}\n" for nn_id, source_id in zip(nn_ids, source_ids)),
    encoding="utf-8",
)
PY

declare -A SOURCE_ID_BY_NNUNET
while IFS=$'\t' read -r NNUNET_ID SOURCE_ID; do
    [[ -z "${NNUNET_ID}" ]] && continue
    SOURCE_ID_BY_NNUNET["${NNUNET_ID}"]="${SOURCE_ID}"
done < "${MAPPING_TSV}"

EXPECTED=0
for FOLD in 0 1 2 3 4; do
    VAL_FILE="${BRATS_SPLIT_DIR}/val_fold${FOLD}.txt"
    FOLD_DIR="${RESULT_BASE}/fold_${FOLD}"
    PRED_DIR="${FOLD_DIR}/validation"
    if [[ ! -f "${FOLD_DIR}/checkpoint_final.pth" ]]; then
        echo "Fold ${FOLD} is incomplete; missing ${FOLD_DIR}/checkpoint_final.pth" >&2
        exit 1
    fi
    if [[ ! -s "${VAL_FILE}" || ! -d "${PRED_DIR}" ]]; then
        echo "Fold ${FOLD} split or validation predictions are missing." >&2
        exit 1
    fi

    while IFS= read -r CASE_ID; do
        [[ -z "${CASE_ID}" ]] && continue
        REF_FILE="${REFERENCE_ALL}/${CASE_ID}.nii.gz"
        PRED_FILE="${PRED_DIR}/${CASE_ID}.nii.gz"
        SOURCE_ID="${SOURCE_ID_BY_NNUNET[${CASE_ID}]:-}"
        if [[ -z "${SOURCE_ID}" ]]; then
            echo "Missing source-case mapping for ${CASE_ID}" >&2
            exit 1
        fi
        if [[ ! -f "${REF_FILE}" ]]; then
            echo "Missing reference for fold ${FOLD}: ${REF_FILE}" >&2
            exit 1
        fi
        if [[ ! -f "${PRED_FILE}" ]]; then
            echo "Missing prediction for fold ${FOLD}: ${PRED_FILE}" >&2
            exit 1
        fi
        if [[ -e "${PREDICTION_VIEW}/${SOURCE_ID}.nii.gz" ]]; then
            echo "Duplicate out-of-fold prediction source ID: ${SOURCE_ID}" >&2
            exit 1
        fi
        # BraTS_evaluation matches the official BraTS-MET-xxxxx-xxx suffix,
        # so expose source IDs instead of nnU-Net's BraTSMET_000001 IDs.
        ln -s "${REF_FILE}" "${REFERENCE_VIEW}/${SOURCE_ID}.nii.gz"
        ln -s "${PRED_FILE}" "${PREDICTION_VIEW}/${SOURCE_ID}.nii.gz"
        EXPECTED=$((EXPECTED + 1))
    done < "${VAL_FILE}"
done

REF_COUNT=$(find "${REFERENCE_VIEW}" -maxdepth 1 -type l -name '*.nii.gz' | wc -l | tr -d ' ')
PRED_COUNT=$(find "${PREDICTION_VIEW}" -maxdepth 1 -type l -name '*.nii.gz' | wc -l | tr -d ' ')
if [[ "${REF_COUNT}" != "${EXPECTED}" || "${PRED_COUNT}" != "${EXPECTED}" ]]; then
    echo "Out-of-fold materialization mismatch: expected=${EXPECTED}, ref=${REF_COUNT}, pred=${PRED_COUNT}" >&2
    exit 1
fi

brats-evaluate \
    --config mets \
    --ref_path "${REFERENCE_VIEW}" \
    --pred_path "${PREDICTION_VIEW}" \
    --summary_json "${SUMMARY_JSON}"

python - "${SUMMARY_JSON}" "${EXPECTED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(f"Evaluation did not create summary JSON: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
metrics = data.get("metrics", [])
missings = data.get("missings", [])
if len(metrics) != expected or missings:
    raise SystemExit(
        f"BraTS evaluation coverage mismatch: expected={expected}, "
        f"metrics={len(metrics)}, missings={len(missings)}"
    )
PY

brats-parse-metrics mets \
    --json_path "${SUMMARY_JSON}" \
    --vol_threshold "${VOL_THRESHOLD}" \
    --overlap_threshold "${OVERLAP_THRESHOLD}" \
    --output_csv_path "${METRICS_CSV}"

if [[ ! -s "${METRICS_CSV}" ]]; then
    echo "Metrics parser did not create a non-empty CSV: ${METRICS_CSV}" >&2
    exit 1
fi

echo "===== EVALUATION DONE: $(date) ====="
echo "Cases  : ${EXPECTED}"
echo "JSON   : ${SUMMARY_JSON}"
echo "Metrics: ${METRICS_CSV}"
