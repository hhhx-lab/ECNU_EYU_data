#!/bin/bash
#SBATCH --job-name=s2_realonly
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --output=logs/s2_realonly_%j.out
#SBATCH --error=logs/s2_realonly_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

# ============================================================
# S2 real-only baseline - nnU-Net v2 + RC-aware trainer
#
# This job uses the original real training data only. It does not use G1
# completion output or synthetic augmentation. Data are materialized from the
# G2 real-only mapping into Dataset260_BraTS2026_MET_RealOnly.
# ============================================================

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
G2_REFRESH_REALONLY="${G2_REFRESH_REALONLY:-1}"

export nnUNet_raw="${nnUNet_raw:-${PROJ}/work_space/S2/data/nnunet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJ}/work_space/S2/data/nnunet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJ}/work_space/S2/data/nnunet_results}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S2_REPO}/data/splits}"
export BRATS_NNUNET_MAPPING_CSV="${BRATS_NNUNET_MAPPING_CSV:-${G2_RESULTS}/manifests/nnunet_case_mapping_realonly.csv}"
export BRATS_NNUNET_TRAINVAL_MAPPING_CSV="${BRATS_NNUNET_TRAINVAL_MAPPING_CSV:-${BRATS_SPLIT_DIR}/nnunet_case_mapping_realonly_train_val.csv}"
export BRATS_SPLIT_JSON="${BRATS_SPLIT_JSON:-${G2_RESULTS}/splits/splits_final_train_val_test.json}"
export S2_DATASET_ID="${S2_DATASET_ID:-260}"
export NNUNET_DATASET_DIR="${NNUNET_DATASET_DIR:-${nnUNet_raw}/Dataset260_BraTS2026_MET_RealOnly}"

mkdir -p "${LOG_DIR}" "${BRATS_SPLIT_DIR}" "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "===== Job Info ====="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-unknown}"
echo "Date       : $(date)"
echo "Project    : ${PROJ}"
echo "S2 repo    : ${S2_REPO}"
echo "Dataset    : ${NNUNET_DATASET_DIR}"
echo "Raw data   : ${RAW_DATA_DIR}"
echo "Mapping CSV: ${BRATS_NNUNET_MAPPING_CSV}"
echo "Train/val mapping CSV: ${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}"
echo "Split JSON : ${BRATS_SPLIT_JSON}"
echo "===================="

if [[ ! -d "${S2_REPO}" ]]; then
    echo "Missing S2 repository: ${S2_REPO}" >&2
    exit 1
fi
if [[ ! -d "${RAW_DATA_DIR}" ]]; then
    echo "Missing raw data directory: ${RAW_DATA_DIR}" >&2
    exit 1
fi
if [[ ! -f "${G2_RAW_INTAKE_SCRIPT}" ]]; then
    echo "Missing G2 raw intake script: ${G2_RAW_INTAKE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda activation script: ${CONDA_SH}" >&2
    echo "Submit with --export=ALL,CONDA_SH=/path/to/conda.sh if your cluster uses a different path." >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

cd "${S2_REPO}"

if [[ "${G2_REFRESH_REALONLY}" == "1" || ! -f "${BRATS_NNUNET_MAPPING_CSV}" || ! -f "${BRATS_SPLIT_JSON}" ]]; then
    echo "===== Refresh G2 real-only mapping/split from raw data ====="
    python "${G2_RAW_INTAKE_SCRIPT}" \
        --project-root "${PROJ}" \
        --data-root "${RAW_DATA_DIR}" \
        --results-root "${G2_RESULTS}" \
        --fail-if-no-valid-cases
fi
if [[ ! -f "${BRATS_NNUNET_MAPPING_CSV}" ]]; then
    echo "Missing G2 real-only mapping CSV after raw intake: ${BRATS_NNUNET_MAPPING_CSV}" >&2
    exit 1
fi
if [[ ! -f "${BRATS_SPLIT_JSON}" ]]; then
    echo "Missing G2 fixed split JSON after raw intake: ${BRATS_SPLIT_JSON}" >&2
    exit 1
fi

echo "===== Build fixed G2 train/val split for S2 ====="
python - <<'PY'
import csv
import json
import os
from pathlib import Path

mapping_csv = Path(os.environ["BRATS_NNUNET_MAPPING_CSV"])
trainval_mapping_csv = Path(os.environ["BRATS_NNUNET_TRAINVAL_MAPPING_CSV"])
split_json = Path(os.environ["BRATS_SPLIT_JSON"])
split_dir = Path(os.environ["BRATS_SPLIT_DIR"])
with split_json.open() as f:
    data = json.load(f)
split = data[0] if isinstance(data, list) else data
train_ids = split["train"]
val_ids = split["val"]
test_ids = split.get("test", [])
split_dir.mkdir(parents=True, exist_ok=True)
(split_dir / "train_full.txt").write_text("\n".join(train_ids) + "\n")
(split_dir / "val_full.txt").write_text("\n".join(val_ids) + "\n")
(split_dir / "test_internal_locked.txt").write_text("\n".join(test_ids) + "\n")

with mapping_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))
fieldnames = rows[0].keys()
by_id = {row["nnunet_case_id"]: row for row in rows}
selected_ids = train_ids + val_ids
missing = [case_id for case_id in selected_ids if case_id not in by_id]
if missing:
    raise SystemExit(f"split ids missing from mapping: {missing[:10]} ... total={len(missing)}")
with trainval_mapping_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for case_id in selected_ids:
        writer.writerow(by_id[case_id])

print("train:", len(train_ids))
print("val:", len(val_ids))
print("test locked:", len(test_ids))
print("train/val mapping:", trainval_mapping_csv)
PY

echo "===== Materialize Dataset260 real-only train+val from G2 mapping ====="
python scripts/01_convert_to_nnunet.py \
    --mapping-csv "${BRATS_NNUNET_TRAINVAL_MAPPING_CSV}" \
    --dst "${NNUNET_DATASET_DIR}" \
    --mode symlink \
    --clean

echo "===== nnU-Net plan and preprocess ====="
nnUNetv2_plan_and_preprocess -d "${S2_DATASET_ID}" --verify_dataset_integrity

echo "===== Train S2 real-only ====="
bash train.sh

echo "===== DONE: $(date) ====="
echo "nnUNet_raw          : ${nnUNet_raw}"
echo "nnUNet_preprocessed : ${nnUNet_preprocessed}"
echo "nnUNet_results      : ${nnUNet_results}"
