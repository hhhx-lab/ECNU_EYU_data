#!/bin/bash
#SBATCH --job-name=s1_realonly
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=96:00:00
#SBATCH --output=logs/s1_realonly_%j.out
#SBATCH --error=logs/s1_realonly_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

# ============================================================
# S1 real-only baseline - MONAI multitask tumor + RC model
#
# This job uses original real training data only. It builds a lightweight
# S1 training view from the G2 real-only mapping, then generates S1 auxiliary
# tumor_label.nii.gz and rc_label.nii.gz inside that view.
# ============================================================

set -e

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
export PROJ
S1_REPO="${PROJ}/work_space/S1/brats2026_multitask_S1_v2/repository"
G2_RESULTS="${PROJ}/work_space/G2/results"
G2_RAW_INTAKE_SCRIPT="${PROJ}/work_space/G2/code/g2_build_realonly_from_raw.py"
RAW_DATA_DIR="${RAW_DATA_DIR:-${PROJ}/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training}"
LOG_DIR="${PROJ}/logs"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2026_s1}"
G2_REFRESH_REALONLY="${G2_REFRESH_REALONLY:-1}"

export S1_REALONLY_VIEW="${S1_REALONLY_VIEW:-${PROJ}/work_space/S1/data/real_only_cases}"
export BRATS_SPLIT_DIR="${BRATS_SPLIT_DIR:-${S1_REPO}/data/splits}"
export BRATS_NNUNET_MAPPING_CSV="${BRATS_NNUNET_MAPPING_CSV:-${G2_RESULTS}/manifests/nnunet_case_mapping_realonly.csv}"
export BRATS_SPLIT_JSON="${BRATS_SPLIT_JSON:-${G2_RESULTS}/splits/splits_final_train_val_test.json}"
export BRATS_TRAIN_ROOT="${S1_REALONLY_VIEW}"
export S1_CHECKPOINT_DIR="${S1_CHECKPOINT_DIR:-${PROJ}/work_space/S1/results/realonly/checkpoints}"
export S1_TENSORBOARD_DIR="${S1_TENSORBOARD_DIR:-${PROJ}/work_space/S1/results/realonly/tensorboard}"
export S1_OUTPUT_DIR="${S1_OUTPUT_DIR:-${PROJ}/work_space/S1/results/realonly/predictions}"

mkdir -p "${LOG_DIR}" "${BRATS_SPLIT_DIR}" "${S1_CHECKPOINT_DIR}" "${S1_TENSORBOARD_DIR}" "${S1_OUTPUT_DIR}"

echo "===== Job Info ====="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-unknown}"
echo "Date       : $(date)"
echo "Project    : ${PROJ}"
echo "S1 repo    : ${S1_REPO}"
echo "View root  : ${S1_REALONLY_VIEW}"
echo "Raw data   : ${RAW_DATA_DIR}"
echo "Mapping CSV: ${BRATS_NNUNET_MAPPING_CSV}"
echo "Split JSON : ${BRATS_SPLIT_JSON}"
echo "===================="

if [[ ! -d "${S1_REPO}" ]]; then
    echo "Missing S1 repository: ${S1_REPO}" >&2
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

cd "${S1_REPO}"

if [[ "${G2_REFRESH_REALONLY}" == "1" || ! -f "${BRATS_NNUNET_MAPPING_CSV}" || ! -f "${BRATS_SPLIT_JSON}" ]]; then
    echo "===== Refresh G2 real-only mapping/split from raw data ====="
    python "${G2_RAW_INTAKE_SCRIPT}" \
        --project-root "${PROJ}" \
        --data-root "${RAW_DATA_DIR}" \
        --results-root "${G2_RESULTS}" \
        --exclude-ids BraTS-MET-01094-002 \
        --exclude-ids BraTS-MET-01184-002 \
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

echo "===== Build S1 real-only training view and fixed split ====="
python - <<'PY'
import csv
import json
import os
import shutil
from pathlib import Path

proj = Path(os.environ["PROJ"])
view_root = Path(os.environ["S1_REALONLY_VIEW"])
mapping_csv = Path(os.environ["BRATS_NNUNET_MAPPING_CSV"])
split_json = Path(os.environ["BRATS_SPLIT_JSON"])
split_dir = Path(os.environ["BRATS_SPLIT_DIR"])

def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return proj / path

with mapping_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))
by_nnunet = {row["nnunet_case_id"]: row for row in rows}

with split_json.open() as f:
    data = json.load(f)
split = data[0] if isinstance(data, list) else data
train_ids = split["train"]
val_ids = split["val"]
test_ids = split.get("test", [])
selected_ids = train_ids + val_ids

missing_ids = [case_id for case_id in selected_ids if case_id not in by_nnunet]
if missing_ids:
    raise SystemExit(f"missing mapping ids: {missing_ids[:10]} ... total={len(missing_ids)}")

if view_root.exists():
    shutil.rmtree(view_root)
view_root.mkdir(parents=True, exist_ok=True)
split_dir.mkdir(parents=True, exist_ok=True)

modalities = {
    "t1n": "t1n_source_path",
    "t1c": "t1c_source_path",
    "t2w": "t2w_source_path",
    "t2f": "t2f_source_path",
}

def link_case(row):
    source_case = row["source_case_id"]
    case_dir = view_root / source_case
    case_dir.mkdir(parents=True, exist_ok=True)
    for mod, column in modalities.items():
        src = resolve(row[column])
        dst = case_dir / f"{source_case}-{mod}.nii.gz"
        if not src.exists():
            raise FileNotFoundError(src)
        dst.symlink_to(src)
    seg_src = resolve(row["seg_source_path"])
    seg_dst = case_dir / f"{source_case}-seg.nii.gz"
    if not seg_src.exists():
        raise FileNotFoundError(seg_src)
    seg_dst.symlink_to(seg_src)
    return source_case

train_cases = [link_case(by_nnunet[case_id]) for case_id in train_ids]
val_cases = [link_case(by_nnunet[case_id]) for case_id in val_ids]

(split_dir / "train_full.txt").write_text("\n".join(train_cases) + "\n")
(split_dir / "val_full.txt").write_text("\n".join(val_cases) + "\n")
(split_dir / "test_internal_locked.txt").write_text("\n".join(test_ids) + "\n")

with (view_root / "s1_realonly_view_manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["nnunet_case_id", "source_case_id", "split", "case_dir", "label_source"])
    writer.writeheader()
    for split_name, ids in (("train", train_ids), ("val", val_ids)):
        for nnunet_id in ids:
            row = by_nnunet[nnunet_id]
            writer.writerow({
                "nnunet_case_id": nnunet_id,
                "source_case_id": row["source_case_id"],
                "split": split_name,
                "case_dir": str(view_root / row["source_case_id"]),
                "label_source": row["label_source"],
            })

print("train:", len(train_cases))
print("val:", len(val_cases))
print("test locked:", len(test_ids))
print("view:", view_root)
PY

echo "===== Build S1 tumor/RC labels ====="
python scripts/08_build_full_multitask_labels.py --train-root "${BRATS_TRAIN_ROOT}"

echo "===== Audit S1 view ====="
python scripts/09_dataset_audit.py --train-root "${BRATS_TRAIN_ROOT}"

echo "===== Train S1 real-only ====="
python trainers/trainer_v1_final.py --config configs/multitask_v1_full.yaml

echo "===== DONE: $(date) ====="
echo "Checkpoints: ${S1_CHECKPOINT_DIR}"
echo "TensorBoard: ${S1_TENSORBOARD_DIR}"
