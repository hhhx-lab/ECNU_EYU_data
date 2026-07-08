#!/bin/bash
#SBATCH --job-name=g1_train
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu
# ============================================================
# G1 T2W Completion - Step 2: Training (EncDec + BBDM)  [NYU HPC]
# Run after 01_prepare_data_nyu.slurm:
#   sbatch --dependency=afterok:<PREP_JOB_ID> slurm/02_train_nyu.slurm
# ============================================================

set -e

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
CODE_DIR="$PROJ/work_space/G1/code/brats2025-latent-ensemble-synthesis-main"
DATA_DIR="$PROJ/work_space/G1/data"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2025}"

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Date   : $(date)"
echo "Project: ${PROJ}"
echo "===================="

source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -uo pipefail
echo "Python : $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available())"

cd "$CODE_DIR"
echo "CWD    : $(pwd)"

if [[ ! -f "$DATA_DIR/data_csv.csv" ]]; then
    echo "Missing data_csv.csv. Run 01_prepare_data_nyu.slurm first: $DATA_DIR/data_csv.csv" >&2
    exit 1
fi

python - "$DATA_DIR/data_csv.csv" <<'PY'
import csv
import sys
from collections import Counter

csv_path = sys.argv[1]
with open(csv_path, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
counts = Counter(str(row.get("split", "")).strip().lower() for row in rows)
print("data_csv:", csv_path)
print("split counts:", dict(counts))
if counts["train"] == 0:
    raise SystemExit("data_csv.csv has no split=train rows; stop before training.")
if counts["val"] == 0:
    raise SystemExit("data_csv.csv has no split=val rows; stop before training.")
PY

echo "===== Training EncDec ====="
echo "Start: $(date)"
python training_endec.py
echo "End:   $(date)"

echo "===== Training BBDM ====="
echo "Start: $(date)"
python training_bbdm.py
echo "End:   $(date)"

echo "===== DONE: $(date) ====="
echo "  EncDec ckpts: $CODE_DIR/training/endec/check_points/"
echo "  BBDM   ckpts: $CODE_DIR/training/bbdm/check_points/"
