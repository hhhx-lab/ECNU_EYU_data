#!/bin/bash
#SBATCH --job-name=g1_prep_data
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep_data_%j.out
#SBATCH --error=logs/prep_data_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu
# ============================================================
# G1 T2W Completion - Step 1: Data Preparation (NYU HPC)
# ============================================================

set -e   # -u added AFTER conda activate (conda shell funcs trip 'set -u')

PROJ="${PROJ:-/scratch/bf2260/ECNU_EYU_data}"
CODE_DIR="$PROJ/work_space/G1/code/brats2025-latent-ensemble-synthesis-main"
DATA_DIR="$PROJ/work_space/G1/data"
G2_REAL_MANIFEST="${G2_REAL_MANIFEST:-$PROJ/work_space/G2/results/manifests/real_train_manifest.csv}"
G2_FAKE_T2W_CASES="${G2_FAKE_T2W_CASES:-$PROJ/work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv}"
CONDA_SH="${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-brats2025}"

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Date   : $(date)"
echo "Project: ${PROJ}"
echo "Manifest: ${G2_REAL_MANIFEST}"
echo "===================="

source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -uo pipefail
echo "Python : $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available())"

cd "$CODE_DIR"
echo "CWD    : $(pwd)"

echo "===== Verify VAE ====="
python test_vae.py

if [[ ! -f "$G2_REAL_MANIFEST" ]]; then
    echo "Missing G2 real manifest: $G2_REAL_MANIFEST" >&2
    exit 1
fi
if [[ ! -f "$G2_FAKE_T2W_CASES" ]]; then
    echo "Missing fake T2W case list: $G2_FAKE_T2W_CASES" >&2
    exit 1
fi

echo "===== Place G1 data ====="
python prepare_g1_t2w_data.py \
    --real-manifest "$G2_REAL_MANIFEST" \
    --fake-t2w-cases "$G2_FAKE_T2W_CASES" \
    --data-root "$DATA_DIR" \
    --mode symlink \
    --clean \
    --overwrite

echo "===== Preprocess ====="
python preprocess.py --input-dir "$DATA_DIR/input"

echo "===== Mark splits ====="
python mark_val_split_from_g2.py

echo "===== Attention masks ====="
python generate_attmask.py

echo "===== Channel weights ====="
python compute_weights.py

echo "===== DONE: $(date) ====="
