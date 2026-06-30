#!/bin/bash
#SBATCH --job-name=g1_prep_data
#SBATCH --partition=defq
#SBATCH --account=jhuchar
#SBATCH --qos=defqos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/prep_data_%j.out
#SBATCH --error=logs/prep_data_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@jh.edu

# ============================================================
# G1 T2W Completion — Step 1: Data Preparation
# Reference: G1_G2_服务器训练推理QC运行手册 Sections 4-7
# ============================================================

set -e

echo "===== Job Info ====="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "Date:    $(date)"
echo "===================="

# ---------- 1. Activate conda environment ----------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate brats2025
echo "Conda env: $(conda info --envs | grep '*')"

# ---------- 2. Navigate to code directory ----------
CODE_DIR="work_space/G1/code/brats2025-latent-ensemble-synthesis-main"
if [ ! -d "${CODE_DIR}" ]; then
    echo "ERROR: Code directory not found: ${CODE_DIR}"
    exit 1
fi
cd "${CODE_DIR}"
echo "Working dir: $(pwd)"

# ---------- 3. Verify VAE weights ----------
echo ""
echo "===== Verifying VAE weights ====="
python test_vae.py

# ---------- 4. Auto-place G1 data ----------
echo ""
echo "===== Step 1: Placing G1 data ====="
python prepare_g1_t2w_data.py \
    --data-root ../../data \
    --mode symlink \
    --clean \
    --overwrite

# ---------- 5. Preprocess ----------
echo ""
echo "===== Step 2: Preprocessing ====="
python preprocess.py

# ---------- 6. Write fixed train/val/test split ----------
echo ""
echo "===== Step 3: Marking splits ====="
python mark_val_split_from_g2.py

# ---------- 7. Generate auxiliary files ----------
echo ""
echo "===== Step 4: Generate attention masks ====="
python generate_attmask.py

echo ""
echo "===== Step 5: Compute channel weights ====="
python compute_weights.py

# ---------- Done ----------
echo ""
echo "===== Data preparation completed ====="
echo "Checkpoints:"
echo "  CSV:        work_space/G1/data/data_csv.csv"
echo "  Manifest:   work_space/G1/data/g1_data_placement_manifest.csv"
echo "  Attn masks: work_space/G1/data/attention_masks/"
echo "  Weights:    work_space/G1/data/channel_weights/"
echo "Date: $(date)"
