#!/bin/bash
#SBATCH --job-name=g1_train
#SBATCH --partition=gpu
#SBATCH --account=jhuchar
#SBATCH --qos=defqos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@jh.edu

# ============================================================
# G1 T2W Completion — Step 2: Training (EncDec + BBDM)
# Reference: G1_G2_服务器训练推理QC运行手册 Section 8
# ============================================================
#
# Dependency: Run after 01_prepare_data.slurm completes.
#   sbatch --dependency=afterok:<PREP_JOB_ID> slurm/02_train.slurm
#
# Tuning order: 1) BBDM s  2) weight_decay  3) other params
# ============================================================

set -e

echo "===== Job Info ====="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    ${SLURMD_NODENAME}"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
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

# ---------- 3. Create logs dir ----------
mkdir -p logs

# ---------- 4. Train EncDec ----------
echo ""
echo "===== Training EncDec ====="
echo "Start: $(date)"
python training_endec.py
echo "End:   $(date)"

# ---------- 5. Train BBDM ----------
echo ""
echo "===== Training BBDM ====="
echo "Start: $(date)"
python training_bbdm.py
echo "End:   $(date)"

# ---------- Done ----------
echo ""
echo "===== Training completed ====="
echo "Checkpoints:"
echo "  EncDec: work_space/G1/code/brats2025-latent-ensemble-synthesis-main/training/endec/check_points/"
echo "  BBDM:   work_space/G1/code/brats2025-latent-ensemble-synthesis-main/training/bbdm/check_points/"
echo "Date: $(date)"
