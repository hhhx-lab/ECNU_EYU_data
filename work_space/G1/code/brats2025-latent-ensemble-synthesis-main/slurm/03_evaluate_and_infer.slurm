#!/bin/bash
#SBATCH --job-name=g1_eval_infer
#SBATCH --partition=gpu
#SBATCH --account=jhuchar
#SBATCH --qos=defqos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_infer_%j.out
#SBATCH --error=logs/eval_infer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@jh.edu

# ============================================================
# G1 T2W Completion — Step 3: Evaluation + Inference
# Reference: G1_G2_服务器训练推理QC运行手册 Sections 9-10
# ============================================================
#
# Dependency: Run after 02_train.slurm completes.
#   sbatch --dependency=afterok:<TRAIN_JOB_ID> slurm/03_evaluate_and_infer.slurm
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

# ---------- 4. Evaluate on Validation set ----------
echo ""
echo "===== Evaluating on Validation set ====="
echo "Start: $(date)"
python evaluate.py \
    --input_dir ../../data/raw/Validation \
    --synthesis_type ensamble \
    --gpu_id 0 \
    --verbose \
    --save_csv ../../data/eval_validation_metrics.csv \
    --save_output
echo "End: $(date)"

# ---------- 5. Inference on input_inference ----------
echo ""
echo "===== Running inference on input_inference ====="
echo "Start: $(date)"
python main.py \
    --input_dir ../../data/input_inference \
    --output_dir ../../data/output \
    --synthesis_type ensamble \
    --gpu_id 0 \
    --verbose \
    --compute_bmask
echo "End: $(date)"

# ---------- Done ----------
echo ""
echo "===== Evaluation + Inference completed ====="
echo "Outputs:"
echo "  Metrics:  work_space/G1/data/eval_validation_metrics.csv"
echo "  Synth:    work_space/G1/data/eval_synthesized/"
echo "  T2W:      work_space/G1/data/output/"
echo ""
echo "Next: hand over to G2 for QC (Section 11 of the manual)"
echo "Date: $(date)"
