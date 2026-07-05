#!/bin/bash
#SBATCH --job-name=g1_eval_infer
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/eval_infer_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/eval_infer_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu
# ============================================================
# G1 T2W Completion - Step 3: Evaluation + Inference  [NYU HPC]
# ============================================================

set -e
PROJ=/scratch/bf2260/ECNU_EYU_data
CODE_DIR="$PROJ/work_space/G1/code/brats2025-latent-ensemble-synthesis-main"
DATA_DIR="$PROJ/work_space/G1/data"
CONDA_SH=/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Date   : $(date)"
echo "===================="

source "$CONDA_SH"
conda activate brats2025
set -uo pipefail
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "$CODE_DIR"
echo "CWD    : $(pwd)"

# --- Evaluate: scores synthesis vs ground-truth T2W. MUST use data/input (has all 4 modalities). ---
echo "===== Evaluate (on complete cases) ====="
echo "Start: $(date)"
python evaluate.py \
    --input_dir "$DATA_DIR/input" \
    --synthesis_type ensamble \
    --gpu_id 0 \
    --verbose \
    --save_csv "$DATA_DIR/eval_metrics.csv" \
    --save_output
echo "End:   $(date)"

# --- Inference: generates missing T2W for the challenge cases. ---
# --- NOTE: --compute_bmask dropped (needs TotalSegmentator). Add it back after installing if wanted. ---
echo "===== Inference (generate missing T2W) ====="
echo "Start: $(date)"
python main.py \
    --input_dir "$DATA_DIR/input_inference" \
    --output_dir "$DATA_DIR/output" \
    --synthesis_type ensamble \
    --gpu_id 0 \
    --verbose
echo "End:   $(date)"

echo "===== DONE: $(date) ====="
echo "  Metrics: $DATA_DIR/eval_metrics.csv"
echo "  Synth:   $DATA_DIR/eval_synthesized/"
echo "  Output:  $DATA_DIR/output/"
