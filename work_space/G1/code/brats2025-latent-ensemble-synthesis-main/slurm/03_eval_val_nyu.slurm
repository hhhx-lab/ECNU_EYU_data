#!/bin/bash
#SBATCH --job-name=g1_eval_val
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_val_%j.out
#SBATCH --error=logs/eval_val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu
# ============================================================
# G1 T2W Completion - Step 3: Validation-only evaluation [NYU HPC]
#
# This job evaluates only the fixed G2 val split from data_csv.csv.
# Do not use all complete cases for model selection or tuning.
# Run 04_infer_missing_t2w_nyu.slurm only after reviewing this output.
# ============================================================

set -e
PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="$PROJ/work_space/G1/code/brats2025-latent-ensemble-synthesis-main"
DATA_DIR="$PROJ/work_space/G1/data"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-brats2025}
EVAL_SPLIT=${EVAL_SPLIT:-val}
ENCDEC_CKPT_DIR="$CODE_DIR/training/endec/check_points"
BBDM_CKPT_DIR="$CODE_DIR/training/bbdm/check_points"

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Date   : $(date)"
echo "Split  : ${EVAL_SPLIT}"
echo "===================="

source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -uo pipefail
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "$CODE_DIR"
echo "CWD    : $(pwd)"

if [[ ! -f "$DATA_DIR/data_csv.csv" ]]; then
    echo "Missing data_csv.csv. Run 01_prepare_data_nyu.slurm first: $DATA_DIR/data_csv.csv" >&2
    exit 1
fi
if ! find "$ENCDEC_CKPT_DIR" -maxdepth 1 -type f -name "*.pt" -print -quit | grep -q .; then
    echo "Missing EncDec .pt checkpoint under: $ENCDEC_CKPT_DIR" >&2
    exit 1
fi
if ! find "$BBDM_CKPT_DIR" -maxdepth 1 -type f -name "*.pt" -print -quit | grep -q .; then
    echo "Missing BBDM .pt checkpoint under: $BBDM_CKPT_DIR" >&2
    exit 1
fi

echo "===== Evaluate fixed ${EVAL_SPLIT} split ====="
echo "Start: $(date)"
python evaluate.py \
    --input_dir "$DATA_DIR/input" \
    --csv_path "$DATA_DIR/data_csv.csv" \
    --split "$EVAL_SPLIT" \
    --synthesis_type ensamble \
    --gpu_id 0 \
    --verbose \
    --save_csv "$DATA_DIR/eval_metrics_${EVAL_SPLIT}.csv" \
    --save_output \
    --output_dir "$DATA_DIR/eval_synthesized_${EVAL_SPLIT}"
echo "End:   $(date)"

echo "===== DONE: $(date) ====="
echo "  Metrics: $DATA_DIR/eval_metrics_${EVAL_SPLIT}.csv"
echo "  Synth:   $DATA_DIR/eval_synthesized_${EVAL_SPLIT}/"
echo "Review validation metrics and synthesized images before running 04_infer_missing_t2w_nyu.slurm."
