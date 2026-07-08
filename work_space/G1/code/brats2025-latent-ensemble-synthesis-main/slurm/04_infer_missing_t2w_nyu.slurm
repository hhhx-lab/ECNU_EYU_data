#!/bin/bash
#SBATCH --job-name=g1_infer_t2w
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/infer_t2w_%j.out
#SBATCH --error=logs/infer_t2w_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu
# ============================================================
# G1 T2W Completion - Step 4: Generate missing T2W [NYU HPC]
#
# Run this only after 03_eval_val_nyu.slurm has produced acceptable
# validation metrics and synthesized validation images.
# ============================================================

set -e
PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="$PROJ/work_space/G1/code/brats2025-latent-ensemble-synthesis-main"
DATA_DIR="$PROJ/work_space/G1/data"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-brats2025}
ENCDEC_CKPT_DIR="$CODE_DIR/training/endec/check_points"
BBDM_CKPT_DIR="$CODE_DIR/training/bbdm/check_points"

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "Date   : $(date)"
echo "===================="

source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -uo pipefail
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

cd "$CODE_DIR"
echo "CWD    : $(pwd)"

if [[ ! -d "$DATA_DIR/input_inference" ]]; then
    echo "Missing input_inference directory. Run 01_prepare_data_nyu.slurm first: $DATA_DIR/input_inference" >&2
    exit 1
fi
if ! find "$DATA_DIR/input_inference" -mindepth 1 -maxdepth 1 -type d -print -quit | grep -q .; then
    echo "No subject folders found under: $DATA_DIR/input_inference" >&2
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

echo "===== Inference: generate missing T2W ====="
echo "Start: $(date)"
python main.py \
    --input_dir "$DATA_DIR/input_inference" \
    --output_dir "$DATA_DIR/output" \
    --synthesis_type ensamble \
    --gpu_id 0 \
    --verbose
echo "End:   $(date)"

echo "===== DONE: $(date) ====="
echo "  Output: $DATA_DIR/output/"
echo "Next step: run G2 synthetic raw intake and QC on this output directory."
