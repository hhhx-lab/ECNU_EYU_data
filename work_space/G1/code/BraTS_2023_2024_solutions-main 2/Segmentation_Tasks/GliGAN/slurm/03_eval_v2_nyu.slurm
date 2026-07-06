#!/bin/bash
#SBATCH --job-name=g1_diffv2_eval
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_eval_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-brats}
LOGDIR=${LOGDIR:-brats2026_diffusion_v2_complete_only}
DATASET=${DATASET:-BRATS_2024}
CROP_SIZE=${CROP_SIZE:-64}
GENERATOR_TYPE=${GENERATOR_TYPE:-Unet_NnU}
NOISE_SCHEDULE=${NOISE_SCHEDULE:-edm}
SAMPLING_METHOD=${SAMPLING_METHOD:-edm_heun}
SAMPLING_STEPS=${SAMPLING_STEPS:-18}
EVAL_MODE=${EVAL_MODE:-whole_brain}
EVAL_SPLIT=${EVAL_SPLIT:-val}
MAX_CASES=${MAX_CASES:-0}
OUTPUT_DIR="./eval_results/${LOGDIR}_${EVAL_MODE}"

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-allocated}"
echo "Date   : $(date)"
echo "Logdir : ${LOGDIR}"
echo "Mode   : ${EVAL_MODE}"
echo "Split  : ${EVAL_SPLIT}"
echo "===================="

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

CSV_PATH="../../Checkpoint/${LOGDIR}/${LOGDIR}.csv"
CKPT_DIR="../../Checkpoint/${LOGDIR}"
if [ ! -f "${CSV_PATH}" ]; then
  echo "ERROR: CSV not found: ${CSV_PATH}" >&2
  exit 1
fi
if [ ! -d "${CKPT_DIR}" ]; then
  echo "ERROR: checkpoint dir not found: ${CKPT_DIR}" >&2
  exit 1
fi

python src/infer/evaluate_generation.py \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --csv_path "${CSV_PATH}" \
  --dataset "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --generator_type "${GENERATOR_TYPE}" \
  --crop_size "${CROP_SIZE}" \
  --evaluation_mode "${EVAL_MODE}" \
  --split "${EVAL_SPLIT}" \
  --noise_schedule "${NOISE_SCHEDULE}" \
  --sampling_method "${SAMPLING_METHOD}" \
  --sampling_steps "${SAMPLING_STEPS}" \
  --max_cases "${MAX_CASES}" \
  --use_compile

echo "===== DONE: $(date) ====="
echo "Metrics/output: ${CODE_DIR}/${OUTPUT_DIR}"
