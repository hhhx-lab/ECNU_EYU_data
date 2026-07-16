#!/bin/bash
#SBATCH --job-name=g1_diffv3_eval
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_eval_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-g1_diffusion_v3}
RUN_ROOT=${RUN_ROOT:-"${PROJ}/runs/g1_diffusion_v3"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${RUN_ROOT}/checkpoints"}
CSV_PATH=${CSV_PATH:-"${RUN_ROOT}/splits/current/lesions.csv"}
LOGDIR=${LOGDIR:-brats2026_diffusion_v3_edm_zscore}
EVAL_MODE=${EVAL_MODE:-whole_brain}
EVAL_SPLIT=${EVAL_SPLIT:-val}
MAX_CASES=${MAX_CASES:-0}
SAMPLING_METHOD=${SAMPLING_METHOD:-edm_heun}
SAMPLING_STEPS=${SAMPLING_STEPS:-18}
# NYU default: compile on. Set USE_COMPILE=0 if Triton fails.
USE_COMPILE=${USE_COMPILE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"${RUN_ROOT}/eval/${LOGDIR}_${EVAL_MODE}"}

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-allocated}"
echo "Date   : $(date)"
echo "Logdir : ${LOGDIR}"
echo "Mode   : ${EVAL_MODE}"
echo "Split  : ${EVAL_SPLIT}"
echo "===================="

mkdir -p "${PROJ}/logs" "${OUTPUT_DIR}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
echo "Python : $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

CKPT_DIR="${CHECKPOINT_ROOT}/${LOGDIR}"
if [[ ! -f "${CSV_PATH}" ]]; then
  echo "ERROR: CSV not found: ${CSV_PATH}" >&2
  exit 1
fi
for modality in t1c t1n t2w t2f; do
  test -s "${CKPT_DIR}/${modality}/weights/diffusion_100000.pt"
done

COMPILE_ARGS=()
if [[ "${USE_COMPILE}" == "1" ]]; then
  COMPILE_ARGS=(--use_compile)
fi

python src/infer/evaluate_generation.py \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --csv_path "${CSV_PATH}" \
  --dataset BRATS_2024 \
  --output_dir "${OUTPUT_DIR}" \
  --generator_type Unet_NnU \
  --crop_size 64 \
  --normalization zscore \
  --sigma_data 1.0 \
  --evaluation_mode "${EVAL_MODE}" \
  --large_lesion_mode tile \
  --split "${EVAL_SPLIT}" \
  --noise_schedule edm \
  --sampling_method "${SAMPLING_METHOD}" \
  --sampling_steps "${SAMPLING_STEPS}" \
  --max_cases "${MAX_CASES}" \
  "${COMPILE_ARGS[@]}"

test -s "${OUTPUT_DIR}/metrics.json"
echo "EVALUATION_PASS metrics=${OUTPUT_DIR}/metrics.json"
echo "===== DONE: $(date) ====="
echo "Metrics/output: ${OUTPUT_DIR}"
