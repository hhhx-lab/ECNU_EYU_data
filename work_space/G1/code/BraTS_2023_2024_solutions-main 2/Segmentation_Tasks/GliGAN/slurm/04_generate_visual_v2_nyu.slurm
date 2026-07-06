#!/bin/bash
#SBATCH --job-name=g1_diffv2_visual
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_visual_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_visual_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-brats}
LOGDIR=${LOGDIR:-brats2026_diffusion_v2_complete_only}
CASE_ID=${CASE_ID:-BraTS-MET-00001-000}
DATASET=${DATASET:-BRATS_2024}
CROP_SIZE=${CROP_SIZE:-64}
MERGE_DIST=${MERGE_DIST:-16}
GENERATOR_TYPE=${GENERATOR_TYPE:-Unet_NnU}
NOISE_SCHEDULE=${NOISE_SCHEDULE:-edm}
SAMPLING_METHOD=${SAMPLING_METHOD:-edm_heun}
SAMPLING_STEPS=${SAMPLING_STEPS:-18}
OUTPUT_DIR="./visual_output/${LOGDIR}_${CASE_ID}"

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU    : ${CUDA_VISIBLE_DEVICES:-allocated}"
echo "Date   : $(date)"
echo "Case   : ${CASE_ID}"
echo "===================="

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

LABEL_PATH="DataSet/${CASE_ID}/${CASE_ID}-seg.nii.gz"
CKPT_DIR="../../Checkpoint/${LOGDIR}"
if [ ! -f "${LABEL_PATH}" ]; then
  echo "ERROR: label not found: ${LABEL_PATH}" >&2
  exit 1
fi
if [ ! -d "${CKPT_DIR}" ]; then
  echo "ERROR: checkpoint dir not found: ${CKPT_DIR}" >&2
  exit 1
fi

python src/infer/generate_from_label.py \
  --label_path "${LABEL_PATH}" \
  --diffusion_ckpt_dir "${CKPT_DIR}" \
  --dataset "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  --generator_type "${GENERATOR_TYPE}" \
  --crop_size "${CROP_SIZE}" \
  --merge_dist "${MERGE_DIST}" \
  --noise_schedule "${NOISE_SCHEDULE}" \
  --sampling_method "${SAMPLING_METHOD}" \
  --sampling_steps "${SAMPLING_STEPS}" \
  --modality all \
  --use_compile

echo "===== DONE: $(date) ====="
echo "Visual output: ${CODE_DIR}/${OUTPUT_DIR}"
