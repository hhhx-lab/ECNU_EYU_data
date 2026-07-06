#!/bin/bash
#SBATCH --job-name=g1_diffv2_train4
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_train4_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_train4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-brats}
LOGDIR=${LOGDIR:-brats2026_diffusion_v2_complete_only}
DATASET=${DATASET:-BRATS_2024}
BATCH_SIZE=${BATCH_SIZE:-16}
CROP_SIZE=${CROP_SIZE:-64}
SMALL_LESION_WEIGHT=${SMALL_LESION_WEIGHT:-3.0}
NUM_STEPS=${NUM_STEPS:-100000}
GENERATOR_TYPE=${GENERATOR_TYPE:-Unet_NnU}
NOISE_SCHEDULE=${NOISE_SCHEDULE:-edm}

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU(s) : ${CUDA_VISIBLE_DEVICES:-all allocated}"
echo "Date   : $(date)"
echo "Code   : ${CODE_DIR}"
echo "Logdir : ${LOGDIR}"
echo "===================="

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
echo "Python : $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count())"

CSV_PATH="../../Checkpoint/${LOGDIR}/${LOGDIR}.csv"
if [ ! -f "${CSV_PATH}" ]; then
  echo "ERROR: CSV not found. Run slurm/01_create_csv_v2_nyu.slurm first: ${CSV_PATH}" >&2
  exit 1
fi

mkdir -p "slurm/logs"
MODALITIES=(t1c t1n t2w t2f)
PIDS=()

for IDX in 0 1 2 3; do
  MODALITY="${MODALITIES[$IDX]}"
  LOG_FILE="slurm/logs/train_${LOGDIR}_${MODALITY}_${SLURM_JOB_ID}.log"
  echo "Launching ${MODALITY} on GPU ${IDX}; log=${LOG_FILE}"
  CUDA_VISIBLE_DEVICES="${IDX}" python src/train/tumour_main_diffusion.py \
    --dataset "${DATASET}" \
    --modality "${MODALITY}" \
    --logdir "${LOGDIR}" \
    --split train \
    --batch_size "${BATCH_SIZE}" \
    --generator_type "${GENERATOR_TYPE}" \
    --crop_size "${CROP_SIZE}" \
    --small_lesion_weight "${SMALL_LESION_WEIGHT}" \
    --num_steps "${NUM_STEPS}" \
    --noise_schedule "${NOISE_SCHEDULE}" \
    --use_compile > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    STATUS=1
  fi
done

echo "===== Training status: ${STATUS} ====="
echo "Loss monitor examples:"
echo "python scripts/watch_loss.py ../../Checkpoint/${LOGDIR} t1c --live"
echo "python scripts/watch_loss.py ../../Checkpoint/${LOGDIR} t1n --live"
echo "python scripts/watch_loss.py ../../Checkpoint/${LOGDIR} t2w --live"
echo "python scripts/watch_loss.py ../../Checkpoint/${LOGDIR} t2f --live"
echo "===== DONE: $(date) ====="
exit "${STATUS}"
