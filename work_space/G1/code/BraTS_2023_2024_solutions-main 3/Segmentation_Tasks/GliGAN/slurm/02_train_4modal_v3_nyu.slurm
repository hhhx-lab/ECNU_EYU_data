#!/bin/bash
#SBATCH --job-name=g1_diffv3_train4
#SBATCH --account=torch_pr_522_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_train4_%j.out
#SBATCH --error=/scratch/bf2260/ECNU_EYU_data/logs/g1_diffv3_train4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bf2260@nyu.edu

set -e

PROJ=${PROJ:-/scratch/bf2260/ECNU_EYU_data}
CODE_DIR="${PROJ}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
CONDA_SH=${CONDA_SH:-/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-g1_diffusion_v3}
RUN_ROOT=${RUN_ROOT:-"${PROJ}/runs/g1_diffusion_v3"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${RUN_ROOT}/checkpoints"}
SPLIT_DIR=${SPLIT_DIR:-"${RUN_ROOT}/splits/current"}
CSV_PATH=${CSV_PATH:-"${SPLIT_DIR}/lesions.csv"}
LOGDIR=${LOGDIR:-brats2026_diffusion_v3_edm_zscore}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_STEPS=${NUM_STEPS:-100000}
LOADER_WORKERS=${LOADER_WORKERS:-0}
# NYU Greene normally supports torch.compile. Set USE_COMPILE=0 to disable.
USE_COMPILE=${USE_COMPILE:-1}
AUTO_RESUME=${AUTO_RESUME:-1}

echo "===== Job Info ====="
echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : ${SLURMD_NODENAME:-unknown}"
echo "GPU(s) : ${CUDA_VISIBLE_DEVICES:-all allocated}"
echo "Date   : $(date)"
echo "Code   : ${CODE_DIR}"
echo "Run    : ${RUN_ROOT}"
echo "Logdir : ${LOGDIR}"
echo "Batch  : ${BATCH_SIZE}"
echo "Steps  : ${NUM_STEPS}"
echo "Compile: ${USE_COMPILE}"
echo "===================="

mkdir -p "${PROJ}/logs" "${CHECKPOINT_ROOT}" "${CODE_DIR}/slurm/logs"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -uo pipefail

cd "${CODE_DIR}"
echo "CWD    : $(pwd)"
echo "Python : $(which python)"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| n_gpu', torch.cuda.device_count())"

test -s "${RUN_ROOT}/PREPARED.ok"
test -s "${CSV_PATH}"
N_GPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "${N_GPU}" -lt 4 ]]; then
  echo "ERROR: need 4 GPUs, found ${N_GPU}" >&2
  exit 1
fi

COMPILE_ARGS=()
if [[ "${USE_COMPILE}" == "1" ]]; then
  COMPILE_ARGS=(--use_compile)
fi

MODALITIES=(t1c t1n t2w t2f)
PIDS=()
STATUS=0

for IDX in 0 1 2 3; do
  MODALITY="${MODALITIES[$IDX]}"
  WEIGHTS_DIR="${CHECKPOINT_ROOT}/${LOGDIR}/${MODALITY}/weights"
  LOG_FILE="slurm/logs/train_${LOGDIR}_${MODALITY}_${SLURM_JOB_ID}.log"
  RESUME_ARGS=()

  if [[ -d "${WEIGHTS_DIR}" ]]; then
    LATEST_STEP=$(find "${WEIGHTS_DIR}" -maxdepth 1 -type f -name 'diffusion_*.pt' \
      | sed -nE 's|.*/diffusion_([0-9]+)\.pt$|\1|p' | sort -n | tail -1)
    if [[ -n "${LATEST_STEP:-}" ]]; then
      if (( LATEST_STEP >= NUM_STEPS )); then
        echo "TRAIN_ALREADY_COMPLETE modality=${MODALITY} step=${LATEST_STEP}"
        continue
      fi
      if [[ "${AUTO_RESUME}" != "1" ]]; then
        echo "Existing checkpoint found but AUTO_RESUME=${AUTO_RESUME}: ${LATEST_STEP}" >&2
        exit 3
      fi
      RESUME_ARGS=(--resume_iter "${LATEST_STEP}")
      echo "auto_resume modality=${MODALITY} step=${LATEST_STEP}"
    fi
  fi

  echo "Launching ${MODALITY} on GPU ${IDX}; log=${LOG_FILE}"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="${IDX}" python src/train/tumour_main_diffusion.py \
      --dataset BRATS_2024 \
      --modality "${MODALITY}" \
      --logdir "${LOGDIR}" \
      --checkpoint_root "${CHECKPOINT_ROOT}" \
      --csv_path "${CSV_PATH}" \
      --split train \
      --batch_size "${BATCH_SIZE}" \
      --loader_workers "${LOADER_WORKERS}" \
      --generator_type Unet_NnU \
      --network_channels 48,96,192,384 \
      --network_strides 2,2,2 \
      --crop_size 64 \
      --small_lesion_weight 3.0 \
      --small_lesion_threshold 27.0 \
      --small_lesion_clamp 1.0 \
      --patient_balance_mode sqrt \
      --num_steps "${NUM_STEPS}" \
      --noise_schedule edm \
      --normalization zscore \
      --sigma_data 1.0 \
      "${COMPILE_ARGS[@]}" \
      "${RESUME_ARGS[@]}"
    test -s "${WEIGHTS_DIR}/diffusion_${NUM_STEPS}.pt"
    echo "TRAIN_MODALITY_PASS modality=${MODALITY} checkpoint=${WEIGHTS_DIR}/diffusion_${NUM_STEPS}.pt"
  ) > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

if ((${#PIDS[@]} > 0)); then
  for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
      STATUS=1
    fi
  done
fi

echo "===== Training status: ${STATUS} ====="
echo "Per-modality logs under: ${CODE_DIR}/slurm/logs/"
echo "Loss monitor examples:"
echo "python scripts/watch_loss.py ${CHECKPOINT_ROOT}/${LOGDIR} t1c --live"
echo "python scripts/watch_loss.py ${CHECKPOINT_ROOT}/${LOGDIR} t1n --live"
echo "python scripts/watch_loss.py ${CHECKPOINT_ROOT}/${LOGDIR} t2w --live"
echo "python scripts/watch_loss.py ${CHECKPOINT_ROOT}/${LOGDIR} t2f --live"
echo "===== DONE: $(date) ====="
exit "${STATUS}"
