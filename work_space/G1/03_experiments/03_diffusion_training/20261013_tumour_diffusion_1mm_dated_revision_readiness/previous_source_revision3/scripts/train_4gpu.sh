#!/usr/bin/env bash
# Train the four modality models in parallel, one GPU per modality.
# Run from TumourDiffusion_1mm/ after Step 1 (preprocess) and Step 2 (CSV).
#
#   bash scripts/train_4gpu.sh                      # defaults below
#   LOGDIR=brats_1mm NUM_STEPS=100000 bash scripts/train_4gpu.sh
#   GPUS="4 5 6 7" LOADER_WORKERS=4 bash scripts/train_4gpu.sh
#   EXTRA_ARGS="--patient_balance_mode sqrt" bash scripts/train_4gpu.sh
set -euo pipefail

LOGDIR=${LOGDIR:-brats_1mm}               # Checkpoint/<LOGDIR>/ must contain the CSV from Step 1
NUM_STEPS=${NUM_STEPS:-200000}            # README_DIFFUSION recommends 150k-200k for a first full run; pick the best checkpoint with Step 5
BATCH_SIZE=${BATCH_SIZE:-16}
LOADER_WORKERS=${LOADER_WORKERS:-8}       # per process; 4 processes x 8 = 32 CPU cores
CACHE_WORKERS=${CACHE_WORKERS:-8}         # threads used once to build the ROI cache
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
EXTRA_ARGS=${EXTRA_ARGS:-}
MODS=(t1c t1n t2w t2f)

mkdir -p logs
for i in 0 1 2 3; do
  mod=${MODS[$i]}
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} nohup python src/train/tumour_main_diffusion.py \
      --dataset BRATS_2024 --modality "$mod" --logdir "$LOGDIR" \
      --batch_size "$BATCH_SIZE" --generator_type Unet_NnU \
      --crop_size 64 --small_lesion_weight 3.0 \
      --num_steps "$NUM_STEPS" --noise_schedule edm --use_compile \
      --normalization nnunet --sigma_data 1.0 --planner_gpu_gb 48 \
      --num_workers "$CACHE_WORKERS" --loader_workers "$LOADER_WORKERS" $EXTRA_ARGS \
      > "logs/train_${mod}.log" 2>&1 &
  echo "started ${mod} on GPU ${GPU_LIST[$i]} (pid $!) -> logs/train_${mod}.log"
done
echo "All four started. Follow progress with: tail -f logs/train_t1c.log"
wait
echo "All four finished."
