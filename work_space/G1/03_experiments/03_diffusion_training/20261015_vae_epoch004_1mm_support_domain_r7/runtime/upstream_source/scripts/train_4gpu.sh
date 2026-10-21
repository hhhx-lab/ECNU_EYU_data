#!/usr/bin/env bash
# Train the four modality models (t1c t1n t2w t2f) on 1-4 GPUs.
# Run from TumourDiffusion_1mm/ after Step 1 (preprocess) and Step 2 (CSV).
#
# GPUS lists parallel slots, one GPU ID per slot. The four modalities are dealt round-robin
# across the slots; slots run in parallel and each slot trains its modalities one after another.
#
#   bash scripts/train_4gpu.sh                      # 4 GPUs, all four modalities in parallel
#   GPUS="0 1" bash scripts/train_4gpu.sh           # 2 GPUs: t1c->t2w on 0, t1n->t2f on 1
#   GPUS="0" bash scripts/train_4gpu.sh             # 1 GPU: t1c->t1n->t2w->t2f in series
#   GPUS="0 0" bash scripts/train_4gpu.sh           # 2 jobs at a time sharing GPU 0 (if VRAM allows)
#   LOGDIR=brats_1mm NUM_STEPS=100000 bash scripts/train_4gpu.sh
#   GPUS="4 5 6 7" LOADER_WORKERS=4 bash scripts/train_4gpu.sh
#   EXTRA_ARGS="--patient_balance_mode sqrt" bash scripts/train_4gpu.sh
#
# The architecture is pinned by --planner_gpu_gb 48, so every layout trains the same network.
# Each modality's exit code is also written to logs/train_<modality>.status.
set -euo pipefail

LOGDIR=${LOGDIR:-brats_1mm}               # Checkpoint/<LOGDIR>/ must contain the CSV from Step 1
NUM_STEPS=${NUM_STEPS:-200000}            # README_DIFFUSION recommends 150k-200k for a first full run; pick the best checkpoint with Step 5
BATCH_SIZE=${BATCH_SIZE:-16}
LOADER_WORKERS=${LOADER_WORKERS:-8}       # per running process; 4 slots x 8 = 32 CPU cores
CACHE_WORKERS=${CACHE_WORKERS:-8}         # threads used once to build the ROI cache
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
EXTRA_ARGS=${EXTRA_ARGS:-}
MODS=(t1c t1n t2w t2f)
N_SLOTS=${#GPU_LIST[@]}

if (( N_SLOTS < 1 || N_SLOTS > ${#MODS[@]} )); then
  echo "GPUS must list between 1 and ${#MODS[@]} GPU IDs (got ${N_SLOTS}); one slot per entry." >&2
  exit 2
fi

# Round-robin assignment: modality i runs in slot i % N_SLOTS.
SLOT_MODS=()
for i in "${!MODS[@]}"; do
  s=$(( i % N_SLOTS ))
  SLOT_MODS[s]="${SLOT_MODS[s]:-} ${MODS[$i]}"
done

run_slot() {
  local gpu=$1
  shift
  # The queue must outlive the launching terminal: ignore hangups (the training processes inherit
  # this, as with nohup) and keep going when progress messages can no longer be written. Each
  # outcome is recorded explicitly in its status file, so errexit is not needed here.
  trap '' HUP PIPE
  set +e
  local mod pid status
  for mod in "$@"; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu nohup python src/train/tumour_main_diffusion.py \
        --dataset BRATS_2024 --modality "$mod" --logdir "$LOGDIR" \
        --batch_size "$BATCH_SIZE" --generator_type Unet_NnU \
        --crop_size 64 --small_lesion_weight 3.0 \
        --num_steps "$NUM_STEPS" --noise_schedule edm --use_compile \
        --normalization nnunet --sigma_data 1.0 --planner_gpu_gb 48 \
        --num_workers "$CACHE_WORKERS" --loader_workers "$LOADER_WORKERS" $EXTRA_ARGS \
        > "logs/train_${mod}.log" 2>&1 &
    pid=$!
    echo "started ${mod} on GPU ${gpu} (pid ${pid}) -> logs/train_${mod}.log"
    wait "$pid"
    status=$?
    printf '%s\n' "$status" > "logs/train_${mod}.status"
    echo "${mod} ended on GPU ${gpu} (exit ${status})"
  done
}

mkdir -p logs
for mod in "${MODS[@]}"; do
  rm -f "logs/train_${mod}.status"   # never report a previous run's result
done

echo "Training ${#MODS[@]} modalities in ${N_SLOTS} parallel slot(s):"
for (( s = 0; s < N_SLOTS; s++ )); do
  # shellcheck disable=SC2086
  echo "  slot $(( s + 1 )) on GPU ${GPU_LIST[$s]}:$(printf ' %s' ${SLOT_MODS[$s]})"
done

SLOT_PIDS=()
for (( s = 0; s < N_SLOTS; s++ )); do
  # shellcheck disable=SC2086
  run_slot "${GPU_LIST[$s]}" ${SLOT_MODS[$s]} &
  SLOT_PIDS+=("$!")
done
echo "Follow progress with: tail -f logs/train_<modality>.log"

for (( s = 0; s < N_SLOTS; s++ )); do
  # An explicit conditional keeps set -e from aborting before the other slots are checked.
  if ! wait "${SLOT_PIDS[$s]}"; then
    echo "Slot on GPU ${GPU_LIST[$s]} terminated abnormally; its unfinished modalities are reported below." >&2
  fi
done

failed=0
for mod in "${MODS[@]}"; do
  status_file="logs/train_${mod}.status"
  if [[ ! -s "$status_file" ]]; then
    echo "${mod} FAILED (did not complete; no exit status recorded). See logs/train_${mod}.log" >&2
    failed=1
    continue
  fi
  status=$(<"$status_file")
  if [[ "$status" == "0" ]]; then
    echo "${mod} finished successfully (exit 0)."
  else
    echo "${mod} FAILED (exit ${status}). See logs/train_${mod}.log" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  echo "Four-modality training failed; all child processes have been checked." >&2
  exit 1
fi
echo "All four finished successfully."
