#!/bin/bash
# Pipeline 2: Generate single-modality T1c MRI from a tumour label
# Usage: bash run_generate_t1ce.sh
set -e

: "${TASK1_ROOT:?Please export TASK1_ROOT to the 2026 raw data root before running}"

LABEL="$TASK1_ROOT/BraTS-MET-00001-000/BraTS-MET-00001-000-seg.nii.gz"
CKPT="../../Checkpoint/brats2026_diffusion"
DATASET="BRATS_2026"
OUTDIR="./generated_scans"
STEPS=50

python generate_from_label.py \
    --label_path="$LABEL" \
    --diffusion_ckpt_dir="$CKPT" \
    --dataset="$DATASET" \
    --output_dir="$OUTDIR" \
    --sampling_method="ddim" \
    --sampling_steps="$STEPS" \
    --modality="t1c" \
    --device="cuda"
