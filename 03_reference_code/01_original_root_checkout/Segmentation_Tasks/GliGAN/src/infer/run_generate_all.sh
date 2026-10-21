#!/bin/bash
# Pipeline 2: Generate all 4-modality brain MRI from a tumour label
# Usage: bash run_generate_all.sh
set -e

LABEL="../../DataSet/patient_000/BraTS2024-BraTS-GLI-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz"
CKPT="../../Checkpoint/brats2024"
DATASET="BRATS_2024"
OUTDIR="./generated_scans"
STEPS=50

python generate_from_label.py \
    --label_path="$LABEL" \
    --diffusion_ckpt_dir="$CKPT" \
    --dataset="$DATASET" \
    --output_dir="$OUTDIR" \
    --sampling_method="ddim" \
    --sampling_steps="$STEPS" \
    --modality="all" \
    --device="cuda"
