#!/bin/bash

export nnUNet_raw=/root/autodl-tmp/nnunet_raw
export nnUNet_preprocessed=/root/autodl-tmp/nnunet_preprocessed
export nnUNet_results=/root/autodl-tmp/nnunet_results

export nnUNet_extTrainer=$(pwd)/custom_nnunet
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Starting BraTS2026 RC training..."

nnUNetv2_train \
    501 3d_fullres 0 \
    -tr nnUNetTrainerBraTS2026RC \
    -num_gpus 1

