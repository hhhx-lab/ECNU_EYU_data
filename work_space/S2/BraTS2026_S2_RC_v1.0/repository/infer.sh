#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage:"
    echo "bash infer.sh INPUT_FOLDER OUTPUT_FOLDER"
    exit 1
fi

export nnUNet_raw=/root/autodl-tmp/nnunet_raw
export nnUNet_preprocessed=/root/autodl-tmp/nnunet_preprocessed
export nnUNet_results=/root/autodl-tmp/nnunet_results

export nnUNet_extTrainer=$(pwd)/custom_nnunet
export PYTHONPATH=$(pwd):$PYTHONPATH

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

nnUNetv2_predict \
    -i ${INPUT_FOLDER} \
    -o ${OUTPUT_FOLDER} \
    -d 501 \
    -c 3d_fullres \
    -f 0

