#!/bin/bash
set -e

# 确保在 GliGAN 目录下
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== Step 1: 创建 CSV =========="
python src/train/csv_creator.py \
  --dataset BRATS_2024 \
  --datadir DataSet/patient_000/BraTS2024-BraTS-GLI-TrainingData \
  --logdir quick_test

echo "========== Step 2: 极速训练 =========="
python src/train/tumour_main_diffusion.py \
  --dataset BRATS_2024 \
  --modality t1ce \
  --logdir quick_test \
  --batch_size 1 \
  --in_channels 5 \
  --out_channels 1 \
  --num_steps 10 \
  --n_steps 10 \
  --beta_schedule cosine \
  --generator_type Unet \
  --optim_lr 2e-4 \
  --normalization minmax

echo "========== Step 3: 推理 =========="
python src/infer/generate_from_label.py \
  --label_path "DataSet/patient_000/BraTS2024-BraTS-GLI-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz" \
  --diffusion_ckpt_dir "../../Checkpoint/quick_test" \
  --dataset BRATS_2024 \
  --output_dir "./quick_test_output" \
  --generator_type Unet \
  --n_steps 10 \
  --beta_schedule cosine \
  --sampling_method ddim \
  --sampling_steps 5 \
  --eta 0.0 \
  --modality t1ce

echo "========== 全部完成 =========="
