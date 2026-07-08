# BraTS2026 Stage 2 Repository

## Overview

This repository contains the complete training and inference pipeline for the BraTS2026 Stage 2 challenge.

The implementation is based on nnU-Net v2 with custom modifications specifically designed for RC lesion optimization.

---

## Main modifications

Compared with the original nnU-Net:

1. Fixed train/validation split.
2. RC-aware loss weighting.
3. Extended training schedule (1000 epochs).
4. Separate inference pipeline for pseudo-test evaluation.

---

## Repository structure

custom_nnunet/
    Modified nnU-Net trainer.

scripts/
    Dataset preparation scripts.

docs/
    Detailed documentation.

train.sh
    Training entry point.

infer.sh
    Inference entry point.

---

## Training

Run:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd ${PROJECT_ROOT}
mkdir -p logs
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_nyu.slurm
```

The Slurm job first runs `work_space/G2/code/g2_build_realonly_from_raw.py` on raw data. Cases missing `t2w` or any required file are recorded and skipped. It then reads the generated G2 real-only mapping/split and materializes train+val cases only. The internal locked test list is recorded but not included in the training dataset:

```text
work_space/S2/data/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly
work_space/S2/data/nnunet_preprocessed
work_space/S2/data/nnunet_results
work_space/S2/BraTS2026_S2_RC_v1.0/repository/data/splits/test_internal_locked.txt
```

Manual training after preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d 260 --verify_dataset_integrity
bash train.sh
```

`train.sh` automatically syncs the custom trainer into the active nnU-Net v2 environment before training.

---

## Inference

Run:

bash infer.sh INPUT_FOLDER OUTPUT_FOLDER

---

## Challenge notes

MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip
is treated as a pseudo-test set.

It is NOT used for:

- training,
- validation,
- parameter tuning.

It is ONLY used for final inference evaluation.
