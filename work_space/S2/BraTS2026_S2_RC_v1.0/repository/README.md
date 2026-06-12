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

bash train.sh

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

