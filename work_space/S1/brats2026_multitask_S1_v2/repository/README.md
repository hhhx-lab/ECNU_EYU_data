# BraTS2026 Multi-Task Framework

Multi-task framework for:

1. Tumor Subregion Segmentation
2. Recurrence Component Segmentation (RC)

based on a shared MONAI 3D UNet backbone.

---

# Dataset

BraTS-METS 2025 / BraTS 2026

MRI modalities:

* T1C
* T1N
* T2F
* T2W

Tumor labels:

* 0 = Background
* 1 = NETC
* 2 = SNFH
* 3 = ET

RC labels:

* 0 = Background
* 1 = RC

---

# Data Preparation

Apply corrected labels:

```bash
python scripts/01_apply_corrected_labels.py
```

Check invalid labels:

```bash
python scripts/02_find_invalid_labels.py
```

Fix invalid labels:

```bash
python scripts/03_fix_invalid_labels.py
```

Build multitask labels:

```bash
python scripts/08_build_full_multitask_labels.py
```

Audit dataset:

```bash
python scripts/09_dataset_audit.py
```

Create train/validation split:

```bash
python scripts/10_create_full_split.py
```

---

# Training

Full dataset training:

```bash
python trainers/trainer_v1_final.py \
  --config configs/multitask_v1_full.yaml
```

Resume training:

Set in:

```yaml
train:
  resume: checkpoints_full/latest.pth
```

then run:

```bash
python trainers/trainer_v1_final.py \
  --config configs/multitask_v1_full.yaml
```

---

# Inference

```bash
python inference/infer_multitask.py
```

---

# Model

Backbone:

* MONAI 3D UNet

Heads:

* Tumor Head (4 classes)
* RC Head (2 classes)

Loss:

* DiceCE Loss
* Uncertainty Weighting

