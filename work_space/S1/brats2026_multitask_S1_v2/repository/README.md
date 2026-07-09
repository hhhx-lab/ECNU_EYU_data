# BraTS2026 Multi-Task Framework

Multi-task framework for:

1. Tumor Subregion Segmentation
2. Recurrence Component Segmentation (RC)

based on a shared MONAI 3D UNet backbone.

---

# Dataset

BraTS-METS 2025 / BraTS 2026

MRI modalities:

* T1N
* T1C
* T2W
* T2F

Channel order:

* 0 = T1N
* 1 = T1C
* 2 = T2W
* 3 = T2F

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

Recommended real-only path on the shared project:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd ${PROJECT_ROOT}
mkdir -p logs
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/05_s1_realonly_nyu.slurm
```

That Slurm job first runs `work_space/G2/code/g2_build_realonly_from_raw.py` on the raw data directory. The G2 intake prefers corrected labels when available; cases missing `t2w`, missing any required file, or still containing illegal label values without a clean corrected label are recorded in `realonly_skipped_incomplete_cases.csv` and are not used for training. It then reads the generated G2 real-only mapping/split and builds:

```text
work_space/S1/data/real_only_cases
work_space/S1/results/realonly/checkpoints
work_space/S1/results/realonly/tensorboard
```

The original data are not edited. Four modalities and the effective seg label are symlinked into the S1 view, then `tumor_label.nii.gz` and `rc_label.nii.gz` are generated inside that view only.

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
export BRATS_TRAIN_ROOT=/scratch/bf2260/ECNU_EYU_data/work_space/S1/data/real_only_cases
export BRATS_SPLIT_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/S1/brats2026_multitask_S1_v2/repository/data/splits
export S1_CHECKPOINT_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/S1/results/realonly/checkpoints
export S1_TENSORBOARD_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/S1/results/realonly/tensorboard
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
