# Training

## Sample Training

Dataset:

sample50

Train:

40

Validation:

10

Patch Size:

96 x 96 x 96

Command:

python trainers/trainer_v1_final.py

## Full Training

Use:

configs/multitask_v1_full.yaml

Recommended:

4 x A100

300 epochs

Mixed Precision

Checkpoint:

best.pth

latest.pth

## G2-Aligned Real-Only Slurm

Use this for the original-data baseline:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd ${PROJECT_ROOT}
mkdir -p logs
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/05_s1_realonly_nyu.slurm
```

Main inputs:

```text
work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training/
work_space/G2/code/g2_build_realonly_from_raw.py
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_train_val_test.json
```

Main outputs:

```text
work_space/S1/data/real_only_cases
work_space/S1/results/realonly/checkpoints
work_space/S1/results/realonly/tensorboard
```

Manual equivalent:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
export BRATS_TRAIN_ROOT=${PROJECT_ROOT}/work_space/S1/data/real_only_cases
export BRATS_SPLIT_DIR=${PROJECT_ROOT}/work_space/S1/brats2026_multitask_S1_v2/repository/data/splits
export S1_CHECKPOINT_DIR=${PROJECT_ROOT}/work_space/S1/results/realonly/checkpoints
export S1_TENSORBOARD_DIR=${PROJECT_ROOT}/work_space/S1/results/realonly/tensorboard
python trainers/trainer_v1_final.py --config configs/multitask_v1_full.yaml
```
