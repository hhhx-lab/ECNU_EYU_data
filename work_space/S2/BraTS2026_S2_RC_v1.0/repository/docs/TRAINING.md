# Training Protocol

## Dataset

Dataset ID:

260

Configuration:

3d_fullres

---

## Trainer

Custom trainer:

nnUNetTrainerBraTS2026RC

Location:

custom_nnunet/nnUNetTrainerBraTS2026RC.py

---

## Modifications

### Fixed split

Training:

829 cases

Validation:

207 cases

Internal locked test:

259 cases, not materialized into the training dataset

No overlap exists.

---

### RC loss weighting

Cross-entropy weights:

[1, 1, 1, 1, 3]

The RC class receives a weight of 3.

---

### Epochs

Training duration:

1000 epochs

---

## Training command

Recommended server submission:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd ${PROJECT_ROOT}
mkdir -p logs
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_nyu.slurm
```

The Slurm job refreshes `work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv` and `work_space/G2/results/splits/splits_final_train_val_test.json` from raw data before conversion. The G2 intake prefers corrected labels when available. Cases missing `t2w`, missing any required file, or still containing illegal label values without a clean corrected label are written to `work_space/G2/results/manifests/realonly_skipped_incomplete_cases.csv` and skipped.

Manual command after conversion and preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d 260 --verify_dataset_integrity
bash train.sh
```

`train.sh` first copies `custom_nnunet/nnUNetTrainerBraTS2026RC.py` into the active Python environment under `nnunetv2/training/nnUNetTrainer/`. This makes `nnUNetv2_train -tr nnUNetTrainerBraTS2026RC` discover the custom trainer on the server.
