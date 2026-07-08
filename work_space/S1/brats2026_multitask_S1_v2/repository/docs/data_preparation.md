# Data Preparation

## Source

BraTS2025 MET Training Dataset from raw data:

```text
work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training/
```

The Slurm entry first generates these lightweight G2 artifacts from raw data:

```text
work_space/G2/code/g2_build_realonly_from_raw.py
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_train_val_test.json
work_space/G2/results/manifests/realonly_skipped_incomplete_cases.csv
```

On the server, use the S1 real-only Slurm script:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd ${PROJECT_ROOT}
mkdir -p logs
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/05_s1_realonly_nyu.slurm
```

It creates a symlinked training view under:

```text
work_space/S1/data/real_only_cases
```

The raw training data directory is not modified.

## Processing Pipeline

For the G2-aligned real-only run, the Slurm script performs:

1. Scan raw data and generate `nnunet_case_mapping_realonly.csv`.
2. Skip cases missing `t2w` or any required file.
3. Generate `splits_final_train_val_test.json`.
4. Materialize train/val cases into `work_space/S1/data/real_only_cases`.
5. Write S1 split files with original `BraTS-MET-*` case IDs.
6. Run `08_build_full_multitask_labels.py`.
7. Run `09_dataset_audit.py`.
8. Run `trainer_v1_final.py`.

The S1 auxiliary labels are:

tumor_label.nii.gz

rc_label.nii.gz

Legacy standalone utilities are still available:

01_apply_corrected_labels.py

Replace corrected labels.

02_find_invalid_labels.py

Detect illegal labels.

03_fix_invalid_labels.py

Repair remaining label issues.

## Tumor Labels

0 Background

1 NETC

2 SNFH

3 ET

## RC Labels

0 Non-recurrence

1 Recurrence
