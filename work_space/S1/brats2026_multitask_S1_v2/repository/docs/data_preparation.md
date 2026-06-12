# Data Preparation

## Source

BraTS2025 MET Training Dataset

## Processing Pipeline

01_apply_corrected_labels.py

Replace corrected labels.

02_find_invalid_labels.py

Detect illegal labels.

03_fix_invalid_labels.py

Repair remaining label issues.

04_create_sample50.py

Generate debugging subset.

05_create_split.py

Generate train/validation split.

06_build_multitask_dataset.py

Create:

tumor_label.nii.gz

rc_label.nii.gz

## Tumor Labels

0 Background

1 NETC

2 SNFH

3 ET

## RC Labels

0 Non-recurrence

1 Recurrence

