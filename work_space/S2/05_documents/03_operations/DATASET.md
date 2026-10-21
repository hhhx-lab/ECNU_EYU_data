# S2 Dataset

## Current Dataset

```text
Dataset263_BraTS2026_MET_RealOnly_Current
mapping: work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
split:   work_space/G2/results/splits/splits_final_train_val_test.json
counts:  823 train / 103 val / 104 test
```

Only 926 train+val cases are materialized into `imagesTr/labelsTr`. The 104-case test remains locked outside training preprocessing.

Channel order:

```text
0000=T1N
0001=T1C
0002=T2W
0003=T2F
labels={0,1,2,3,4}
```

Split artifacts live under `data/splits/current/`:

```text
train_fixed.txt
val_fixed.txt
test_internal_locked.txt
test_internal_locked_source_ids.txt
fixed_split_membership.csv
fixed_split_summary.json
fixed_split_cache_audit.json
nnunet_case_mapping_realonly_train_val.csv
```

`04_build_fixed_split.py` enforces `823/103/104`, source-case isolation, and patient-group isolation. `05_validate_fixed_split_cache.py` requires exact equality among train+val split IDs, raw Dataset263 IDs, and preprocessed cache IDs.

## Historical Dataset260

`Dataset260_BraTS2026_MET_RealOnly` uses historical `828/207/259`. Legacy recovery derives train/val from the actual Dataset260 and `fold_0/validation`, uses the master mapping to recover source identities, and writes separate artifacts under `data/splits/legacy/`.

Legacy Dataset260 is not a valid paired baseline for the current G2 split.

## Official Unlabeled Validation

The downloadable Task 1 validation set is independent of the internal split:

```text
source: work_space/G1/data/raw/Validation/
count:  179 cases
files:  t1n/t1c/t2w/t2f only; no seg
```

`scripts/06_prepare_official_validation.py` validates the source tree and creates a flat inference view under `work_space/S2/data/official_validation_nnunet_input/`. Source case IDs are preserved, so nnU-Net emits official-compatible filenames such as `BraTS-MET-00833-000.nii.gz`.

This set is inference-only. It must never be added to `imagesTr`, `labelsTr`, the 823/103 split, model selection, or local metric computation because its ground truth is private.
