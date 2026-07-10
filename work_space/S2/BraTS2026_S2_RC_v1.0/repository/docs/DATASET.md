# Dataset Description

Dataset ID:

260

Dataset name:

Dataset260_BraTS2026_MET_RealOnly

---

## Five-fold split

Current CV pool after preserving the completed fold-0 exclusions:

1035 cases

Per-fold training:

828 cases

Per-fold validation:

207 cases

Internal locked test:

259 cases

Train/validation overlap within each fold:

0

Each CV-pool case appears in validation exactly once across folds 0-4. Fold 0
keeps the completed fixed validation set unchanged. The 259-case internal locked
test never enters Dataset260 training or validation.

---

## Labels

Labels come from the G2 real-only mapping. The mapping prefers official corrected labels when available. A raw segmentation is used only when its label values are inside `{0,1,2,3,4}`. Cases that still contain illegal labels without a clean corrected label are skipped before nnU-Net materialization.

RC lesions are included in the segmentation targets.

## Channel order

The nnU-Net image suffixes are unified with G2 materialized datasets:

```text
0000 = T1N
0001 = T1C
0002 = T2W
0003 = T2F
```

---

## Environment variables

nnUNet_raw

nnUNet_preprocessed

nnUNet_results

must be configured before training.

If they are not set, `train.sh` and `infer.sh` use repository-local defaults:

```text
data/nnunet_raw
data/nnunet_preprocessed
data/nnunet_results
```

Data conversion can also be controlled with:

```text
BRATS_TRAIN_ROOT
NNUNET_DATASET_DIR
BRATS_NNUNET_MAPPING_CSV
BRATS_MATERIALIZE_MODE
BRATS_SPLIT_DIR
S2_FOLD
```

Recommended real-only conversion command after generating the train+val-only mapping file:

```bash
python scripts/01_convert_to_nnunet.py \
  --mapping-csv /scratch/bf2260/ECNU_EYU_data/work_space/S2/BraTS2026_S2_RC_v1.0/repository/data/splits/nnunet_case_mapping_realonly_train_val.csv \
  --dst /scratch/bf2260/ECNU_EYU_data/work_space/S2/data/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly \
  --mode symlink \
  --clean
```
