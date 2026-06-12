# Training Protocol

## Dataset

Dataset ID:

501

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

1167 cases

Validation:

129 cases

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

bash train.sh

