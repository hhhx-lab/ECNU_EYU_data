# S2 Real-Only Fixed-Split Baseline — Dataset260, 3d_fullres

Historical result only. It does not use the current G2 patient-group split and must not be used as the paired baseline for current V2/V3 experiments.

nnU-Net v2 + custom nnUNetTrainerBraTS2026RC, real-only training data.
Two source masks excluded for out-of-range labels: BraTS-MET-01094-002 (label 6),
BraTS-MET-01184-002 (label 8). Train/val = 828/207; internal test locked = 259.
The directory name retains `fold0` because nnU-Net uses internal key `fold_0`;
this is one fixed-split baseline, not a cross-validation result.

Trained 1000 epochs. Mean validation Dice: 0.540 (see summary.json for per-class).

Files:
- summary.json  : per-class + mean Dice on the 207-case val set
- progress.png  : training loss / pseudo-Dice curves
- training_log_2026_7_10_05_32_06.txt : final-run epoch log
- debug.json    : run metadata
- nnUNetPlans.json / dataset.json : config to reproduce

Not tracked (large): checkpoint_*.pth, validation/*.nii.gz.
