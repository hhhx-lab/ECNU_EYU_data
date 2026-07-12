# BraTS2026 S2 RC Segmentation

nnU-Net v2 `3d_fullres` segmentation with RC-aware loss weighting.

## Current Baseline

```text
mode                   current
dataset                Dataset263_BraTS2026_MET_RealOnly_Current
train/val/test         823/103/104
nnU-Net key            fold_0 (API/storage only)
cross-validation       disabled
```

Run from the project root:

```bash
mkdir -p logs
PREP_JOB=$(sbatch --parsable --export=ALL,S2_EXPERIMENT_MODE=current \
  work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_prepare_nyu.slurm)
sbatch --dependency=afterok:${PREP_JOB} --export=ALL,S2_EXPERIMENT_MODE=current \
  work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_nyu.slurm
```

Do not add `--array`. Preparation checks the count contract, patient-group isolation, raw/preprocessed ID equality, and label integrity before training.

Historical Dataset260 recovery is available only through `S2_EXPERIMENT_MODE=legacy`. See `docs/S2_服务器运行手册.md`.
