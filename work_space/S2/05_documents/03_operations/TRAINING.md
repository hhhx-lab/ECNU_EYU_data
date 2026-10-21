# S2 Training Protocol

## Current Configuration

- Dataset ID: `263`
- Dataset name: `Dataset263_BraTS2026_MET_RealOnly_Current`
- Fixed split: `823 train / 103 val / 104 test`
- Configuration: `3d_fullres`
- Trainer: `nnUNetTrainerBraTS2026RC`
- Epochs: `1000`
- RC cross-entropy weight: `3`
- Cross-validation: disabled

```bash
PREP_JOB=$(sbatch --parsable --export=ALL,S2_EXPERIMENT_MODE=current \
  work_space/S2/slurm/legacy_realonly/04_s2_realonly_prepare_nyu.slurm)

TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} \
  --export=ALL,S2_EXPERIMENT_MODE=current \
  work_space/S2/slurm/legacy_realonly/04_s2_realonly_nyu.slurm)
```

The job is submitted once and always calls nnU-Net key `0`. `train.sh` rejects other fold values, validates the `823/103` contract and cache ID space, resumes `checkpoint_latest.pth`, validates an existing final checkpoint, or skips a fully completed model.

Historical Dataset260 execution requires `S2_EXPERIMENT_MODE=legacy` for both preparation and training.
