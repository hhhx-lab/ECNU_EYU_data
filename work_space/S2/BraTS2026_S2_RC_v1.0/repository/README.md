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

After training, create/check the dedicated official-evaluation environment and submit the CPU evaluation job:

```bash
bash work_space/S2/BraTS2026_S2_RC_v1.0/repository/scripts/setup_brats_eval_env.sh

sbatch --export=ALL,S2_EXPERIMENT_MODE=current,CONDA_ENV=brats_eval \
  work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_official_eval_nyu.slurm
```

The job runs `BraTS-evaluation==0.0.8` with the official MET config and writes `panoptica_evaluation_summary.json` plus `leaderboard_metrics.csv` under `work_space/S2/results/realonly_current_fixed_validation/`. G2 synthetic-data QC is not a substitute for this segmentation evaluation.

Historical Dataset260 recovery is available only through `S2_EXPERIMENT_MODE=legacy`. See `docs/S2_服务器运行手册.md`.
