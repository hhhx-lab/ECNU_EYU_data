# G1 Diffusion V2 SLURM Scripts

These scripts run the V2 GliGAN diffusion augmentation line under:

```text
work_space/G1/code/BraTS_2023_2024_solutions-main 2/Segmentation_Tasks/GliGAN
```

They follow `README_DIFFUSION.md`.

## Scripts

| Script | GPU | Purpose |
|---|---:|---|
| `01_create_csv_v2_nyu.slurm` | 0 | Create lesion-level CSV with fixed train/val split |
| `02_train_4modal_v2_nyu.slurm` | 4 GPU | Train `t1c/t1n/t2w/t2f`, one modality per GPU |
| `03_eval_v2_nyu.slurm` | 1 GPU | Run generation-backed validation evaluation; default `whole_brain`, `split=val` |
| `04_generate_visual_v2_nyu.slurm` | 1 GPU | Generate one case for visual inspection |

## Before Submit

1. Put complete non-missing cases under `DataSet/`, one case per folder.
2. Keep fake/broken T2W cases out of `DataSet/` for this first V2 run.
3. Check account and conda settings at the top of each script.
4. Ensure the log folder exists:

```bash
mkdir -p /scratch/bf2260/ECNU_EYU_data/logs
```

## Submit

```bash
CSV_JOB=$(sbatch --parsable slurm/01_create_csv_v2_nyu.slurm)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${CSV_JOB} slurm/02_train_4modal_v2_nyu.slurm)
sbatch --dependency=afterok:${TRAIN_JOB} slurm/03_eval_v2_nyu.slurm
sbatch --dependency=afterok:${TRAIN_JOB} slurm/04_generate_visual_v2_nyu.slurm
```

## Monitor

```bash
squeue -u ${USER}
tail -f /scratch/bf2260/ECNU_EYU_data/logs/g1_diffv2_train4_<JOB_ID>.out

python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t1c --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t1n --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t2w --live
python scripts/watch_loss.py ../../Checkpoint/brats2026_diffusion_v2_complete_only t2f --live
```

## Useful Overrides

```bash
LOGDIR=my_test sbatch slurm/01_create_csv_v2_nyu.slurm
LOGDIR=my_test sbatch slurm/02_train_4modal_v2_nyu.slurm
LOGDIR=my_test EVAL_MODE=patch EVAL_SPLIT=val MAX_CASES=20 sbatch slurm/03_eval_v2_nyu.slurm
LOGDIR=my_test CASE_ID=BraTS-MET-00004-000 sbatch slurm/04_generate_visual_v2_nyu.slurm
```

The scripts use Greene-compatible generic GPU requests (`#SBATCH --gres=gpu:N`). If the allocation has a required GPU partition or constraint for A100 nodes, add that local cluster option at the top of the script before submitting.
