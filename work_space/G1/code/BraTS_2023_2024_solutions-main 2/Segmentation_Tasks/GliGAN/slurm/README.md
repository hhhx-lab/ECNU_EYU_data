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

`04_generate_visual_v2_nyu.slurm` 不是批量 augmentation 生产入口。它只生成一个可视化病例，也不会创建完整的 G2 run metadata。

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

## G2 Handoff

正式 V2 批量输出必须带 `generation_config.json`，记录 run ID、seed、source manifest、`sampling_method`、`sampling_steps`、`eta`、`crop_size` 及四模态 checkpoint 或 checkpoint 目录。批量生成后不要把平铺输出直接交给分割模型，应运行：

```bash
python work_space/G2/code/g2_v2_compose_augmentation.py \
  --v2-output-root /path/to/v2_flat_output \
  --source-manifest work_space/G2/results/manifests/g1_v2_source_manifest.csv \
  --output-run-root /path/to/g2_composed/v2_run_id

python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root /path/to/g2_composed/v2_run_id \
  --generation-mode full_generation
```

`--output-run-root` 必须是空目录。只有确认整轮重建时才加 `--overwrite`；它会清空整个 composed run，而不是只覆盖单个病例。

只有 `allowed_as_v2_source=True` 的病例可生成。当前 source manifest 放行 823 个 authentic-T2W master-train 病例。G2 会恢复 source 强度和几何、复制 corrected seg、分配稳定 synthetic ID，并保证 augmentation 不进入 val/test。
