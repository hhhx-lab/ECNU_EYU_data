# Training

## Sample Training

Dataset:

sample50

Train:

40

Validation:

10

Patch Size:

96 x 96 x 96

Command:

```bash
python trainers/trainer_v1_final.py --config configs/multitask_v1.yaml
```

## Full Training

Use:

```text
configs/multitask_v1_full.yaml
```

Recommended hardware:

- GPU with **≥40GB** memory (A100 / H100 preferred)
- Do **not** rely on a generic `gpu:1` that may land on 16/24GB cards

Recommended defaults (already in the full config):

| Setting | Value | Reason |
| --- | --- | --- |
| `batch_size` | 1 | Peak activation memory of 96³ UNet |
| `gradient_accumulation` | 2 | Keep effective batch ≈ 2 |
| `amp_dtype` | `auto` | BF16 on Ampere/Hopper, else FP16 |
| `lesion_probability` | 0.8 | Small-met balanced sampling |
| `normalize` | true | Per-modality nonzero Z-score |
| validation | full-volume joint SWI | Avoid center-crop missing peripheral mets |
| best checkpoint | `checkpoint_score` | Region Dice + lesion/small-F1 proxies |
| scheduler | `plateau` | ReduceLROnPlateau on full-val score |
| early stopping | patience 30 | Stop when full-val score stalls |

If still OOM after batch_size=1 + AMP:

1. Set `train.patch_size` / `validation.roi_size` to `[80, 80, 80]`
2. Only then consider narrowing `model.channels`

Checkpoint files:

```text
best.pth      # highest full-val checkpoint_score
latest.pth    # last epoch
resolved_config.yaml
val_metrics_history.json
```

## G2-Aligned Real-Only Slurm

Use this for the original-data baseline (canonical entry):

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd ${PROJECT_ROOT}
mkdir -p logs
sbatch --gres=gpu:a100:1 \
  --export=ALL,PROJ=${PROJECT_ROOT} \
  work_space/S1/slurm/01_s1_realonly.slurm
```

Full server manual:

```text
work_space/S1/docs/S1_服务器运行手册.md
```

Legacy wrapper (same runner):

```bash
sbatch --gres=gpu:a100:1 \
  --export=ALL,PROJ=${PROJECT_ROOT} \
  work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/05_s1_realonly_nyu.slurm
```

Main inputs:

```text
work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training/
work_space/G2/code/g2_build_realonly_from_raw.py
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_train_val_test.json
```

Main outputs:

```text
work_space/S1/data/real_only_cases
work_space/S1/results/realonly/checkpoints
work_space/S1/results/realonly/tensorboard
```

Manual equivalent:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
export BRATS_TRAIN_ROOT=${PROJECT_ROOT}/work_space/S1/data/real_only_cases
export BRATS_SPLIT_DIR=${PROJECT_ROOT}/work_space/S1/brats2026_multitask_S1_v2/repository/data/splits
export S1_CHECKPOINT_DIR=${PROJECT_ROOT}/work_space/S1/results/realonly/checkpoints
export S1_TENSORBOARD_DIR=${PROJECT_ROOT}/work_space/S1/results/realonly/tensorboard
python trainers/trainer_v1_final.py --config configs/multitask_v1_full.yaml
```

## What the trainer monitors

TensorBoard scalars include:

- `train/loss`, `train/tumor_loss`, `train/rc_loss`
- `train/weight_tumor`, `train/weight_rc` (uncertainty weights)
- `train/lr`
- `val/checkpoint_score`, `val/region_dice_mean`, per-region Dice / lesion / small-F1 proxies

If `weight_rc` becomes much smaller than `weight_tumor`, the trainer prints a warning.
Tighten `loss.max_log_sigma` or set `loss.use_uncertainty: false` with fixed weights.
