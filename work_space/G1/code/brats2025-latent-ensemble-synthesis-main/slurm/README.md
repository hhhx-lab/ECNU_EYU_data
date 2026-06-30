# G1 T2W Completion — SLURM Scripts for JH ARCH

## Quick Start

```bash
# 1. Submit data preparation (CPU only)
PREP_JOB=$(sbatch --parsable slurm/01_prepare_data.slurm)
echo "Prep job: ${PREP_JOB}"

# 2. Submit training (GPU, waits for prep)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} slurm/02_train.slurm)
echo "Train job: ${TRAIN_JOB}"

# 3. Submit eval+inference (GPU, waits for train)
sbatch --dependency=afterok:${TRAIN_JOB} slurm/03_evaluate_and_infer.slurm
```

Or submit individually and chain by job ID:

```bash
sbatch slurm/01_prepare_data.slurm
# Wait for completion, then:
sbatch --dependency=afterok:<PREP_JOB_ID> slurm/02_train.slurm
sbatch --dependency=afterok:<TRAIN_JOB_ID> slurm/03_evaluate_and_infer.slurm
```

## Script Overview

| Script | GPU | Walltime | What it does |
|--------|-----|----------|-------------|
| `01_prepare_data.slurm` | No | 4h | Data placement, preprocessing, split marking, attention masks, channel weights |
| `02_train.slurm` | 1 GPU | 72h | EncDec training → BBDM training |
| `03_evaluate_and_infer.slurm` | 1 GPU | 12h | Validation evaluation → inference on input_inference |

## Before You Run

1. **Verify `--account=`** in each script matches your group's allocation (`jhuchar` is the default).
2. **Verify conda environment name** (`brats2025`) — change if your env is named differently.
3. **Verify `work_space/` path** — the scripts assume `work_space/G1/` is accessible from the login node.
4. **Ensure G2 prerequisites exist** (see manual Section 2).
5. **Run `mkdir -p logs`** in the code directory before submitting (or let the scripts create them).

## Monitoring

```bash
# Check job status
squeue -u ${USER}

# View output
tail -f logs/prep_data_<JOB_ID>.out
tail -f logs/train_<JOB_ID>.out
tail -f logs/eval_infer_<JOB_ID>.out

# Cancel a job
scancel <JOB_ID>
```

## Customization

- **Walltime**: Adjust `#SBATCH --time=` if your dataset size differs significantly.
- **Memory**: Increase `--mem=` if training OOMs (BBDM on large batches may need >64G).
- **Account/QOS**: Change `--account=` and `--qos=` per your PI's allocation.
