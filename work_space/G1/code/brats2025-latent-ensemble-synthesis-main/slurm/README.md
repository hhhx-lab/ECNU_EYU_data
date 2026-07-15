# ECNU-EYU BraTS Slurm Scripts

## Quick Start

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
mkdir -p ${PROJECT_ROOT}/logs
cd ${PROJECT_ROOT}

# G1: submit data preparation (uses 1 GPU for VAE latent encoding)
PREP_JOB=$(sbatch --parsable work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/01_prepare_data_nyu.slurm)
echo "Prep job: ${PREP_JOB}"

# G1: submit training (GPU, waits for prep)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${PREP_JOB} work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/02_train_nyu.slurm)
echo "Train job: ${TRAIN_JOB}"

# G1: submit validation-only evaluation (GPU, waits for train)
EVAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/03_eval_val_nyu.slurm)
echo "Eval job: ${EVAL_JOB}"

# Review work_space/G1/data/eval_metrics_val.csv and eval_synthesized_val/.
# Submit missing-T2W inference only after validation output is acceptable:
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_infer_missing_t2w_nyu.slurm

# S2: prepare the current Dataset263 patient-group fixed split (CPU)
S2_PREP_JOB=$(sbatch --parsable --export=ALL,S2_EXPERIMENT_MODE=current work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_prepare_nyu.slurm)

# S2: submit the single current fixed-split model
S2_TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${S2_PREP_JOB} --export=ALL,S2_EXPERIMENT_MODE=current work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_nyu.slurm)

# S2: run BraTS metrics on the current fixed 103-case validation set
sbatch --dependency=afterok:${S2_TRAIN_JOB} --export=ALL,S2_EXPERIMENT_MODE=current,CONDA_ENV=brats_eval work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_official_eval_nyu.slurm

# S2: official 179-case validation inference and Synapse ZIP packaging
sbatch --export=ALL,S2_EXPERIMENT_MODE=current,S2_INFERENCE_TARGET=official_validation work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_infer_nyu.slurm

# S1: original-data MONAI multitask real-only baseline
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/05_s1_realonly_nyu.slurm
```

Or submit individually and chain by job ID:

```bash
cd /path/to/ECNU_EYU_data
mkdir -p logs
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/01_prepare_data_nyu.slurm
# Wait for completion, then:
sbatch --dependency=afterok:<PREP_JOB_ID> work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/02_train_nyu.slurm
sbatch --dependency=afterok:<TRAIN_JOB_ID> work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/03_eval_val_nyu.slurm
# Review val metrics/images, then:
sbatch work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_infer_missing_t2w_nyu.slurm
```

## Script Overview

| Script | GPU | Walltime | What it does |
|--------|-----|----------|-------------|
| `01_prepare_data_nyu.slurm` | 1 GPU | 4h | Data placement, VAE latent preprocessing, split marking, attention masks, channel weights |
| `02_train_nyu.slurm` | 1 GPU | 72h | EncDec training -> BBDM training |
| `03_eval_val_nyu.slurm` | 1 GPU | 12h | Fixed val split evaluation only; writes `eval_metrics_val.csv` and `eval_synthesized_val/` |
| `04_infer_missing_t2w_nyu.slurm` | 1 GPU | 12h | Missing-T2W inference on `input_inference/`; run only after val output is acceptable |
| `04_s2_realonly_prepare_nyu.slurm` | CPU | 24h | Build current Dataset263 fixed split and preprocessing; optional legacy recovery |
| `04_s2_realonly_nyu.slurm` | 1 GPU | 96h | S2 fixed-split training; skip complete model or resume interruption |
| `04_s2_realonly_official_eval_nyu.slurm` | CPU | 24h | Run BraTS MET metrics on the fixed validation set |
| `04_s2_realonly_infer_nyu.slurm` | 1 GPU | 24h | Official 179-case inference, spatial/label audit, and Synapse ZIP packaging; generic mode remains available |
| `05_s1_realonly_nyu.slurm` | 1 GPU | 96h | S1 MONAI multitask real-only baseline from G2 mapping/split |

## S1/S2 Real-Only Baselines

The S1/S2 scripts are for the original-data comparison path. They do not use G1 generated completion data or synthetic augmentation. They first run the G2 raw intake script to scan raw data, prefer corrected labels when available, skip incomplete cases such as cases without T2W, skip cases that still contain illegal label values, and generate the real-only mapping/split artifacts used by S1/S2/S3.

Default project root:

```bash
/scratch/bf2260/ECNU_EYU_data
```

S2 defaults to `G2_REFRESH_REALONLY=0` so a submitted run consumes the reviewed,
versioned G2 artifacts. Current mode may explicitly refresh from raw data when
the official fake/broken-T2W list and corrected labels are available; legacy
mode never refreshes identities:

```text
work_space/G2/code/g2_build_realonly_from_raw.py
work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training/
```

Generated artifacts:

```text
work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv
work_space/G2/results/splits/splits_final_train_val_test.json
work_space/G2/results/manifests/realonly_skipped_incomplete_cases.csv
```

`realonly_skipped_incomplete_cases.csv` records cases missing `t1n/t1c/t2w/t2f/seg` and cases without a clean corrected/raw label. Those cases are not materialized into S1/S2 training data.

Current S2 mode writes a train+val-only symlinked nnU-Net raw dataset for the G2 `823/103/104` patient-group split. The 104-case locked test is listed but not materialized into training:

```text
work_space/S2/data/nnunet_raw/Dataset263_BraTS2026_MET_RealOnly_Current
work_space/S2/data/nnunet_preprocessed
work_space/S2/data/nnunet_results
work_space/S2/BraTS2026_S2_RC_v1.0/repository/data/splits/current/nnunet_case_mapping_realonly_train_val.csv
```

Default `S2_EXPERIMENT_MODE=current` enforces `823/103/104`, patient-group
isolation, and exact equality among split, raw Dataset263, and preprocessed IDs.
Historical Dataset260 recovery is available only with
`S2_EXPERIMENT_MODE=legacy`; it writes separate split files and is not a valid
paired baseline for current G2 experiments.

The training entry is submitted once:

```bash
bash train.sh
```

Do not submit `04_s2_realonly_nyu.slurm` with `--array`. nnU-Net still stores the fixed model under `fold_0` for API compatibility.

`05_s1_realonly_nyu.slurm` writes a symlinked S1 training view:

```text
work_space/S1/data/real_only_cases
work_space/S1/results/realonly/checkpoints
work_space/S1/results/realonly/tensorboard
work_space/S1/results/realonly/predictions
```

It then generates `tumor_label.nii.gz` and `rc_label.nii.gz` inside the S1 view, audits the view, and starts `trainer_v1_final.py`.

Submit with a different project root or environment when needed:

```bash
cd /path/to/ECNU_EYU_data
mkdir -p logs
S2_PREP_JOB=$(sbatch --parsable --export=ALL,PROJ=/path/to/ECNU_EYU_data,CONDA_ENV=brats2026_s2,S2_EXPERIMENT_MODE=current work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_prepare_nyu.slurm)
sbatch --dependency=afterok:${S2_PREP_JOB} --export=ALL,PROJ=/path/to/ECNU_EYU_data,CONDA_ENV=brats2026_s2,S2_EXPERIMENT_MODE=current work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_nyu.slurm
sbatch --export=ALL,PROJ=/path/to/ECNU_EYU_data,CONDA_ENV=brats2026_s1 work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/05_s1_realonly_nyu.slurm
```

## Before You Run

1. **Verify `--account=`** in each script matches your group's allocation.
2. **Verify conda environment name**. G1 defaults to `brats2025`; S1/S2 default to `brats2026_s1` and `brats2026_s2`.
3. **Verify project path**. S1/S2 default to `/scratch/bf2260/ECNU_EYU_data`; override `PROJ` at submit time if needed.
4. **Ensure raw data exists** under `work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training/`, or submit with `RAW_DATA_DIR=/path/to/raw`.
5. **Submit from the project root and ensure `logs/` exists before submitting.** The Slurm `--output/--error` paths are opened before the shell body runs.
6. **If your project root is not `/scratch/bf2260/ECNU_EYU_data`, pass `PROJ=/path/to/ECNU_EYU_data` at submit time.**
7. **G1 data preparation uses `work_space/G2/results/manifests/real_train_manifest.csv` by default.** This is the mixed raw-layout manifest covering root-level cases and `UCSD - Training` cases. Override with `G2_REAL_MANIFEST=/path/to/manifest.csv` only if the server data layout is intentionally different.
8. **S2 official inference requires the extracted 179-case validation set at `work_space/G1/data/raw/Validation/`.** It must contain four modalities per case and no `seg`; override with `S2_OFFICIAL_VALIDATION_ROOT=/absolute/path/to/Validation` only when necessary.

## Monitoring

```bash
# Check job status
squeue -u ${USER}

# View output
tail -f logs/prep_data_<JOB_ID>.out
tail -f logs/train_<JOB_ID>.out
tail -f logs/eval_val_<JOB_ID>.out
tail -f logs/infer_t2w_<JOB_ID>.out
tail -f logs/s2_realonly_prepare_<JOB_ID>.out
tail -f logs/s2_realonly_fixed_<JOB_ID>.out
tail -f logs/s1_realonly_<JOB_ID>.out

# Cancel a job
scancel <JOB_ID>
```

## Customization

- **Walltime**: Adjust `#SBATCH --time=` if your dataset size differs significantly.
- **Memory**: Increase `--mem=` if training OOMs (BBDM on large batches may need >64G).
- **Account/QOS**: Change `--account=` and `--qos=` per your PI's allocation.
