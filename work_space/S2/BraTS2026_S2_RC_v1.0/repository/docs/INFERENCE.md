# Inference Protocol

## Input

`infer.sh` expects an nnU-Net input directory. Each case must contain four files:

```text
<case_id>_0000.nii.gz  # T1N
<case_id>_0001.nii.gz  # T1C
<case_id>_0002.nii.gz  # T2W
<case_id>_0003.nii.gz  # T2F
```

The pseudo-test or official test dataset must not be used for training,
validation, model selection, or parameter tuning.

## Five-fold ensemble

After folds 0-4 are complete, run:

```bash
bash infer.sh INPUT_FOLDER OUTPUT_FOLDER
```

Defaults:

```text
Dataset ID   = 260
Configuration= 3d_fullres
Trainer      = nnUNetTrainerBraTS2026RC
Folds        = 0 1 2 3 4
```

`nnUNetv2_predict` ensembles all five folds. The script verifies every
`checkpoint_final.pth` before inference.

For a temporary fold-0-only smoke test:

```bash
S2_FOLDS=0 bash infer.sh INPUT_FOLDER OUTPUT_FOLDER
```

Fold-0-only output is not the final five-fold result.

## Slurm

```bash
sbatch \
  --export=ALL,S2_INFERENCE_INPUT=/path/to/nnunet_input \
  work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_infer_nyu.slurm
```

The default output directory is:

```text
work_space/S2/results/realonly_5fold_inference/
```
