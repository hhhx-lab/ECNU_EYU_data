# S2 Inference Protocol

Each input case requires:

```text
<case_id>_0000.nii.gz  T1N
<case_id>_0001.nii.gz  T1C
<case_id>_0002.nii.gz  T2W
<case_id>_0003.nii.gz  T2F
```

Current inference uses:

```text
Dataset263_BraTS2026_MET_RealOnly_Current/
nnUNetTrainerBraTS2026RC__nnUNetPlans__3d_fullres/
fold_0/checkpoint_final.pth
```

```bash
sbatch \
  --export=ALL,S2_EXPERIMENT_MODE=current,S2_INFERENCE_INPUT=/path/to/nnunet_input \
  work_space/G1/code/brats2025-latent-ensemble-synthesis-main/slurm/04_s2_realonly_infer_nyu.slurm
```

Default output is `work_space/S2/results/realonly_current_fixed_inference/`. The script always calls `nnUNetv2_predict -f 0` and rejects multi-fold inference.

Use `S2_EXPERIMENT_MODE=legacy` only to infer with the historical Dataset260 checkpoint.
