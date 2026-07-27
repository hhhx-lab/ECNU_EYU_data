# S2 Inference Protocol

## Official Task 1 Validation

The official unlabeled validation dataset is a separate set from S2's 103-case internal validation split.

The downloaded source on the local workstation is:

```text
/Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据/Validation/
```

It has been audited as:

```text
cases     179
t1n       179
t1c       179
t2w       179
t2f       179
seg         0
```

Place the extracted directory on Greene at:

```text
${PROJ}/work_space/G1/data/raw/Validation/
```

Each case must retain the official folder structure:

```text
Validation/BraTS-MET-00833-000/
  BraTS-MET-00833-000-t1n.nii.gz
  BraTS-MET-00833-000-t1c.nii.gz
  BraTS-MET-00833-000-t2w.nii.gz
  BraTS-MET-00833-000-t2f.nii.gz
```

There must be no `seg` files. All 179 cases already contain T2W, so this official validation run does not require G1 missing-modality completion.

Current inference uses:

```text
Dataset263_BraTS2026_MET_RealOnly_Current/
nnUNetTrainerBraTS2026RC__nnUNetPlans__3d_fullres/
fold_0/checkpoint_final.pth
```

Submit from the project root:

```bash
PROJECT_ROOT=/scratch/bf2260/ECNU_EYU_data
cd "${PROJECT_ROOT}"
mkdir -p logs

S2_OFFICIAL_JOB=$(sbatch --parsable \
  --export=ALL,S2_EXPERIMENT_MODE=current,S2_INFERENCE_TARGET=official_validation \
  work_space/S2/slurm/legacy_realonly/04_s2_realonly_infer_nyu.slurm)

echo "S2_OFFICIAL_JOB=${S2_OFFICIAL_JOB}"
```

Override the source only when the server uses another location:

```bash
sbatch \
  --export=ALL,S2_EXPERIMENT_MODE=current,S2_INFERENCE_TARGET=official_validation,S2_OFFICIAL_VALIDATION_ROOT=/absolute/path/to/Validation \
  work_space/S2/slurm/legacy_realonly/04_s2_realonly_infer_nyu.slurm
```

The Slurm job performs the complete submission pipeline:

1. Checks exactly 179 official case directories.
2. Requires exactly `t1n/t1c/t2w/t2f` and rejects `seg` or unknown NIfTI files.
3. Creates the flat nnU-Net input with `0000=t1n, 0001=t1c, 0002=t2w, 0003=t2f`.
4. Runs the current fixed checkpoint with `nnUNetv2_predict -f 0`.
5. Requires exactly one prediction for every official case ID.
6. Checks labels are integers in `{0,1,2,3,4}`.
7. Checks dimensions, spacing, origin, orientation, and affine against each source case.
8. Creates a flat ZIP containing only the 179 prediction NIfTI files.

Outputs:

```text
work_space/S2/data/official_validation_nnunet_input/

work_space/S2/results/realonly_current_official_validation_predictions/
  BraTS-MET-00833-000.nii.gz
  ... exactly 179 predictions

work_space/S2/results/realonly_current_official_validation_submission/
  official_validation_input_manifest.csv
  official_validation_preparation.json
  official_submission_manifest.csv
  official_submission_validation.json
  S2_realonly_current_Task1_validation_179.zip
```

Only `S2_realonly_current_Task1_validation_179.zip` is uploaded to Synapse. Audit CSV/JSON files stay outside the archive.

Official filename and geometry rules are enforced directly: each output ends in the official 5-digit case ID plus 3-digit timepoint, for example `BraTS-MET-00833-000.nii.gz`, and exactly matches the source spatial metadata.

## Generic nnU-Net Input

For a non-official four-channel directory that is already flattened:

```text
<case_id>_0000.nii.gz  T1N
<case_id>_0001.nii.gz  T1C
<case_id>_0002.nii.gz  T2W
<case_id>_0003.nii.gz  T2F
```

Run:

```bash
sbatch \
  --export=ALL,S2_EXPERIMENT_MODE=current,S2_INFERENCE_TARGET=nnunet_input,S2_INFERENCE_INPUT=/path/to/nnunet_input \
  work_space/S2/slurm/legacy_realonly/04_s2_realonly_infer_nyu.slurm
```

The generic output defaults to `work_space/S2/results/realonly_current_fixed_inference/`. It does not create an official submission ZIP.

Use `S2_EXPERIMENT_MODE=legacy` only for historical Dataset260 inference. Official validation submission is locked to `current` mode.
