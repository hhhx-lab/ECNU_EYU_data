# G1 VAE Extended Fine-tuning 20260901 R3 Attempt 03

Status: `experimental_unvalidated`  
Operator approved: `false`  
Formal gate: `not_run_not_passed`

Attempt 03 is an independent correction after attempt-02 CPU scan job `3554513`
failed before any GPU training was submitted. The previous scan processed all
823 rows and identified 16 valid cases whose source segmentation is completely
empty. The scanner incorrectly dereferenced a missing lesion-support audit for
those cases and reported `TypeError: 'NoneType' object is not subscriptable`.

The corrected scanner handles the two valid states explicitly:

- Nonempty source tumor: require a nonempty binary model-space tumor and a
  zero-outside-voxel support audit.
- Empty source tumor: require empty source/model labels, zero source/model
  tumor voxels, no lesion-support audit, and record
  `not_applicable_empty_source` with zero outside voxels.

The VAE patch sampler already falls back to brain-centered sampling when the
tumor union is empty. No source NIfTI, shared `synthesis` code, pretrained
weights, trainer logic, or previous evidence is modified.

Execution remains fail-closed: static preflight, one 823-case CPU scan, strict
scan validation, and only then one fresh single-A100 training submission.
