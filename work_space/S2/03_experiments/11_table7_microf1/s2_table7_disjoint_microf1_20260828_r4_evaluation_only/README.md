# Table 7 disjoint micro-F1: evaluation-only r4

Status: `experimental_unvalidated`

This immutable r4 run corrects the case-name pairing failure observed in the
r2 V100 run. The r2 inference completed successfully for all 104 locked-test
cases, but the official evaluator interpreted six-digit nnU-Net IDs such as
`BraTSMET_001248` as the five-digit suffix `00124`. That caused missing and
cross-case matches, including geometry mismatches.

r4 performs no training and no inference. It reads the frozen r2 predictions,
maps each nnU-Net ID through `nnunet_case_mapping_master.csv`, and creates two
read-only symlink views whose filenames use the canonical source IDs, for
example `BraTS-MET-01333-002.nii.gz`. Prediction and reference use the same
104-entry bijection. The mapping, symlink targets, geometry, and source-file
hashes are audited before evaluation.

The official BraTS evaluator is rerun at volume thresholds 27 and 50. The
independent mutually exclusive micro-F1 calculation uses 3D 26-connectivity,
one-to-one maximum-cardinality matching at DSC >= 0.2, reference volume for TP
and FN size, prediction volume for FP size, and pooled TP/FP/FN over all 104
cases. For each threshold, `all = small + large` is required separately for
TP, FP, and FN.

r2 is read-only and remains preserved as failure evidence. The r3 static
preflight failure caused by reading the job ID from the wrong evidence file is
also preserved; r3 submitted no job. All r4 artifacts
retain:

- `artifact_status=experimental_unvalidated`
- `operator_approved=false`
- `formal_gate_status=not_run_not_passed`
