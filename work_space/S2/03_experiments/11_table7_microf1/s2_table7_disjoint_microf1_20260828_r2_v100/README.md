# Table 7 mutually exclusive size-stratified micro-F1 (V100 migration)

Status: `experimental_unvalidated`

This immutable ECNU r2 run regenerates the 104 locked-test predictions from the
frozen completion-plus-focal checkpoint used for Tables 6 and 7. It first
reproduces the published BraTS-evaluation parser output, then computes a new
lesion-instance micro-F1 whose size strata are mutually exclusive.

The r1 A100 job was held while pending because all accessible A100 GPUs were
allocated and its scheduler estimate was several days away. This r2 run uses
one immediately available Tesla V100-PCIE-32GB. The data, checkpoint,
inference code, and evaluation contract are unchanged; the hardware change is
recorded and the published three-decimal Table 7 values remain a mandatory
reproduction gate.

The new metric uses 26-connected components and one maximum-cardinality
one-to-one matching per case and region over component pairs with DSC >= 0.2.
Each accepted pair is a TP assigned by reference volume. Each unmatched
reference is an FN assigned by reference volume, and each unmatched prediction
is an FP assigned by prediction volume. Counts are pooled over all 104 cases.
For each threshold T, small means volume < T and large means volume >= T.

Both T=27 and T=50 voxels are reported. T=50 matches the manuscript Table 7
size definition; T=27 preserves the later audited challenge-analysis contract.

The run is not a formal challenge gate and must retain:

- `artifact_status=experimental_unvalidated`
- `operator_approved=false`
- `formal_gate_status=not_run_not_passed`
