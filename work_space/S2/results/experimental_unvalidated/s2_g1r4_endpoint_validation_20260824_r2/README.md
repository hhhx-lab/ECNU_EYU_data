# S2 G1-r4 Endpoint Validation (2026-08-24 r2)

This directory freezes the experimental validation chain requested on 2026-08-24.
It is not an official challenge result and must remain:

- `artifact_status=experimental_unvalidated`
- `operator_approved=false`
- `formal_gate_status=not_run_not_passed`

This is the independent r2 attempt. The r1 remote static preflight stopped
before any Slurm submission because three shared G1 source files differed from
the locally frozen bytes. Its failure evidence is retained in the r1 root.
r2 bundles the historical remote G1 runtime whose file modification times all
precede r4 job 3391396; jobs import this bundle and do not import the mutable
shared G1 source tree.

## Frozen questions

1. On 27 internal missing-T2W validation cases, how do R, B, E, and Fix-v3
   full-200 perform after the same G1 r4 Ensemble completion?
2. On the same 103 authentic-T2W validation cases, how does one frozen Fix-v3
   checkpoint perform with authentic T2W versus r4 Ensemble synthesized T2W?
3. After all contracts are frozen, what is the final Fix-v3 endpoint result on
   the 26 missing-T2W locked-test cases?

The 27-case cohort has no authentic T2W counterfactual. It supports an
end-to-end endpoint evaluation only. The 103-case paired experiment is the
direct test of whether replacing authentic T2W with synthesized T2W changes
segmentation performance.

## Order and gates

1. `val27_r4_synthesis`: synthesize 27 T2W volumes and validate native geometry.
2. `val27_four_models`: run R, B, E, and Fix-v3 on the identical 27 inputs.
3. `fixed103_real_vs_synthetic`: run Fix-v3 twice on the identical 103 cases.
4. `test26_locked_endpoint`: only after the first three stages pass, synthesize
   and evaluate the 26 locked-test cases with Fix-v3.

No result from the 27 or 103 cases may alter checkpoints, preprocessing,
thresholds, model choice, or the locked-test pipeline.

## Interpretation boundary

G1 reconstruction quality, offline completion plus segmentation, and online
Diffusion augmentation plus segmentation are separate evidence layers. Image
reconstruction rank cannot be presented as segmentation benefit. Reference
segmentations are used by the already-frozen r4 offline spatial contract to
audit and retain lesion support, but they are never model input channels; this
offline-evaluation limitation must be disclosed in reporting.
