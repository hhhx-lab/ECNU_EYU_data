# K1/K2/K4 175-epoch training submission acceptance (2026-09-29)

## Scope and status

The user explicitly authorized K1, K2, and K4 training after reporting completion of the manual review. This is engineering execution authorization, not formal scientific approval. The original smoke CSV files remain unchanged. K2 uses the preregistered shared RC24 technical smoke (K1 R3J, automatic 24/24) plus its own frozen D empirical contract; no K2-specific 24-montage review is claimed. K4 uses its independent small8 24/24 automatic smoke and prior user authorization.

All three jobs use the original E checkpoint SHA256 `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`, 1035 train / 103 fixed development-validation cases, seed `20260724`, one A100, no DDP or torch compile, 175 epochs, and checkpoints at 25/50/75/100/125/150/175. X2 in K2 retains the original epoch-83-to-125 decay and stays zero through 175; the epoch extension does not silently re-register that schedule.

| Route | Isolated attempt | Job | Last observed status | GPU node |
| --- | --- | ---: | --- | --- |
| K1 | `s2_met_aug_k1245_20260929_k1_175_r3` | `3827543` | `RUNNING 0:0`, STARTED, no FAILED | gpu18 |
| K2 | `s2_met_aug_k1245_20260929_k2_175_r2` | `3827555` | `RUNNING 0:0`, STARTED, no FAILED | gpu19 |
| K4 | `s2_met_aug_k1245_20260929_k4_175_r6` | `3827580` | `RUNNING 0:0`, STARTED, no FAILED | gpu19 |

Each route passed local `bash -n`, plan/auth/path checks, remote checkpoint and calibration SHA checks, full manifest and route-config loading, trainer import, a segmamba NumPy/Torch import with `MKL_THREADING_LAYER=GNU`, Slurm resource checks, and a unique held launch audit followed by release. The submission-state snapshot was taken at local UTC `2026-09-29T02:55:01.502496Z` and remote UTC `2026-09-29T02:44:22.410008Z` on `login1`: [snapshot](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/readonly_train175_mkl_recovery_20260929_20260929T025501Z/FINAL_READONLY_SNAPSHOT.json). A second one-shot [runtime-start snapshot](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/readonly_train175_runtime_start_20260929_20260929T025820Z/RUNTIME_START_SNAPSHOT.json) at local UTC `02:58:20Z` / remote UTC `02:47:40Z` still showed all three `RUNNING 0:0`, STARTED, no FAILED; stdout was initializing G1/nnU-Net, but none had reached `MET_AUG_K1245_175_RUNTIME_PASS` or an Epoch line. The VPN started for each operation was closed afterwards.

## Preserved failures

- K1 `20260925_k1_175_r1`: deployment timed out before `sbatch`; the partial root and local failure audit are preserved. No job ID.
- K4 `20260929_k4_175_r4`: static import found that the R5 smoke-side small8 manifest lacked its referenced `components.jsonl` and target-group file. No job ID. The new K4 attempt binds the complete R2 frozen small8 pool with the same manifest identity (`956e6384...e70`); the old R5 smoke evidence is unchanged.
- K1 `3827458`, K2 `3827469`, and K4 `3827503` were accepted by Slurm but ended `FAILED 1:0` before STARTED because `MKL_THREADING_LAYER=INTEL` conflicted with `libgomp`. The independent recovery Slurm files explicitly export `MKL_THREADING_LAYER=GNU`; all three old FAILED records and logs remain intact.
- Earlier K4 `3741523` calibration-variable failure and the R3 QOS rejection also remain intact; neither was reused.

This document accepts **Slurm submission and process startup only**, not entry into the training epoch loop. It does not accept 175-epoch completion, checkpoint selection, official MET, paired comparison, or superiority. K1's strict preregistered F gate remains `false` and fallback eligibility is separate. K5 is not authorized. The same fixed103 set will select checkpoints and report metrics, so any later comparison is exploratory. All new artifacts remain `artifact_status=experimental_unvalidated`, `operator_approved=false`, and `formal_gate_status=not_run_not_passed`.
