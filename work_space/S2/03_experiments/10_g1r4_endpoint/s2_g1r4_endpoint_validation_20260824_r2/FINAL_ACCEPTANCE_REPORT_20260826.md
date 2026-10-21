# S2 G1-r4 Endpoint Validation Final Acceptance Report

- Local acceptance generated: `2026-08-26T21:21:30+08:00`
- UTC acceptance generated: `2026-08-26T13:21:30+00:00`
- Final remote snapshot: `2026-08-26T13:06:30Z` on `login1`
- Overall result: **PASS (experimental, unvalidated)**
- `operator_approved=false`; `formal_gate_status=not_run_not_passed`
- No model training or tuning occurred in this pipeline.

## Frozen contract

- Fix-v3 full-200 checkpoint SHA256: `b8b2d13a6268231f73d43c37cb097d1be3daeec654b9bba6ddd24f410cf7b27e`
- Spatial contract: `foreground_centered_isotropic_resample_v1`.
- Official evaluator: BraTS-evaluation `0.0.8`, panoptica `2.1.0`, NumPy `1.26.4`.
- Paired bootstrap: 20,000 replicates, seed `20260824`.
- Cohorts are pairwise disjoint: val27 / fixed103 / locked-test26.

Model labels used below:

- R: real-only reference nnU-Net.
- B: completion-trained nnU-Net with the standard loss.
- E: completion-trained nnU-Net with the focal-loss configuration.
- FixV3: focal nnU-Net trained with online Diffusion augmentation, fresh 0->200 epochs.

## Stage acceptance

| Stage | Job | Cases | Completed UTC | Acceptance |
|---|---:|---:|---|---|
| 27-case r4 T2W synthesis | `3530868` | 27 | `2026-08-24T08:52:38.017154+00:00` | PASS |
| 27-case R/B/E/FixV3 endpoint | `3535754` | 27 | `2026-08-26T10:11:14.809598+00:00` | PASS |
| 103-case real vs synthetic T2W | `3536323` | 103 | `2026-08-26T12:37:14.535560+00:00` | PASS |
| 26-case locked endpoint | `3536432` | 26 | `2026-08-26T13:00:01.242464+00:00` | PASS |

- Val27 synthesis: 27 NIfTI generated remotely, 27 spatial rows, foreground/lesion outside = 0, geometry mismatch/repaired = 0, source T2W reads = 0.
- Val27 segmentation: 4 frozen models x 27 cases = 108 predictions; 72 official metrics; all 6 model pairs generated with the frozen bootstrap contract.
- Fixed103 paired: 103 real-T2W and 103 r4-synthetic-T2W predictions from the identical Fix-v3 checkpoint; IDs, labels, and non-T2W channels match.
- Locked-test26: 26 synthesized T2W and 26 Fix-v3 predictions; geometry mismatch/repaired = 0 and source T2W reads = 0.
- Local pull: lightweight evidence only; zero NIfTI files and no model snapshots.

## Val27 four-model means

Among 20 rankable primary endpoints, B and FixV3 each have 8 unique best means; E has 2 and R has 0. Tied best counts are R=2, B=10, E=4, FixV3=9. This is not a predeclared composite score.

| Metric | Direction | R | B | E | FixV3 | Best |
|---|---|---:|---:|---:|---:|---|
| `all_instance_f1_et` | higher_is_better | 0.836006 | 0.876114 | 0.862965 | 0.855017 | B |
| `all_instance_f1_rc` | higher_is_better | 0 | 0 | 0 | NA | R=B=E |
| `all_instance_f1_tc` | higher_is_better | 0.836631 | 0.876134 | 0.863589 | 0.856289 | B |
| `all_instance_f1_wt` | higher_is_better | 0.815455 | 0.846893 | 0.835555 | 0.82058 | B |
| `large_instance_f1_et` | higher_is_better | 0.8697 | 0.911593 | 0.904007 | 0.913093 | FixV3 |
| `large_instance_f1_rc` | higher_is_better | 0 | 0 | 0 | 0 | R=B=E=FixV3 |
| `large_instance_f1_tc` | higher_is_better | 0.870488 | 0.911391 | 0.904795 | 0.914636 | FixV3 |
| `large_instance_f1_wt` | higher_is_better | 0.847407 | 0.872868 | 0.877921 | 0.870819 | E |
| `lesionwise_dsc_mean_et` | higher_is_better | 0.656916 | 0.696073 | 0.703031 | 0.719547 | FixV3 |
| `lesionwise_dsc_mean_rc` | higher_is_better | NA | NA | NA | NA | NA |
| `lesionwise_dsc_mean_tc` | higher_is_better | 0.664192 | 0.70191 | 0.709748 | 0.727667 | FixV3 |
| `lesionwise_dsc_mean_wt` | higher_is_better | 0.584566 | 0.606516 | 0.611997 | 0.611452 | E |
| `lesionwise_hd95_mean_et` | lower_is_better | 75.123 | 51.3404 | 55.7904 | 49.897 | FixV3 |
| `lesionwise_hd95_mean_rc` | lower_is_better | NA | NA | NA | NA | NA |
| `lesionwise_hd95_mean_tc` | lower_is_better | 74.5484 | 51.3665 | 55.24 | 48.7756 | FixV3 |
| `lesionwise_hd95_mean_wt` | lower_is_better | 87.1775 | 70.0646 | 73.2185 | 72.5979 | B |
| `lesionwise_nsd_mean_et` | higher_is_better | 0.758358 | 0.811643 | 0.810935 | 0.824782 | FixV3 |
| `lesionwise_nsd_mean_rc` | higher_is_better | NA | NA | NA | NA | NA |
| `lesionwise_nsd_mean_tc` | higher_is_better | 0.75993 | 0.811578 | 0.813052 | 0.828194 | FixV3 |
| `lesionwise_nsd_mean_wt` | higher_is_better | 0.659935 | 0.696383 | 0.694805 | 0.6945 | B |
| `small_instance_f1_et` | higher_is_better | 0.454607 | 0.480475 | 0.477204 | 0.419847 | B |
| `small_instance_f1_rc` | higher_is_better | NA | NA | NA | NA | NA |
| `small_instance_f1_tc` | higher_is_better | 0.45831 | 0.482879 | 0.48029 | 0.427254 | B |
| `small_instance_f1_wt` | higher_is_better | 0.346878 | 0.393482 | 0.374677 | 0.333799 | B |

Four RC lesionwise/small-instance primary endpoints are non-finite for all methods and therefore are not rankable. Exact case-level paired results are in `summary/VAL27_PRIMARY_PAIRED_STATISTICS.csv`.

### Val27 pair summary

W/T/L below sums method-level case outcomes across available primary endpoints; it is not a count of unique patients and is not a composite endpoint.

| A vs B | Available primary metrics | CI favors A | CI favors B | W/T/L observations |
|---|---:|---:|---:|---:|
| R vs B | 20 | 0 | 11 | 103/138/249 |
| R vs E | 19 | 0 | 13 | 78/139/269 |
| R vs FixV3 | 19 | 0 | 8 | 101/121/264 |
| B vs E | 19 | 0 | 0 | 132/163/191 |
| B vs FixV3 | 19 | 3 | 2 | 157/157/172 |
| E vs FixV3 | 19 | 0 | 0 | 200/148/138 |

For the directly relevant E versus FixV3 comparison, none of the 19 available primary endpoints has a 95% bootstrap CI excluding zero. The observed endpoint means are mixed; this dataset does not establish a primary-endpoint advantage for online Diffusion augmentation over E.

These endpoint-wise bootstrap results are exploratory and are not adjusted for multiple comparisons.

## Fixed103 real versus synthesized T2W

Positive benefit means favor real T2W after accounting for metric direction; negative benefit means favor synthesized T2W.

| Metric | Real mean | Synthetic mean | Benefit real over synthetic | 95% CI | Bootstrap p | W/T/L | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_instance_f1_et` | 0.697869 | 0.700446 | 0.00492672 | [-0.0206265, 0.0361029] | 0.7759 | 13/58/22 | 93 |
| `large_instance_f1_et` | 0.679233 | 0.683127 | -0.00389432 | [-0.0246618, 0.0168714] | 0.707 | 12/72/19 | 103 |
| `lesionwise_dsc_mean_et` | 0.627825 | 0.629538 | -0.00171297 | [-0.0290179, 0.0246256] | 0.9139 | 47/4/34 | 85 |
| `lesionwise_hd95_mean_et` | 80.8132 | 78.4233 | -2.38983 | [-16.5507, 12.5775] | 0.7212 | 29/26/30 | 85 |
| `lesionwise_nsd_mean_et` | 0.701448 | 0.702613 | -0.00116565 | [-0.0318549, 0.0292631] | 0.9388 | 44/7/34 | 85 |
| `small_instance_f1_et` | 0.291139 | 0.253872 | 0.0372668 | [-0.00439796, 0.0864489] | 0.087 | 10/37/6 | 53 |
| `all_instance_f1_rc` | 0.368421 | 0.388889 | 0 | [0, 0] | 1 | 0/18/0 | 18 |
| `large_instance_f1_rc` | 0.0711974 | 0.0711974 | 0 | [0, 0] | 1 | 0/103/0 | 103 |
| `lesionwise_dsc_mean_rc` | 0.342694 | 0.344201 | -0.0015074 | [-0.00748634, 0.00397997] | 0.6223 | 4/7/4 | 15 |
| `lesionwise_hd95_mean_rc` | 201.065 | 201.035 | -0.0306769 | [-0.210332, 0.199404] | 0.7188 | 3/7/5 | 15 |
| `lesionwise_nsd_mean_rc` | 0.239837 | 0.238412 | 0.0014253 | [-0.0122964, 0.0153428] | 0.8453 | 4/7/4 | 15 |
| `small_instance_f1_rc` | 0 | 0 | 0 | [0, 0] | 1 | 0/3/0 | 3 |
| `all_instance_f1_tc` | 0.734354 | 0.734354 | 0.00789668 | [-0.0169587, 0.037614] | 0.6067 | 11/66/16 | 93 |
| `large_instance_f1_tc` | 0.710678 | 0.714665 | -0.00398746 | [-0.0214336, 0.0124053] | 0.6479 | 8/81/14 | 103 |
| `lesionwise_dsc_mean_tc` | 0.676963 | 0.671481 | 0.00548227 | [-0.0188562, 0.0279608] | 0.6167 | 49/5/32 | 86 |
| `lesionwise_hd95_mean_tc` | 66.0828 | 67.5284 | 1.44563 | [-10.6155, 14.794] | 0.8476 | 29/33/24 | 86 |
| `lesionwise_nsd_mean_tc` | 0.731887 | 0.723551 | 0.00833582 | [-0.0181601, 0.0344033] | 0.5234 | 48/9/29 | 86 |
| `small_instance_f1_tc` | 0.329793 | 0.287374 | 0.0424186 | [-0.00821351, 0.102961] | 0.1156 | 9/35/5 | 49 |
| `all_instance_f1_wt` | 0.723136 | 0.725697 | 0.00521414 | [-0.0173683, 0.0346547] | 0.7566 | 16/57/20 | 93 |
| `large_instance_f1_wt` | 0.677012 | 0.679474 | -0.00246251 | [-0.0191604, 0.0137534] | 0.7577 | 15/72/16 | 103 |
| `lesionwise_dsc_mean_wt` | 0.611382 | 0.609312 | 0.00206935 | [-0.0210527, 0.02423] | 0.8386 | 54/2/31 | 87 |
| `lesionwise_hd95_mean_wt` | 88.8632 | 90.7577 | 1.89442 | [-10.5599, 15.7941] | 0.7983 | 36/17/34 | 87 |
| `lesionwise_nsd_mean_wt` | 0.616464 | 0.605507 | 0.0109569 | [-0.0132846, 0.0357446] | 0.3765 | 55/2/30 | 87 |
| `small_instance_f1_wt` | 0.271997 | 0.247975 | 0.024022 | [-0.0169357, 0.0712478] | 0.2746 | 8/33/5 | 46 |

All 72 available official metrics, including all 24 primary endpoints, have 95% bootstrap CIs crossing zero and bootstrap p >= 0.05. Thus this experiment found no statistically detectable segmentation degradation from replacing real T2W with r4 Ensemble T2W under the frozen Fix-v3 model. This is not a formal equivalence or non-inferiority claim.

The bootstrap p-values are endpoint-wise and unadjusted; the absence of significance must not be read as proof that the two inputs are identical.

## Locked-test26 endpoint

| Metric | FixV3 + r4 synthetic T2W mean |
|---|---:|
| `all_instance_f1_et` | 0.897373 |
| `all_instance_f1_rc` | NA |
| `all_instance_f1_tc` | 0.898594 |
| `all_instance_f1_wt` | 0.890613 |
| `large_instance_f1_et` | 0.919542 |
| `large_instance_f1_rc` | 0 |
| `large_instance_f1_tc` | 0.921566 |
| `large_instance_f1_wt` | 0.913581 |
| `lesionwise_dsc_mean_et` | 0.746327 |
| `lesionwise_dsc_mean_rc` | NA |
| `lesionwise_dsc_mean_tc` | 0.75407 |
| `lesionwise_dsc_mean_wt` | 0.697202 |
| `lesionwise_hd95_mean_et` | 45.4779 |
| `lesionwise_hd95_mean_rc` | NA |
| `lesionwise_hd95_mean_tc` | 44.0853 |
| `lesionwise_hd95_mean_wt` | 49.5231 |
| `lesionwise_nsd_mean_et` | 0.84165 |
| `lesionwise_nsd_mean_rc` | NA |
| `lesionwise_nsd_mean_tc` | 0.845413 |
| `lesionwise_nsd_mean_wt` | 0.779128 |
| `small_instance_f1_et` | 0.612325 |
| `small_instance_f1_rc` | NA |
| `small_instance_f1_tc` | 0.616029 |
| `small_instance_f1_wt` | 0.553472 |

The locked-test26 cohort has no authentic-T2W counterfactual, so these are endpoint measurements only and must not be interpreted as a real-versus-synthetic effect.

## Interpretation boundaries

- The 27-case cohort has no authentic T2W counterfactual; it compares frozen segmentation models on the same completed inputs.
- The fixed103 paired experiment isolates T2W replacement under one frozen Fix-v3 checkpoint; it does not compare training strategies.
- The locked-test26 result is a final missing-T2W endpoint evaluation after the validation gates were frozen.
- G1 reconstruction quality, offline completion plus segmentation, and online Diffusion augmentation plus segmentation remain separate evidence layers.
- Reconstruction rank must not be reported as segmentation benefit, and these artifacts are not operator-approved or formally validated.
- Reference segmentations were used only by the frozen offline spatial-support audit, never as model input channels; this limitation must be disclosed.

## Detailed artifacts

- `summary/VAL27_PRIMARY_MEAN_RANKINGS.csv`
- `summary/VAL27_PRIMARY_PAIRED_STATISTICS.csv`
- `summary/FIXED103_PRIMARY_PAIRED_STATISTICS.csv`
- `summary/TEST26_PRIMARY_METRIC_MEANS.csv`
- `cohorts/val27/segmentation/paired_comparison/PAIRWISE_BOOTSTRAP.csv` (all 72 metrics x 6 pairs)
- `cohorts/fixed103/paired_comparison/PAIRWISE_BOOTSTRAP.csv` (all 72 metrics)
- `evidence/` and `logs/` preserve stage-level audit and execution evidence.
