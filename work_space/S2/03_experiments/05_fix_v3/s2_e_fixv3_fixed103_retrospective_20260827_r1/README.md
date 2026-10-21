# E vs FixV3 fixed-103 retrospective paired comparison

- Local timestamp: `2026-08-27T14:55:51+0800`
- UTC timestamp: `2026-08-27T06:55:44Z`
- Artifact status: `experimental_unvalidated`
- Operator approved: `false`
- Formal gate status: `not_run_not_passed`
- Cohort: the same 103 fixed-validation cases, authentic T2W
- Bootstrap: 20,000 paired resamples, seed `20260824`
- Missing-value policy: paired complete cases per metric

## Inputs

| Method | Checkpoint SHA256 | Metrics CSV SHA256 |
|---|---|---|
| E | `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267` | `294a1451a8737344a929902c8ef118c89bd9a3da971952d11b453fafccc3e167` |
| FixV3 | `b8b2d13a6268231f73d43c37cb097d1be3daeec654b9bba6ddd24f410cf7b27e` | `19cec61da53dfe2e64b25807fb0bece94688c116b8e78d0ab1e081ecc59f8553` |

The E, FixV3, and frozen cohort manifest case IDs are identical in count, set, and order. Both metric CSVs have the same 73-column schema. Both evaluations use BraTS-evaluation 0.0.8, Panoptica 2.1.0, NumPy 1.26.4, configuration `mets`, volume threshold 27 voxels, and overlap threshold 0.2.

## Primary endpoints

`Paired benefit` is direction-normalized so positive values favor FixV3. For HD95, positive means FixV3 is lower. Means are the evaluator aggregate rows; paired estimates use only finite matched pairs. W/T/L is FixV3 wins/ties/E wins.

| Metric | E mean | FixV3 mean | Paired benefit [95% CI] | p | Paired n | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| `all_instance_f1_et` | 0.693078 | 0.697869 | -0.009956 [-0.044306, +0.019882] | 0.5637 | 94 | 17/55/22 |
| `large_instance_f1_et` | 0.675187 | 0.679233 | +0.004046 [-0.024795, +0.030765] | 0.7391 | 103 | 16/75/12 |
| `lesionwise_dsc_mean_et` | 0.611652 | 0.627825 | +0.016173 [-0.006899, +0.041839] | 0.1845 | 85 | 47/1/37 |
| `lesionwise_hd95_mean_et` | 85.214726 | 80.813170 | +4.401555 [-9.101931, +17.683222] | 0.5013 | 85 | 28/31/26 |
| `lesionwise_nsd_mean_et` | 0.679297 | 0.701448 | +0.022151 [-0.001427, +0.048928] | 0.0681 | 85 | 47/7/31 |
| `small_instance_f1_et` | 0.300882 | 0.291139 | -0.009743 [-0.075372, +0.051352] | 0.7813 | 53 | 10/35/8 |
| `all_instance_f1_rc` | 0.350877 | 0.368421 | +0.017544 [-0.000000, +0.052632] | 0.7158 | 19 | 1/18/0 |
| `large_instance_f1_rc` | 0.067961 | 0.071197 | +0.003236 [-0.000000, +0.009709] | 0.7145 | 103 | 1/102/0 |
| `lesionwise_dsc_mean_rc` | 0.311187 | 0.342694 | +0.031506 [-0.002716, +0.088841] | 0.1351 | 15 | 4/7/4 |
| `lesionwise_hd95_mean_rc` | 213.655565 | 201.065233 | +12.590331 [-0.099036, +37.473442] | 0.1682 | 15 | 4/8/3 |
| `lesionwise_nsd_mean_rc` | 0.215776 | 0.239837 | +0.024061 [-0.003910, +0.068337] | 0.1715 | 15 | 4/7/4 |
| `small_instance_f1_rc` | 0.000000 | 0.000000 | 0.000000 [0.000000, 0.000000] | 1.0000 | 3 | 0/3/0 |
| `all_instance_f1_tc` | 0.726471 | 0.734354 | -0.007573 [-0.043592, +0.024482] | 0.6827 | 94 | 19/57/18 |
| `large_instance_f1_tc` | 0.703130 | 0.710678 | +0.007548 [-0.023342, +0.035841] | 0.5881 | 103 | 18/77/8 |
| `lesionwise_dsc_mean_tc` | 0.657715 | 0.676963 | +0.019248 [-0.012976, +0.050947] | 0.2264 | 86 | 55/2/29 |
| `lesionwise_hd95_mean_tc` | 72.112886 | 66.082755 | +6.030131 [-9.871971, +20.990240] | 0.4342 | 86 | 36/28/22 |
| `lesionwise_nsd_mean_tc` | 0.707341 | 0.731887 | +0.024546 [-0.007170, +0.055928] | 0.1259 | 86 | 50/9/27 |
| `small_instance_f1_tc` | 0.329684 | 0.329793 | +0.000109 [-0.072790, +0.072349] | 0.9917 | 49 | 10/31/8 |
| `all_instance_f1_wt` | 0.728257 | 0.723136 | -0.020616 [-0.053877, +0.008010] | 0.1700 | 94 | 19/49/26 |
| `large_instance_f1_wt` | 0.688472 | 0.677012 | -0.011460 [-0.046190, +0.023403] | 0.5222 | 103 | 19/65/19 |
| `lesionwise_dsc_mean_wt` | 0.617613 | 0.611382 | -0.006231 [-0.038223, +0.025679] | 0.7104 | 87 | 46/1/40 |
| `lesionwise_hd95_mean_wt` | 85.047330 | 88.863248 | -3.815918 [-19.158595, +12.276538] | 0.6272 | 87 | 35/18/34 |
| `lesionwise_nsd_mean_wt` | 0.613113 | 0.616464 | +0.003352 [-0.026882, +0.033992] | 0.8331 | 87 | 48/2/37 |
| `small_instance_f1_wt` | 0.294659 | 0.271997 | -0.022662 [-0.090243, +0.038288] | 0.4857 | 46 | 7/28/11 |

## Interpretation

- By aggregate mean, FixV3 is better on 17/24 primary endpoints, E is better on 6/24, and 1/24 is tied.
- FixV3 shows a coherent favorable trend in ET and TC lesion-wise DSC, HD95, and NSD. E retains favorable WT and some small-instance results.
- No primary endpoint has a paired 95% bootstrap CI excluding zero; none establishes statistical superiority.
- This is a retrospective cross-run comparison. E was evaluated in the July frozen `true1mm` run, whereas FixV3 was inferred in the August endpoint pipeline. Although cohort IDs, metric schema, package versions, and evaluation thresholds match, this artifact does not establish byte-identical inference runtime or reference materialization across runs.
- A publication-grade same-pipeline comparison should rerun E once on the already frozen current fixed-103 real-T2W inputs, then compare that result with the existing FixV3 result. FixV3 does not need to be rerun.

## Files

- `PAIRWISE_BOOTSTRAP.csv`: all 72 paired metrics.
- `METRIC_RANKINGS.csv`: aggregate metric rankings.
- `PAIRED_CASE_METRICS.csv`: case-level E and FixV3 metrics.
- `PAIRED_COMPARISON.json`: machine-readable comparison and source hashes.

