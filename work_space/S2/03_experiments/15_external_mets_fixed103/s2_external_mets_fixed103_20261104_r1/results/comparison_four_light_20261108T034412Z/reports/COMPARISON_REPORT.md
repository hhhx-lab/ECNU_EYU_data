# Same-fixed103 four-model descriptive MET comparison

103 validation scans / 69 patient groups. Not locked test104, external held-out validation, or official ranking.
BraTS-evaluation0.0.8 / panoptica2.1.0 / NumPy1.26.4; mets, 27mm^3, parser DSC0.2. TC/WT exclude RC.
Bio: complete A5+B2 author ensemble/postprocessing, inference4319113. Original scores: CPU4319117.
Original 216 mean and 216 paired rows/CI are reused unchanged; only Bio is newly evaluated.
NA is not zero. Paired comparisons use common-finite scans and 20000 patient-group resamples, seed20260831.
K1 used this development cohort. EXT2025 public training overlap and Bio training exposure remain disclosed.
Intervals are descriptive, not formal superiority. Standard-deviation columns have no quality direction.

## ET

| Metric | EXT2025 | K1_175 | original_E_200 | EXT2026_BIO_INTEL_FULL |
|---|---|---|---|---|
| all_instance_tp_et | 4.407767 (n=103) | 4.000000 (n=103) | 3.951456 (n=103) | 4.077670 (n=103) |
| all_instance_fp_et | 0.679612 (n=103) | 1.038835 (n=103) | 1.000000 (n=103) | 0.504854 (n=103) |
| all_instance_fn_et | 3.242718 (n=103) | 3.650485 (n=103) | 3.699029 (n=103) | 3.572816 (n=103) |
| all_instance_f1_et | 0.760803 (n=93) | 0.708062 (n=93) | 0.693078 (n=96) | 0.714412 (n=97) |
| large_instance_tp_et | 3.407767 (n=103) | 3.165049 (n=103) | 3.106796 (n=103) | 3.223301 (n=103) |
| large_instance_fp_et | 0.679612 (n=103) | 1.038835 (n=103) | 1.000000 (n=103) | 0.504854 (n=103) |
| large_instance_fn_et | 0.631068 (n=103) | 0.873786 (n=103) | 0.932039 (n=103) | 0.815534 (n=103) |
| large_instance_f1_et | 0.736558 (n=103) | 0.686988 (n=103) | 0.675187 (n=103) | 0.728255 (n=103) |
| lesionwise_dsc_mean_et | 0.706513 (n=85) | 0.637271 (n=85) | 0.611652 (n=85) | 0.683729 (n=85) |
| lesionwise_dsc_std_et | 0.154062 (n=85) | 0.173264 (n=85) | 0.187651 (n=85) | 0.165375 (n=85) |
| lesionwise_hd95_mean_et | 57.568045 (n=85) | 80.014328 (n=85) | 85.214726 (n=85) | 64.615406 (n=85) |
| lesionwise_hd95_std_et | 62.993116 (n=85) | 78.008712 (n=85) | 82.964003 (n=85) | 69.647423 (n=85) |
| lesionwise_nsd_mean_et | 0.782222 (n=85) | 0.701794 (n=85) | 0.679297 (n=85) | 0.753202 (n=85) |
| lesionwise_nsd_std_et | 0.162191 (n=85) | 0.193215 (n=85) | 0.207529 (n=85) | 0.180315 (n=85) |
| small_instance_tp_et | 1.754717 (n=53) | 1.339623 (n=53) | 1.433962 (n=53) | 1.603774 (n=53) |
| small_instance_fn_et | 5.264151 (n=53) | 5.679245 (n=53) | 5.584906 (n=53) | 5.415094 (n=53) |
| small_instance_fp_et | 0.943396 (n=53) | 1.452830 (n=53) | 1.320755 (n=53) | 0.641509 (n=53) |
| small_instance_f1_et | 0.363854 (n=53) | 0.285790 (n=53) | 0.300882 (n=53) | 0.334818 (n=53) |

## RC

| Metric | EXT2025 | K1_175 | original_E_200 | EXT2026_BIO_INTEL_FULL |
|---|---|---|---|---|
| all_instance_tp_rc | 0.165049 (n=103) | 0.077670 (n=103) | 0.077670 (n=103) | 0.155340 (n=103) |
| all_instance_fp_rc | 0.019417 (n=103) | 0.048544 (n=103) | 0.038835 (n=103) | 0.310680 (n=103) |
| all_instance_fn_rc | 0.038835 (n=103) | 0.126214 (n=103) | 0.126214 (n=103) | 0.048544 (n=103) |
| all_instance_f1_rc | 0.803571 (n=16) | 0.341667 (n=20) | 0.350877 (n=19) | 0.387879 (n=33) |
| large_instance_tp_rc | 0.145631 (n=103) | 0.077670 (n=103) | 0.077670 (n=103) | 0.145631 (n=103) |
| large_instance_fp_rc | 0.019417 (n=103) | 0.048544 (n=103) | 0.038835 (n=103) | 0.310680 (n=103) |
| large_instance_fn_rc | 0.019417 (n=103) | 0.087379 (n=103) | 0.087379 (n=103) | 0.019417 (n=103) |
| large_instance_f1_rc | 0.126214 (n=103) | 0.069579 (n=103) | 0.067961 (n=103) | 0.129450 (n=103) |
| lesionwise_dsc_mean_rc | 0.698966 (n=15) | 0.322970 (n=15) | 0.311187 (n=15) | 0.710364 (n=15) |
| lesionwise_dsc_std_rc | 0.080191 (n=15) | 0.040821 (n=15) | 0.072966 (n=15) | 0.041963 (n=15) |
| lesionwise_hd95_mean_rc | 63.751272 (n=15) | 205.558279 (n=15) | 213.655565 (n=15) | 51.565047 (n=15) |
| lesionwise_hd95_std_rc | 37.123962 (n=15) | 23.845079 (n=15) | 36.980913 (n=15) | 24.639431 (n=15) |
| lesionwise_nsd_mean_rc | 0.602988 (n=15) | 0.230073 (n=15) | 0.215776 (n=15) | 0.618594 (n=15) |
| lesionwise_nsd_std_rc | 0.088361 (n=15) | 0.037975 (n=15) | 0.066452 (n=15) | 0.054493 (n=15) |
| small_instance_tp_rc | 0.666667 (n=3) | 0.000000 (n=3) | 0.000000 (n=3) | 0.333333 (n=3) |
| small_instance_fn_rc | 0.666667 (n=3) | 1.333333 (n=3) | 1.333333 (n=3) | 1.000000 (n=3) |
| small_instance_fp_rc | 0.333333 (n=3) | 0.000000 (n=3) | 0.000000 (n=3) | 0.000000 (n=3) |
| small_instance_f1_rc | 0.266667 (n=3) | 0.000000 (n=3) | 0.000000 (n=3) | 0.222222 (n=3) |

## TC

| Metric | EXT2025 | K1_175 | original_E_200 | EXT2026_BIO_INTEL_FULL |
|---|---|---|---|---|
| all_instance_tp_tc | 4.398058 (n=103) | 4.000000 (n=103) | 3.951456 (n=103) | 4.087379 (n=103) |
| all_instance_fp_tc | 0.563107 (n=103) | 0.582524 (n=103) | 0.611650 (n=103) | 0.417476 (n=103) |
| all_instance_fn_tc | 2.912621 (n=103) | 3.310680 (n=103) | 3.359223 (n=103) | 3.223301 (n=103) |
| all_instance_f1_tc | 0.784534 (n=93) | 0.746874 (n=93) | 0.726471 (n=96) | 0.744788 (n=97) |
| large_instance_tp_tc | 3.407767 (n=103) | 3.174757 (n=103) | 3.106796 (n=103) | 3.223301 (n=103) |
| large_instance_fp_tc | 0.563107 (n=103) | 0.582524 (n=103) | 0.611650 (n=103) | 0.417476 (n=103) |
| large_instance_fn_tc | 0.640777 (n=103) | 0.873786 (n=103) | 0.941748 (n=103) | 0.825243 (n=103) |
| large_instance_f1_tc | 0.752131 (n=103) | 0.719306 (n=103) | 0.703130 (n=103) | 0.745800 (n=103) |
| lesionwise_dsc_mean_tc | 0.747662 (n=86) | 0.688071 (n=86) | 0.657715 (n=86) | 0.730746 (n=86) |
| lesionwise_dsc_std_tc | 0.147662 (n=86) | 0.151838 (n=86) | 0.169134 (n=86) | 0.149945 (n=86) |
| lesionwise_hd95_mean_tc | 52.158491 (n=86) | 63.522610 (n=86) | 72.112886 (n=86) | 56.489442 (n=86) |
| lesionwise_hd95_std_tc | 55.335306 (n=86) | 61.815358 (n=86) | 70.068299 (n=86) | 58.229920 (n=86) |
| lesionwise_nsd_mean_tc | 0.797666 (n=86) | 0.733740 (n=86) | 0.707341 (n=86) | 0.773819 (n=86) |
| lesionwise_nsd_std_tc | 0.149817 (n=86) | 0.163365 (n=86) | 0.180519 (n=86) | 0.160063 (n=86) |
| small_instance_tp_tc | 1.877551 (n=49) | 1.428571 (n=49) | 1.551020 (n=49) | 1.734694 (n=49) |
| small_instance_fn_tc | 4.979592 (n=49) | 5.428571 (n=49) | 5.306122 (n=49) | 5.122449 (n=49) |
| small_instance_fp_tc | 0.734694 (n=49) | 0.632653 (n=49) | 0.591837 (n=49) | 0.530612 (n=49) |
| small_instance_f1_tc | 0.399909 (n=49) | 0.312552 (n=49) | 0.329684 (n=49) | 0.373424 (n=49) |

## WT

| Metric | EXT2025 | K1_175 | original_E_200 | EXT2026_BIO_INTEL_FULL |
|---|---|---|---|---|
| all_instance_tp_wt | 3.922330 (n=103) | 3.592233 (n=103) | 3.563107 (n=103) | 3.699029 (n=103) |
| all_instance_fp_wt | 0.621359 (n=103) | 0.631068 (n=103) | 0.660194 (n=103) | 0.495146 (n=103) |
| all_instance_fn_wt | 2.796117 (n=103) | 3.126214 (n=103) | 3.155340 (n=103) | 3.019417 (n=103) |
| all_instance_f1_wt | 0.784637 (n=93) | 0.748435 (n=93) | 0.728257 (n=96) | 0.745052 (n=97) |
| large_instance_tp_wt | 3.203883 (n=103) | 2.961165 (n=103) | 2.893204 (n=103) | 3.067961 (n=103) |
| large_instance_fp_wt | 0.621359 (n=103) | 0.631068 (n=103) | 0.660194 (n=103) | 0.495146 (n=103) |
| large_instance_fn_wt | 0.737864 (n=103) | 0.980583 (n=103) | 1.048544 (n=103) | 0.873786 (n=103) |
| large_instance_f1_wt | 0.743912 (n=103) | 0.701538 (n=103) | 0.688472 (n=103) | 0.736069 (n=103) |
| lesionwise_dsc_mean_wt | 0.703347 (n=87) | 0.630263 (n=87) | 0.617613 (n=87) | 0.685620 (n=87) |
| lesionwise_dsc_std_wt | 0.167978 (n=87) | 0.178525 (n=87) | 0.185295 (n=87) | 0.181302 (n=87) |
| lesionwise_hd95_mean_wt | 62.144908 (n=87) | 77.955327 (n=87) | 85.047330 (n=87) | 70.518561 (n=87) |
| lesionwise_hd95_std_wt | 62.626935 (n=87) | 74.871512 (n=87) | 80.766502 (n=87) | 73.845402 (n=87) |
| lesionwise_nsd_mean_wt | 0.706655 (n=87) | 0.627997 (n=87) | 0.613113 (n=87) | 0.677676 (n=87) |
| lesionwise_nsd_std_wt | 0.158565 (n=87) | 0.182674 (n=87) | 0.188447 (n=87) | 0.178688 (n=87) |
| small_instance_tp_wt | 1.413043 (n=46) | 1.000000 (n=46) | 1.130435 (n=46) | 1.304348 (n=46) |
| small_instance_fn_wt | 4.804348 (n=46) | 5.217391 (n=46) | 5.086957 (n=46) | 4.913043 (n=46) |
| small_instance_fp_wt | 0.717391 (n=46) | 1.000000 (n=46) | 0.739130 (n=46) | 0.630435 (n=46) |
| small_instance_f1_wt | 0.338859 (n=46) | 0.261962 (n=46) | 0.294659 (n=46) | 0.327950 (n=46) |

