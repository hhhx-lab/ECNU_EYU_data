# S2 fixed-103 official-metric model selection

- Status: `pass`
- Selected model: `E`
- Selected checkpoint SHA256: `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`
- Eligible models: original `E` and `E_continue_final`; `B` is a comparator only.
- Tiny-lesion metrics: unavailable in the locked official evaluator output.
- Empty official cells remain undefined and were never replaced with zero.

## E-continue minus E

| Region | DSC mean delta | NSD mean delta | Official all-F1 delta | Common-defined paired all-F1 delta | FP delta | FN delta |
|---|---:|---:|---:|---:|---:|---:|
| ET | +0.002249 | +0.001511 | -0.002682 | -0.017428 | +15 | -3 |
| RC | +0.019216 | +0.020636 | +0.017544 | +0.017544 | -1 | +0 |
| TC | -0.001624 | -0.004024 | -0.004694 | -0.020151 | +4 | -2 |
| WT | -0.005249 | -0.005335 | -0.005924 | -0.021419 | +7 | +1 |

## Decision

Original E is retained. E-continue does not satisfy the conservative fallback rule: 
its sparse RC gain is not a broad, statistically robust improvement, while ET/TC/WT 
show false-positive increases and TC/WT primary-metric regressions.

The original evaluation outputs and launch evidence were not modified. The malformed 
candidate map in the preserved launch audit is recorded as a provenance-format warning; 
all three runs were independently rebound through stopped PID files, pass logs, exact 
result SHA256 values, completion markers, and 103-case input hardlinks.
