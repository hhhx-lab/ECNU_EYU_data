# S2 MET-AUG B-F Screen125 Retrieval Acceptance

- Local retrieval timestamp: `2026-09-03T11:10:07+08:00`
- UTC retrieval timestamp: `2026-09-03T03:10:07Z`
- Remote snapshot timestamp: `2026-09-03T02:57:35Z`
- Source experiment: `s2_met_aug_btof_screen125_20260831_r1`
- Status: `experimental_unvalidated`
- Operator approved: `false`
- Formal gate: `not_run_not_passed`

## Retrieved scope

The snapshot contains the plan, route configurations, all training launch and
validation evidence, complete stdout/stderr logs, and the five automatic
nnUNet validation summaries. It contains 46 remotely sourced files (about
1.6 MB) and excludes checkpoints, NIfTI volumes, and `met_aug_events.jsonl`.

All five training jobs completed with exit code `0:0`. Each validation contains
103 cases, all five `TRAINING_VALIDATION` records report `status=pass`, and the
retrieved logs contain no OOM, traceback, CUDA error, nonfinite, ImportError,
exception, or failure match.

## Epoch-125 automatic validation

| Rank | Route | Foreground mean Dice | Label 1 | Label 2 | Label 3 | Label 4 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | F | 0.551289 | 0.494862 | 0.732421 | 0.695929 | 0.281944 |
| 2 | C40 | 0.550523 | 0.493358 | 0.733811 | 0.691774 | 0.283151 |
| 3 | E-SNFH | 0.550495 | 0.483941 | 0.734340 | 0.698301 | 0.285397 |
| 4 | B | 0.548575 | 0.483764 | 0.735089 | 0.690339 | 0.285108 |
| 5 | D | 0.545791 | 0.475818 | 0.736598 | 0.688381 | 0.282366 |

The absolute spread from first to fifth is only `0.005498` foreground Dice.
This table therefore does not establish a meaningful winner by itself.

## Pre-registered checkpoint selection

| Route | Selected epoch | Selection EMA foreground Dice |
|---|---:|---:|
| D | 50 | 0.759616 |
| E-SNFH | 125 | 0.754508 |
| B | 125 | 0.750373 |
| C40 | 50 | 0.749485 |
| F | 125 | 0.749182 |

## Interpretation boundary

The first table compares all routes at epoch 125 using nnUNet's automatic
voxel-level validation output. It is not the pre-registered selected-checkpoint
official MET paired comparison. In particular, C40 and D must use epoch 50 in
that comparison. Official lesion-level MET results, paired wins/ties/losses,
20,000-bootstrap confidence intervals, p-values, and false-positive protection
statistics are not yet available and must not be inferred from this snapshot.
