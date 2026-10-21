# S2 Fix-v3 Full-200 Attempt-11

## Transfer record

- Transfer acceptance timestamp: `2026-08-24T04:06:30Z` (`2026-08-24 12:06:30 +0800`)
- Remote job: `3472424` (`s2f3f2011`), `COMPLETED`, exit code `0:0`
- Artifact status: `experimental_unvalidated`
- `operator_approved=false`
- `formal_gate_status=not_run_not_passed`
- Remote and local SHA256 manifests match exactly for `124` accepted files.

## Training contract

- Fresh second-stage `0 -> 200`; `resume=false`
- Single `NVIDIA A100-PCIE-40GB`
- Fixed training split: `1035` cases = `823` authentic T2W + `212` completion-required cases
- Fixed validation split: `103` authentic-T2W cases
- Train/validation overlap: `0`
- The `27` synthesized-T2W validation cases were not merged into this split. They remain a separate end-to-end evaluation endpoint.

## Accepted outputs

The local artifact tree contains the trainer metadata, `fold_0`, final and best checkpoints, training event/log files, `103` validation NIfTI predictions, and validation summary. Lightweight provenance contains the split, evidence JSON/marker files, Slurm script, stdout/stderr, and SHA256 manifests.

- Final checkpoint: `249,829,936` bytes
- Final checkpoint SHA256: `b8b2d13a6268231f73d43c37cb097d1be3daeec654b9bba6ddd24f410cf7b27e`
- Checkpoint epoch: `200`
- Fixed validation predictions: `103/103`
- Foreground mean Dice: `0.546763109603013`
- Validation summary SHA256: `16e595aff071fcbf95cacea09e352694ff86d81121315e4494d3a330257ce5fb`

The transfer was verified by loading the final checkpoint in the existing `brats2023_seg` Conda environment. The validation summary reported mean class Dice values of `0.4711364229033399`, `0.7359882901680868`, `0.6842557880989798`, and `0.2956719372416456` for labels `1`-`4`.

## Interpretation boundary

This directory records successful transfer and artifact-level acceptance only. It does not mark the formal gate as passed, does not constitute operator approval, and does not claim the official MET paired comparison or the separate 27-case synthesized-T2W endpoint validation. The results must not be used to claim general nnU-Net segmentation superiority outside this experimental contract.

See `TRANSFER_VALIDATION.json` for the machine-readable acceptance record and `LOCAL_SHA256SUMS.txt` / `provenance/REMOTE_SHA256SUMS.txt` for the file-level audit.
