# G1 Diffusion V3 Final Archive

## Scope and separation

This directory is the preserved final archive for the four single-modality G1 Diffusion V3 training runs completed on 2026-07-20. It is intentionally separate from `../missing_t2w_completion/`: that directory contains the prior missing-T2W completion output, while this archive contains model-training provenance, checkpoints, logs, preparation evidence, and data-split audits for downstream checkpoint selection and evaluation.

Modalities and source hosts:

| Modality | Source host | Final checkpoint |
| --- | --- | --- |
| t1c | a800_117.50.198.191 | step 150000 |
| t1n | h20_117.50.196.61 | step 150000 |
| t2w | a100_117.50.177.229 | step 150000 |
| t2f | a800_117.50.198.191 | step 150000 |

## Layout

- `sources/`: immutable, host-separated pullback snapshots. These retain complete server-side provenance, including archived pre-resume logs, checkpoint scan samples, loss records, and every retained checkpoint.
- `canonical/`: the stable downstream entry point. Its checkpoint and supporting files are hard links to `sources/`, so there is no second payload copy and both paths refer to the same archived bytes.
- `canonical/checkpoints/brats2026_diffusion_v3_edm_zscore/<modality>/weights/`: contains retained checkpoints through `diffusion_150000.pt`. Use `diffusion_150000.pt` by default. `diffusion_140000.pt` and `diffusion_145000.pt` are explicit rollback candidates if the 103-case evaluation rejects the final choice.
- `canonical/logs/`: complete current and pre-resume training/watchdog logs, grouped by source host.
- `canonical/splits/current/`: the canonical dataset membership, lesions, validation suffixes, and scan-content QC audit files. The three source copies were hash-equivalent.
- `canonical/prepared/`: one `PREPARED.ok` file per source host. All three files have the same SHA256.
- `canonical/pids/`: host-separated historical PID and watchdog PID files. They are provenance only and must never be interpreted as proof that a process is still running.
- `metadata/`: machine-readable inventories and SHA256 manifests.

## Required checkpoint verification

The default final-checkpoint identities are recorded in `metadata/FINAL_CHECKPOINTS.tsv`:

| Modality | SHA256 |
| --- | --- |
| t1c | `cc49de179dee75af561df377ba323052da99525a58a99c60d2fe48f2c34d51a5` |
| t1n | `bc98c9423dad396ee235c89893c308b5e6d340667a8b10880825020b6e976ad6` |
| t2w | `1b42542f378375406e38a17ca380a608fd0005be4591ef2bcedabca925c3ff60` |
| t2f | `de2f219fe126dbb7974d61d8fe8697d239d1e0b18f735344feeab36d3f7d9e6c` |

Each final checkpoint is 53,749,405 bytes and its paired training log contains `TRAINING_COMPLETE step=150000`.

## Integrity manifests

- `metadata/SOURCE_INVENTORY.tsv`: per-host source role, regular-file count, byte count, and modality assignment.
- `metadata/FINAL_CHECKPOINTS.tsv`: source host, canonical path, step, size, SHA256, and completion-marker verification for all four final weights.
- `metadata/split_equivalence.txt`: hash-equivalence proof for the three source `splits/current` trees.
- `metadata/SHA256SUMS_sources.txt`: SHA256 checksums for all original source-snapshot files.
- `metadata/SHA256SUMS_canonical.txt`: SHA256 checksums for all canonical-view files.

Verify later without changing the archive:

```bash
cd /Users/hwaigc/比赛+课题/ECNU_EYU_data
shasum -a 256 -c work_space/G1/results/g1_diffusion_v3_final_20260720/metadata/SHA256SUMS_sources.txt
shasum -a 256 -c work_space/G1/results/g1_diffusion_v3_final_20260720/metadata/SHA256SUMS_canonical.txt
```

## Downstream use

Checkpoint selection for the 103-case evaluation must begin from the four `diffusion_150000.pt` files under `canonical/checkpoints/`. Record any selection or rollback decision in the evaluation artifact rather than altering this archive. The data-completion result for later G2/S2 stages remains in its separate `missing_t2w_completion` archive.
