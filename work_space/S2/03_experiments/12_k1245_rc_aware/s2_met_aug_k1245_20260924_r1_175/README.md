# K1245 RC-aware 175-epoch revision

This is an independent exploratory revision of the frozen K1245 execution plan.
It changes the training budget for K1-K5 to 175 epochs and does not rewrite the
original 125-epoch plan, smoke attempts, failures, or their evidence.

All routes retain the original E initialization, fixed train/validation split,
single A100, non-DDP execution, seed 20260724, and checkpoints every 25 epochs.
Candidate epochs are 25, 50, 75, 100, 125, 150, and 175. Formal and scientific
gates remain not run/not passed; this batch is exploratory only.

The K4 training submission is intentionally a separate attempt from the K4 R5
smoke job. It must use the K1245 trainer and the frozen K4 route configuration;
the smoke runner is not a training runner.
