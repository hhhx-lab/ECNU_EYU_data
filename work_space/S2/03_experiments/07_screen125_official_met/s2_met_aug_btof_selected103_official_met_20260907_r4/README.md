# S2 Screen125 selected103 official MET recovery r4

This experimental, unvalidated recovery retries only C40 and D after r2 exhausted
node shared memory and r3's exclusive allocation exposed two GPUs. It preserves all
prior attempts, uses fresh output directories, and performs fully in-process,
per-case inference with no multiprocessing queue. C40 and D remain serial single-GPU
jobs. The paired job combines successful r2 B/E-SNFH/F metrics with r4 C40/D.

The fixed 103 development cases were used for checkpoint selection and reporting.
Results are exploratory only; `operator_approved=false` and
`formal_gate_status=not_run_not_passed`.
