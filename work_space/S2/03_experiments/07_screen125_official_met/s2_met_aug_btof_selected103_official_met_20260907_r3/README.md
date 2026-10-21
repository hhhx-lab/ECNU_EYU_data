# S2 Screen125 selected103 official MET recovery r3

This experimental, unvalidated recovery retries only C40 and D after r2 exhausted
node shared memory. It preserves r2 unchanged, uses fresh output directories, one
preprocessing/export worker, file-system tensor sharing, and serial exclusive-node
jobs. The paired job combines successful r2 B/E-SNFH/F metrics with r3 C40/D.

The fixed 103 development cases were used for checkpoint selection and reporting.
Results are exploratory only; `operator_approved=false` and
`formal_gate_status=not_run_not_passed`.
