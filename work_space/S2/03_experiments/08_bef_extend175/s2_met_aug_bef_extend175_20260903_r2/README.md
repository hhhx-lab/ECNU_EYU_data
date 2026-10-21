# S2 MET-AUG B/E/F Extend175 (2026-09-03 R2)

This directory defines an independent exploratory convergence extension for
the B, E-SNFH, and F routes whose pre-registered Screen125 optimum occurred at
epoch 125.

The original five-route 125-epoch experiment is immutable. Its official MET
comparison must continue to use the original pre-registered selected
checkpoints. This extension resumes each selected epoch-125 state and trains
for 50 additional epochs under a continuous low-learning-rate polynomial
decay. Candidate total epochs are 125, 150, and 175.

All outputs remain `experimental_unvalidated`, with
`operator_approved=false` and `formal_gate_status=not_run_not_passed`.

R1 stopped before submission because its preflight round-trip check compared
Python tuples with their valid JSON list representation. R2 changes only that
audit serialization and preserves the R1 evidence.
