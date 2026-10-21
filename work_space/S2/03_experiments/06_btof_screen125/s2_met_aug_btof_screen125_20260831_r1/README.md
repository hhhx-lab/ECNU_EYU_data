# S2 MET-AUG B-F 125-epoch screening (2026-08-31)

This experiment is an exploratory, unvalidated five-route screen derived from
the completed Fix-v3 Route A runtime. It does not alter or resume attempt-11.

The five independent jobs are A+B, A+C40, A+D, A+E-SNFH, and A+F. Each starts
from the same frozen original-E checkpoint, uses the same 1035/103 split, one
A100, seed 20260724, LR 0.001, and 125 epochs. Checkpoints are retained at epochs
25, 50, 75, 100, and 125.

For each route, the best checkpoint is selected only from those five snapshots
by the fixed-103 internal-validation EMA foreground Dice, with the earliest
epoch winning an exact tie. All five candidate scores remain in the audit. The selected five
checkpoints are then evaluated with the same official MET pipeline and compared
case-by-case on the frozen 103 cases. Because this same validation cohort is
used for model selection, the comparison is exploratory and is not an untouched
test-set or formal superiority claim.

E in this screen means only the lower-risk E-SNFH branch. E-NETC and E-COMB are
excluded. F keeps the mean configured selection probability at 0.20 and changes
only its allocation across train-only native lesion-burden strata.

All artifacts retain `experimental_unvalidated`, `operator_approved=false`, and
`formal_gate_status=not_run_not_passed` until a separate formal gate is run.
