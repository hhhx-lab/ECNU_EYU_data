# S2 Screen125 selected-checkpoint official MET comparison

This independent experimental bundle evaluates the five pre-registered selected
checkpoints from the 2026-08-31 Screen125 run on the same frozen 103-case
development validation cohort.

The fixed comparison is B125, C40-50, D50, E-SNFH125, and F125. The B/E/F
125-to-175 convergence extension is a separate experiment and cannot replace
any checkpoint in this equal-budget comparison.

The five inference and official MET jobs run independently on one A100 each.
A CPU-only paired comparison is submitted with an afterok dependency on all
five jobs. It computes all ten method pairs with 20,000 paired bootstrap
replicates and seed 20260831.

These results are exploratory because the same 103 cases were used for
checkpoint selection and reporting. They are not untouched-test or formal
superiority evidence.
