# S2 small-lesion ablation 20260721

This run compares the completed baseline B against three isolated candidates on
the same Dataset264 fixed split:

- A-1 only
- focal CE only
- A-1 plus focal CE

The three candidates share one preprocessing job but write to trainer-specific
nnU-Net result folders. A-1 candidates use audited shape-compatible warm-start;
the focal candidates use `gamma=2` while preserving Dice loss and RC class
weight `3.0`.

Selection was completed on 2026-07-24 after all four models were evaluated on the same
103 validation cases with BraTS official-compatible `mets` metrics and lesion-size
stratification. E (focal CE, `gamma=2`) is frozen as the small-lesion follow-up base;
B remains the conservative whole-tumour baseline. A-1 and A-1+E are retained for audit
but are not selected. The final comparison is in
`final_comparison_20260724.md`, with machine-readable provenance in
`checkpoint_selection.json` and per-case risks in `risk_review_20260724.md`.

Deep Supervision D is designed only as an E-based second-stage contrast and has not been
trained. This run does not authorize online Diffusion, official 179-case inference, or
submission.
