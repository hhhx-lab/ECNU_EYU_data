# G1 VAE three-way fixed-103 validation

This experiment compares three VAE checkpoints on the same 103 development
validation cases:

1. frozen pretrained MAISI VAE;
2. historical `run_3034060/best_model.pt`;
3. R3 attempt-03 `checkpoint_epoch_004.pt`.

Stage 1 performs deterministic four-modality model-space reconstruction. It
reports native-space T2W quality both before intensity postprocessing and after
the deployed postprocessing, and saves the deployed native-space T2W
reconstructions. Stage 2 replaces only T2W, keeps the
same T1N/T1C/T2F and reference label, and evaluates all routes with the same
frozen FixV3 checkpoint and official BraTS-MET evaluator. Authentic T2W is
included as a downstream anchor.

This isolates VAE reconstruction behavior. It does not evaluate missing-T2W
generation through EncDec/BBDM and does not authorize model replacement.

All outputs remain `experimental_unvalidated`,
`operator_approved=false`, and
`formal_gate_status=not_run_not_passed`.
