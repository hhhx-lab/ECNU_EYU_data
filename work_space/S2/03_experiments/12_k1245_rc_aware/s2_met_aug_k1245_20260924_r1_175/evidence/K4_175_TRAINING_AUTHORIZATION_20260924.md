# K4 175-epoch training authorization

- Authorization source: explicit user instruction on 2026-09-24 (Asia/Shanghai).
- Scope: submit one independent K4 RC-aware training attempt at 175 epochs.
- Route: `E-SNFH* + X3`.
- Initialization: original E checkpoint, SHA256 `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`.
- Budget: 175 epochs; retained checkpoints `25/50/75/100/125/150/175`.
- Fixed split: train 1035, validation 103; seed `20260724`; one A100; non-DDP.
- Smoke evidence: automatic K4 R5 report is 24/24 pass. The source `manual_review.csv` was an empty template; image inspection was not converted into a fabricated 24/24 acceptance record.
- This file records user authorization for engineering execution only. It does not assert a scientific gate, formal calibration, or superiority.
- `F_STRICT_PREREG_GATE=false`.
- `artifact_status=experimental_unvalidated`.
- `operator_approved=false`.
- `formal_gate_status=not_run_not_passed`.
