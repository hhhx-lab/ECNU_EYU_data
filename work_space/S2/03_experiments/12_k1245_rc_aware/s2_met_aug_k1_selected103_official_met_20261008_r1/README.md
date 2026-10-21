# K1 selected-175 fixed103 official MET

Independent inference/evaluation attempt for K1 after job 3919543 completed cumulative epoch175. Reuses the SHA-identical K2/K4 official MET inference engine and exact fixed103 real-T2W cohort; no training or online augmentation. One single-A100 job, sequential in-process inference, zero preprocess/export workers. Deploy only after remote static preflight; one held launch audit, no duplicate submission.

Results remain `experimental_unvalidated`, `operator_approved=false`, `formal_gate_status=not_run_not_passed`. The same 103 cases selected checkpoints and report official MET, so all comparisons are exploratory. K5 is not authorized by this package; it requires separate K1 and K2 scientific-gate acceptance.
