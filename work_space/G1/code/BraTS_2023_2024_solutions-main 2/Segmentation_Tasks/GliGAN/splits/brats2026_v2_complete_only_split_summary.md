# BraTS2026 V2 complete-only train/val split

- Source manifest: `work_space/G1/data/g1_data_placement_manifest.csv`
- Inclusion rule: `is_fake_t2w=False` and `final_qc_pass=True`.
- Exclusion rule: fake/broken T2W cases are not used before completion outputs are available.
- Split rule: deterministic SHA256 hash on the 9-character patient suffix; lowest 20% assigned to validation.
- Total complete cases: 1030
- Train cases: 824
- Validation cases: 206
- `csv_creator.py --val_patients` must use `brats2026_v2_complete_only_val_patients_one_line.txt`.
