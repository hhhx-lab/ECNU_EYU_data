# 01_ecnu_legacy_bundle_20260919

保持同批依赖文件相邻，原脚本及记录不修改、不执行。历史部署路径及平铺scripts布局仅适用于原版本，重用前必须制作新计划并预检。

- [01_prepare_completion_ecnu.slurm](01_prepare_completion_ecnu.slurm)
- [02_train_completion_v100_ecnu.slurm](02_train_completion_v100_ecnu.slurm)
- [03_train_small_lesion_ablation_a100_ecnu.slurm](03_train_small_lesion_ablation_a100_ecnu.slurm)
- [04_validate_existing_completion_cache_ecnu.slurm](04_validate_existing_completion_cache_ecnu.slurm)
- [18_prepare_fix_v3_full200_true1mm_ecnu.slurm](18_prepare_fix_v3_full200_true1mm_ecnu.slurm)
- [19_eval_e_missing_t2w_val27_ecnu.slurm](19_eval_e_missing_t2w_val27_ecnu.slurm)
- [19_rebuild_fix_v3_full200_calibration_ecnu.slurm](19_rebuild_fix_v3_full200_calibration_ecnu.slurm)
- [20_smoke_fix_v3_full200_ecnu.slurm](20_smoke_fix_v3_full200_ecnu.slurm)
- [21_train_fix_v3_full200_ecnu.slurm](21_train_fix_v3_full200_ecnu.slurm)
- [22_rebuild_fix_v3_full200_calibration_ecnu_attempt02.slurm](22_rebuild_fix_v3_full200_calibration_ecnu_attempt02.slurm)
- [23_smoke_fix_v3_full200_ecnu_attempt02.slurm](23_smoke_fix_v3_full200_ecnu_attempt02.slurm)
- [24_train_fix_v3_full200_ecnu_attempt02.slurm](24_train_fix_v3_full200_ecnu_attempt02.slurm)
- [dataset264_rsync_receiver.sh](dataset264_rsync_receiver.sh)
- [s2_storage_policy.sh](s2_storage_policy.sh)

整理时间：2026-09-19。本地整理标记：experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
