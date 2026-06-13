# G2 Pretraining Checklist Execution Summary

生成日期：2026-06-14

## 已在本机完成

1. 外部训练集、验证集、corrected labels 路径检查。
2. 训练集 raw manifest。
3. validation manifest，并标记不可作为 synthetic source。
4. corrected label overlay。
5. overlay 后 final train manifest。
6. label 值域统计与非法标签排查。
7. lesion connected component 统计与 tiny/small/large 分档。
8. G1 GliGAN-compatible source CSV。
9. nnU-Net real-only 映射表与 dataset.json 草案。
10. 固定 fold0 split。
11. synthetic QC 模板、消融模板、报告模板。

## 暂缓项

1. 不在本机复制或软链接 31GB 训练 NIfTI 到 nnU-Net raw 目录。
2. 不在本机运行 `nnUNetv2_plan_and_preprocess`。
3. 不在本机训练 GliGAN/diffusion 或执行在线 batch 生成。
4. 不在本机生成大量 synthetic NIfTI。

## 产物列表

- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/reports/local_data_paths_check.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/real_train_manifest_raw.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/real_validation_manifest.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/corrected_label_overlay.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/real_train_manifest.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/reports/real_data_qc_summary.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/stats/real_label_distribution.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/stats/real_lesion_distribution.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/stats/real_lesion_distribution_summary.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/stats/real_lesion_distribution_summary.json`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/stats/target_synthetic_distribution_v1.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/g1_gligan_source_cases_v1.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/splits/splits_final_fold0_realval.json`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_generation_manifest_template_g1.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_normalized_mapping_template.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_metrics_template.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_metrics_template_v2.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/diffusion_quality_metrics_template.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_case_review_template.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_rules_v1.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/reports/ablation_plan_template.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/reports/G2_synthetic_data_quality_report_template.md`
