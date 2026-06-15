# G2 Pretraining Checklist Execution Summary

生成日期：2026-06-15

## 已在本机完成

1. 外部训练集、验证集、corrected labels 路径检查。
2. 训练集 raw manifest 与 final train manifest。
3. validation manifest，并标记不可作为 synthetic source。
4. corrected label overlay。
5. label 值域统计与非法标签排查。
6. lesion connected component 统计与 tiny/small/large 分档。
7. G1 GliGAN-compatible source CSV。
8. nnU-Net real-only 映射表与 dataset.json 草案。
9. 固定 fold0 split。
10. synthetic QC v2 模板、扩散专项模板、人工复查模板、报告模板。
11. 官方 leaderboard 指标对齐策略与 CSV 模板。
12. T2W gzip header fake/synthetic T2W 全量 audit。
13. G1 raw intake/QC、nnU-Net 物化、官方指标解析/校验三个脚本入口。

## 已清理

1. 2026-06-14 smoke run 演示 CSV/JSON/报告。
2. 旧版 `qc_rules_v1.md`。
3. 旧版 `qc_metrics_template.csv`。
4. `work_space/G2/results/.DS_Store` 本地缓存文件。

## 暂缓项

1. 不在本机复制或软链接 31GB 训练 NIfTI 到 nnU-Net raw 目录。
2. 不在本机运行 `nnUNetv2_plan_and_preprocess`。
3. 不在本机训练 GliGAN/diffusion 或执行在线 batch 生成。
4. 不在本机生成大量 synthetic NIfTI。
5. 等 S1/S2 训练出预测后，再填 `official_leaderboard_metrics_template.csv` 同款指标。

## 关键产物

- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/real_train_manifest.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/real_validation_manifest.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/g1_gligan_source_cases_v1.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/nnunet_case_mapping_realonly.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/G2_synthetic_data_QC规则策略_v2.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/official_leaderboard_metrics_template.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_metrics_template_v2.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/diffusion_quality_metrics_template.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_case_review_template.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/official_t2w_gzip_header_audit_2026-06-15.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/official_non000_t2w_cases_2026-06-15.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/stats/real_lesion_distribution_summary.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/splits/splits_final_fold0_realval.json`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/reports/ablation_plan_template.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/G2_synthetic_data_QC报告模板_v2.md`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/code/g2_synthetic_raw_intake_qc.py`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/code/g2_materialize_nnunet_dataset.py`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/code/g2_official_mets_metrics_parser.py`
