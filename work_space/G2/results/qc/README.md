# QC

更新日期：2026-07-12

## 主文件

| 文件 | 用途 |
|---|---|
| `G2_synthetic_data_QC规则策略_v2.md` | V2/V3 技术硬门、质量复核和训练价值验收 |
| `G2_official_metrics_alignment_QC_strategy_2026-06-15.md` | 2026 Task1 官方评价口径 |
| `G2_synthetic_data_QC报告模板_v2.md` | 正式 run 报告模板 |
| `qc_metrics_template_v2.csv` | 逐例总 QC 字段 |
| `diffusion_quality_metrics_template.csv` | 生成质量专项字段 |
| `qc_case_review_template.csv` | 人工复核字段 |
| `official_leaderboard_metrics_template.csv` | 官方 leaderboard 字段模板，必须保留 |

## Release 原则

硬门失败为 `rejected`；技术通过但没有人工/teacher 审批为 `pending_review`；只有正确角色审批后才可成为 `accepted_for_training` 或 `accepted_for_evaluation`。

## 必须保留的真实数据证据

1. `official_t2w_gzip_header_audit_2026-06-15.csv`
2. `official_fake_t2w_cases_by_gzip_header_2026-06-15.csv`
3. `official_non000_t2w_cases_2026-06-15.csv`
4. `UCSD_T2W_内容异常检查报告_2026-06-14.md`

这些不是旧 synthetic 产物，不能删除。
