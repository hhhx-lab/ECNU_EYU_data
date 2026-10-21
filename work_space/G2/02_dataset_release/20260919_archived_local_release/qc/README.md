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

## G1 V3 阶段 5 实际运行目录

```text
v3_paired_validation/run_<stage5_jobid>/  # 103 例影像/病灶指标与 montage
v3_s2_teacher/run_<stage5_jobid>/         # 冻结 S2 成对 teacher 指标
```

paired validation 必须保留同一 Stage 5 run 的 `spatial_audit.csv`，并硬验病例 ID
和 foreground/lesion FOV containment。两个目录都完成后仍需人工复核
paired validation 的 high/medium/routine 分层病例；单独一个目录不能作为阶段 6 批准依据。

## Diffusion 150k checkpoint gate

```text
diffusion_checkpoint_smoke20_150000_*/  # 20 例 smoke 技术与人工复核
diffusion_checkpoint_full94_150000_*/   # 94 个 lesion-positive 全量技术与人工复核
```

固定 validation 口径为 103 例，其中 94 例有病灶并进入四模态生成，9 例 segmentation
全零并单列验证 validation strict no-op。最终 gate 必须同时归档 cohort SHA256、9/9
`was_modified=False + image_equal + seg_equal`、94/94 四模态输出、checkpoint SHA256、
自动 QC、人工复核和冻结的 `checkpoint_selection.json`。

2026-07-21 最终 gate 已完成：

```text
diffusion_checkpoint_full94_150000_a800_recovery_20260721/
```

该目录包含 `automatic_qc/`、`generation_evidence/`、`input_evidence/`、
`manual_review/`、`checkpoint_selection.json`、`g2_diffusion_qc_gate.json` 和
`SHA256SUMS.txt`。最终 `decision=approve`，固定四模态 `150000` checkpoint，
不触发 `145000/140000` 回退对比；本 gate 完成后仍不得自动启动 S2 D 或官方
179 例推理。

## Release 原则

硬门失败为 `rejected`；技术通过但没有人工/teacher 审批为 `pending_review`；只有正确角色审批后才可成为 `accepted_for_training` 或 `accepted_for_evaluation`。

## 必须保留的真实数据证据

1. `official_t2w_gzip_header_audit_2026-06-15.csv`
2. `official_fake_t2w_cases_by_gzip_header_2026-06-15.csv`
3. `official_non000_t2w_cases_2026-06-15.csv`
4. `UCSD_T2W_内容异常检查报告_2026-06-14.md`

这些不是旧 synthetic 产物，不能删除。
