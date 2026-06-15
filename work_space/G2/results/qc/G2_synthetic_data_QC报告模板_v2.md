# G2 Synthetic Data QC 报告模板 v2

报告日期：YYYY-MM-DD
报告人：G2
生成批次：`run_id`
QC 规则版本：`G2_synthetic_data_QC规则策略_v2.md`
结论：`accepted_for_training / accepted_for_ablation_only / needs_regeneration / rejected`

## 1. 批次信息

| 项目 | 内容 |
|---|---|
| generation_run_id |  |
| generator_name |  |
| generator_checkpoint |  |
| generator_git_commit |  |
| generation_mode |  |
| source_manifest |  |
| requested_num_synthetic |  |
| delivered_case_count |  |
| generation_date |  |

## 2. 交付完整性

| 检查项 | 结果 | 备注 |
|---|---|---|
| `synthetic_generation_manifest.csv` 是否存在 |  |  |
| `generation_log.jsonl` 是否存在 |  |  |
| 每例 `metadata.json` 是否存在，或 G2 是否能从 manifest/log 恢复核心字段 |  |  |
| checkpoint/seed/source/label_kind 是否可追溯 |  |  |
| 文件命名是否一致 |  |  |

## 3. 单例 QC 统计

| 指标 | 数量 |
|---|---:|
| delivered cases |  |
| readable cases |  |
| hard rejected cases |  |
| manual review required cases |  |
| manual review passed cases |  |
| accepted cases |  |
| rejected cases |  |

## 4. 强制拒绝原因

| 拒绝原因 | 病例数 | 病例示例 |
|---|---:|---|
| missing modality/seg |  |  |
| unreadable NIfTI |  |  |
| geometry mismatch |  |  |
| NaN/Inf or constant image |  |  |
| illegal label values |  |  |
| non-integer label |  |  |
| empty mask not allowed |  |  |
| source not allowed |  |  |
| validation leakage |  |  |
| severe artifact |  |  |
| missing provenance |  |  |

## 5. source 与泄漏检查

| 检查项 | 结果 |
|---|---:|
| source 不在 real_train_manifest 的病例数 |  |
| source final_qc_pass 非 True 的病例数 |  |
| source 来自固定 val fold 的病例数 |  |
| source 来自 official validation 的病例数 |  |
| 复用真实病例 ID 的 synthetic 数 |  |
| validation leakage 总数 |  |

结论：validation leakage 必须为 `0`。

## 6. 标签与 lesion 分布

| 指标 | 数值 |
|---|---:|
| accepted cases |  |
| 含 NETC 病例数 |  |
| 含 SNFH 病例数 |  |
| 含 ET 病例数 |  |
| 含 RC 病例数 |  |
| lesion component 总数 |  |
| tiny lesion 数 |  |
| small lesion 数 |  |
| large lesion 数 |  |
| 每例 lesion 数 p50/p95/p99/max |  |
| component volume p50/p95/p99/max |  |

## 7. 人工复查结果

| 病例 | 复查原因 | 复查结论 | 备注 |
|---|---|---|---|
|  |  |  |  |

## 8. teacher model 检查

| 指标 | 数值 |
|---|---:|
| teacher model |  |
| mean teacher Dice/NSD proxy |  |
| mean lesion count diff |  |
| large lesion missed by teacher |  |
| teacher 异常需复查病例数 |  |

说明：teacher 结果只作为辅助 QC，不单独决定是否拒绝。

## 9. 扩散生成质量专项

| 指标 | 数值 | 结论 |
|---|---:|---|
| ROI boundary pass rate |  |  |
| z continuity pass rate |  |  |
| modality consistency pass rate |  |  |
| source-synth ROI similarity 异常数 |  |  |
| synth-synth MS-SSIM 异常数 |  |  |
| duplicate/hash 命中数 |  |  |
| medical feature FID/MMD 是否仅作辅助 |  |  |
| excellent / accepted / ablation_only / rejected |  |  |

结论：只有 hard gate 通过、ROI 边界自然、z 轴连续、多模态一致、无明显复制风险，并且人工复查完成后，才能称为“QC 质量优秀”。只有 real+synth 消融不伤害真实验证指标并带来稳定收益，才能称为“训练价值优秀”。

## 10. 官方指标对齐验收

本节只在 S1/S2 训练出预测后填写。字段必须对齐 `official_leaderboard_metrics_template.csv`。

| 实验 | lesionwise_dsc_mean_et | lesionwise_nsd_mean_et | lesionwise_dsc_mean_rc | lesionwise_nsd_mean_rc | lesionwise_dsc_mean_tc | lesionwise_nsd_mean_tc | lesionwise_dsc_mean_wt | lesionwise_nsd_mean_wt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real-only |  |  |  |  |  |  |  |  |
| real+synth |  |  |  |  |  |  |  |  |
| delta |  |  |  |  |  |  |  |  |

| 实验 | small_instance_tp_et | small_instance_fn_et | small_instance_fp_et | small_instance_f1_et | small_instance_tp_tc | small_instance_fn_tc | small_instance_fp_tc | small_instance_f1_tc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real-only |  |  |  |  |  |  |  |  |
| real+synth |  |  |  |  |  |  |  |  |
| delta |  |  |  |  |  |  |  |  |

| 实验 | small_instance_tp_wt | small_instance_fn_wt | small_instance_fp_wt | small_instance_f1_wt | small_instance_tp_rc | small_instance_fn_rc | small_instance_fp_rc | small_instance_f1_rc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real-only |  |  |  |  |  |  |  |  |
| real+synth |  |  |  |  |  |  |  |  |
| delta |  |  |  |  |  |  |  |  |

判定：

1. 官方核心字段是否下降：
2. small-instance FN 是否增加：
3. small-instance FP 是否显著增加：
4. 本批 synthetic 是否允许进入主训练：

## 11. nnU-Net integrity check

| 项目 | 结果 |
|---|---|
| real+synth dataset id |  |
| dataset.json 是否正确 |  |
| imagesTr/labelsTr 数量是否匹配 |  |
| `nnUNetv2_plan_and_preprocess --verify_dataset_integrity` |  |
| 失败原因 |  |

## 12. 结论与后续动作

最终结论：

后续动作：

1.
2.
3.
