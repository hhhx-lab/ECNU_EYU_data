# G2 Synthetic Data QC 报告模板 v2

run ID：`<generation_run_id>`
日期：`YYYY-MM-DD`
产线：`V2 augmentation / V3 completion`
结论：`rejected / pending_review / accepted_for_training / accepted_for_evaluation`

## 1. 运行证据

| 项目 | 值 |
|---|---|
| G1 代码版本 | |
| generator | |
| checkpoint | |
| seed | |
| source manifest | |
| master split | |
| sampling / bbdm_s | |
| config / manifest / log | |

## 2. 数据数量

| 状态 | 数量 |
|---|---:|
| delivered | |
| composed successfully | |
| rejected | |
| pending review | |
| accepted for training | |
| accepted for evaluation | |

## 3. 硬门结果

| 检查 | 通过率 | 失败病例 | 处理 |
|---|---:|---|---|
| metadata | | | |
| source/split/identity | | | |
| file completeness | | | |
| NIfTI geometry | | | |
| numeric validity | | | |
| label validity | | | |
| source seg unchanged | | | |
| V2 non-ROI unchanged | | | |
| V3 protected modalities unchanged | | | |

## 4. 质量指标

记录分布、异常阈值和异常病例，不只写平均值：

1. boundary MAE / gradient jump / block score。
2. z continuity。
3. intensity drift p1/p50/p99，仅 V2。
4. ROI SSIM，仅 V2。
5. lesion count / volume / tiny ratio。
6. ET-T1C、SNFH-T2F/T2W 对齐。
7. brain overlap。

未运行指标：`teacher / MS-SSIM / medical FID-MMD / duplicate detection / 其他`。

## 5. 人工或 Teacher 审批

| 病例 | split | 触发原因 | 复核结果 | 审批角色 | 复核者 | 日期 |
|---|---|---|---|---|---|---|

## 6. 物化与完整性

| 项目 | 结果 |
|---|---|
| nnU-Net view | |
| case-folder view | |
| fixed split | |
| G2 integrity | |
| nnU-Net integrity | |
| synthetic in val/test | 必须为 0 |

## 7. 成对消融

| 实验 | lesionwise DSC | lesionwise NSD | small-instance F1 | 结论 |
|---|---|---|---|---|
| real-only | | | | |
| + V3 completion | | | | |
| + V2 augmentation | | | | |
| + V3 + V2 | | | | |

## 8. 最终结论

明确写出：

1. 哪些病例可训练、哪些仅评估、哪些拒绝。
2. 仍有哪些未完成检查。
3. 是否建议进入主训练。
4. 是否需要 G1 重生成及具体原因。
