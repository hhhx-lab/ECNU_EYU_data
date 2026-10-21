# S2 内部验证评估结果

本目录保存 S2 real-only current 模型在固定 103 例内部 validation 上运行 BraTS 官方兼容评估得到的轻量结果。它用于模型诊断和提交前判断，不是 Synapse 官方 179 例 validation 的提交包。

## 文件索引

| 文件 | 内容 | 当前核验结果 |
|---|---|---|
| `leaderboard_metrics.csv` | 103 个内部 validation 病例的 lesion-wise、NSD、HD95 和 small-instance 指标；末尾附 `mean/std/median` 三行 | 103 例 |
| `nnunet_to_source_id.tsv` | 当前 Dataset263 的 nnU-Net ID 到原始 BraTS-MET ID 的无表头映射；覆盖训练与内部验证全集 | 926 对唯一映射 |
| `panoptica_evaluation_summary.json` | Panoptica 逐病例、逐区域、逐实例的完整评估明细 | 103 例，`missings=[]` |

映射文件中的两份 corrected label 病例为：

```text
BraTSMET_000733 -> BraTS-MET-01094-003
BraTSMET_000931 -> BraTS-MET-01184-002
```

## 使用限制

1. 不要把本目录中的 CSV、TSV 或 JSON 上传到 Synapse。
2. 官方提交必须对独立的 179 例无标签 validation 运行推理，并上传 `S2_realonly_current_Task1_validation_179.zip`。
3. `panoptica_evaluation_summary.json` 是评估器原始输出，其中可能出现 `NaN` 或 `Infinity`；为保持审计一致性，不要手工改写。
4. 若重新训练或更换 checkpoint，应将新结果放入新的 run 目录，不要覆盖本轮文件后仍沿用旧结论。

官方 179 例推理与打包步骤见：

```text
work_space/S2/BraTS2026_S2_RC_v1.0/repository/docs/S2_服务器运行手册.md
```
