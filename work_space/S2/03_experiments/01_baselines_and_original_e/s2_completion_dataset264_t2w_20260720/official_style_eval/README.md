# S2 候选 B 固定 103 例官方兼容评估

## 结论

候选 B 已使用 `BraTS_evaluation==0.0.8`、`panoptica==2.1.0` 和 MET 配置完成固定 103 例内部评估。该结果用于候选 B/D 的同口径内部比较，不是 Synapse 官方 validation 分数。

- `config=mets`
- `vol_threshold=27`
- `overlap_threshold=0.2`
- prediction/reference/mapping：103/103/103
- evaluator `missings=[]`
- `leaderboard_metrics.csv`：103 个病例行 + mean/std/median，共 106 行
- checkpoint SHA256：`78eccc59f9217a529cafdd522733de9a1578f0e96d8765ee7c48731027824db5`

## 核心指标

| 区域 | lesionwise DSC mean | lesionwise DSC median | lesionwise NSD mean | small-instance F1 mean |
|---|---:|---:|---:|---:|
| ET | 0.6193 | 0.7232 | 0.6885 | 0.3429 |
| RC | 0.2359 | 0.2184 | 0.2123 | 0.0000 |
| TC | 0.6523 | 0.7452 | 0.7021 | 0.3858 |
| WT | 0.5935 | 0.6181 | 0.5931 | 0.3306 |

RC 的 small-instance F1 为 0，是候选 B 的明确风险项。候选 D 只有在同一 103 例、同一 evaluator 参数下完成评估后才能与 B 比较；不能用 nnU-Net `validation/summary.json` 的 voxel Dice 替代。

CSV 汇总行中的 small-instance TP/FN/FP 是逐病例数值的均值，不是 103 例全局求和。后续作最终模型决策时应同时检查病例级 CSV、总 FN/FP 和分层风险病例。

## 产物

- `panoptica_evaluation_summary.json`：完整 Panoptica 病灶级结果。
- `leaderboard_metrics.csv`：官方 MET parser 输出。
- `preparation_summary.json`：输入计数、checkpoint 身份和映射合同。
- `nnunet_to_source_id.tsv`：本次 103 例的 nnU-Net ID 到 source ID 映射。
- `evaluation_contract.txt`：固定评估参数。
- `evaluation_environment.txt`：Python、评估包版本和入口哈希。
- `brats_evaluate.log`、`brats_parse_metrics.log`、`evaluation_driver.log`：完整运行日志。
- `EVALUATION_COMPLETE.ok`：完成时间及核心输出 SHA256。

核心输出 SHA256：

```text
c040228b6899bcf0ee173246fa2359ef13748ff9dafe5636f61428ecaf4cb3f5  panoptica_evaluation_summary.json
73f5b9e08aaee848ecb77d59e51698e66d6b050bec7b8073c25418904cce24cc  leaderboard_metrics.csv
```
