# QC

保存 synthetic data 质量控制规则、逐例指标模板、扩散质量专项模板、人工复查表头、官方 leaderboard 对齐模板，以及正式 run 级自动 QC 输出。

主要脚本：

1. `../../code/g2_synthetic_raw_intake_qc.py`：生成 `qc_metrics_{run_id}.csv`、`diffusion_quality_metrics_{run_id}.csv`、`qc_batch_summary_{run_id}.json`。
2. `../../code/g2_official_mets_metrics_parser.py`：解析/校验 2026 Task1 官方同款字段。

核心规则文件：

1. `G2_synthetic_data_QC规则策略_v2.md`：G2 数据入口 QC 和 S1/S2 对接总规则。
2. `G2_official_metrics_alignment_QC_strategy_2026-06-15.md`：官方检测策略完整总结。
