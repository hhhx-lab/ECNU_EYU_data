# S2 fixed-103 three-model aggregate

> status: experimental_unvalidated

本目录把以下三组同一固定 103 例分割评估重新按真实来源聚合：

1. Dataset263 real-only，无缺失模态补全样本增强；
2. Dataset264 completion augmentation + Dice/weighted CE；
3. Dataset264 completion augmentation + Dice/Focal CE。

三组病例 ID 完全一致，评估合同均为 BraTS-evaluation 0.0.8、panoptica 2.1.0、mets、vol_threshold=27、overlap_threshold=0.2。

注意：历史 final_comparison_20260724.md 把 s2_eval_results 的数值标为 B，但 checkpoint_selection.json 又把 B checkpoint 定义为 Dataset264 completion/plain-loss。两者证据绑定不一致。本目录不修改旧证据，而是将 s2_eval_results 明确列为 real-only，并使用 Dataset264 official_style_eval 作为真正的 completion/plain-loss。

文件：

- core_metrics_three_models.csv：常用核心指标三模型并排；
- aggregate_statistics_three_models.csv：全部指标 mean/std/median；
- per_case_three_models_long.csv：三模型 309 条逐病例记录；
- three_model_fixed103_aggregate.xlsx：上述内容的多工作表版本；
- SOURCE_MANIFEST.json：来源、SHA256 和评估合同。
