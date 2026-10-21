# G2 Synthetic Data Ablation Plan Template

## 实验组

| 实验 | 训练数据 | 验证数据 | 目的 |
|---|---|---|---|
| A | Real only | fixed real fold0 | baseline |
| B | Real + 0.25x accepted synthetic | fixed real fold0 | 小比例合成数据 |
| C | Real + 0.5x accepted synthetic | fixed real fold0 | 中等比例合成数据 |
| D | Real + G1 Regular online-style synth | fixed real fold0 | 对齐 G1 Regular 方案 |
| E | Real + G1 Custom online-style synth | fixed real fold0 | 对齐 G1 Custom 方案 |

## 固定变量

1. 同一 nnU-Net 配置。
2. 同一 fold。
3. 同一 preprocessing。
4. 同一训练 epoch/iteration。
5. 同一后处理。
6. 同一 evaluation 脚本。

## 记录指标

主指标必须对齐官方 leaderboard：ET/RC/TC/WT 的 lesionwise DSC/NSD，以及 ET/TC/WT/RC 的 small-instance TP/FN/FP/F1。HD95、AUC、NETC/SNFH/ET/RC 单类均值只能作为内部辅助分析。

| 指标组 | 字段 |
|---|---|
| lesionwise segmentation | `lesionwise_dsc_mean_et/rc/tc/wt`, `lesionwise_nsd_mean_et/rc/tc/wt` |
| small-instance detection | `small_instance_tp/fn/fp/f1_et` |
| small-instance detection | `small_instance_tp/fn/fp/f1_tc` |
| small-instance detection | `small_instance_tp/fn/fp/f1_wt` |
| small-instance detection | `small_instance_tp/fn/fp/f1_rc` |
