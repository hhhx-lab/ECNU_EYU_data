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

Dice、NSD、lesion-wise F1/AUC、tiny/small/large 分档表现、false positive components、NETC/SNFH/ET/RC 分项表现。
