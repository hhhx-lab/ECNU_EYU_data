# Splits

更新日期：2026-07-12

## 唯一规则

病例按 patient group 划分：去掉 `BraTS-MET-xxxxx-yyy` 最后的数字后缀。同一患者的 `-000/-001/-002` 不得跨 split。

## 文件

| 文件 | 口径 |
|---|---|
| `splits_master_train_val_test.json` | master 1295：1035/130/130 |
| `splits_master_train_val_test_membership.csv` | master 逐例身份、patient group、T2W 状态和 split |
| `splits_final_train_val_test.json` | real-only 1030：823/103/104 |
| `splits_final_train_val_test_membership.csv` | real-only 逐例 membership |

master 的 265 个 V3 completion 目标分布为 train 212、val 27、test 26。completion 完成后回到原 split；V2 augmentation 永远只追加 train。

历史 two-way split 已清理，不再作为正式接口。既有历史 checkpoint 若使用不同 split，只能作为历史结果；严格 real-only vs real+synth 消融要在当前 master split 上建立 paired baseline。
