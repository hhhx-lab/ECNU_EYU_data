# Splits

保存固定真实 train/val/test 划分，供 G1、S1、S2、G2 QC、real-only、real+synth 和后续所有消融复用。

当前正式口径：

1. `splits_final_train_val_test.json`
   - G2 全量 final QC pass 真实病例：1295 例。
   - train：829 例，用于真实数据训练池和 synthetic source 池。
   - val：207 例，用于调参、早停、G1 `s`/`weight_decay` 试验和 S1/S2 dev 评估。
   - test：259 例，内部 holdout，只做最终内部测试，不训练、不调参、不作为 synthetic source。

2. `splits_final_train_val_test_membership.csv`
   - 逐病例 membership 表。
   - 包含 `nnunet_case_id`、`source_case_id`、`split` 和稳定 hash 分数，便于人工核查。

历史兼容文件：

1. `splits_final_fold0_realval.json`
   - 旧 two-way fold：1036 train / 259 val。
   - 当前不再把旧 `val` 当调参验证集；它已经锁定为 internal test。
   - 保留它只是为了兼容旧脚本和追溯历史结果。
