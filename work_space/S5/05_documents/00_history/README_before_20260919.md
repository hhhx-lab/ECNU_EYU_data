# S5 工作区

更新日期：2026-07-12

## 任务

S5 是 SegMamba 分割线。任务详情见 `task_assignment.md`。

## 当前 G2 对齐状态

1. 预处理通道已统一为 `t1n,t1c,t2w,t2f`。
2. 预处理可直接读取 G2 带病例前缀的 case-folder view，不需要破坏性重命名。
3. G2 case-folder root 必须包含 `train/val/test`。
4. 训练和预测默认读取固定的 `data/fullres/train|val|test`，不再默认随机 70/10/20。
5. 旧随机切分只有显式配置 `allow_random_split: true` 时才能用于探索，不得用于正式对比。
6. `5_compute_metrics_brats2026.py` 是内部辅助指标脚本，不是官方 BraTS evaluation 替代品。

## 数据准备

先由 G2 materializer 生成病例目录，再在 S5 code 根运行：

```bash
python 2_preprocessing_mri.py \
  --g2-case-folder-root ./data/g2_case_folders \
  --output-root ./data/fullres
```

预期 real-only 数量：823 train、103 val、104 locked test。completion/real+synth 使用新的 G2 物化根和新的输出目录。

## 训练

```bash
cd work_space/S5/code
python 3_train_brats2026.py --config configs/exp_ce_dice.yaml
python 4_predict_brats2026.py --config configs/exp_ce_dice.yaml
python 5_compute_metrics_brats2026.py --config configs/exp_ce_dice.yaml \
  --pred_name segmamba_brats2026_exp_ce_dice
```

正式实验必须确认 config 的 `data_dir` 指向 train，兄弟目录 `val`、`test` 存在，且 `raw_test_dir` 指向同一 G2 case-folder root 的 test。

## 禁止事项

1. 不运行旧 `1_rename_mri_data.py` 处理 G2 数据。
2. 不直接读取 G1 raw output。
3. 不把 locked test 用于训练、调参或 checkpoint 选择。
4. 不用不同 split 比较 real-only 与 real+synth。
5. 不把内部 Dice/HD95/AUC 当作官方最终榜单结果。
