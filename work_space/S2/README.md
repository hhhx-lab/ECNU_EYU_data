# S2 工作区

## 整体任务

详见 [task_assignment.md](task_assignment.md)。

## 当前进度

real-only fold0 已完成 1000 epochs 和固定验证集推理。五折 Slurm Array、fold-aware Trainer、确定性 folds 1-4 划分和官方兼容指标任务已经补齐；下一步在服务器继续 folds 1-4。

## 本周计划

1. 运行一次 S2 real-only preparation job。
2. 以 Slurm Array 并行完成 folds 1-4。
3. 汇总五折 out-of-fold 预测并运行 BraTS_evaluation。
4. 在 G1 synthetic 通过 G2 QC 后建立独立 real+synthetic 消融实验。

## 提交记录

| 日期 | 内容 | 备注 |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |
