# S2 三候选阶段性对比（2026-07-23）

> **历史快照说明（2026-07-24）**：本文只记录 timeout 后、正式评估前的中间状态，
> 已被 `final_comparison_20260724.md` 和 `checkpoint_selection.json` supersede。
> 当前冻结选择为 E（小病灶后续基座）和 B（总体保守基线）；不要把本文的待评估结论
> 当作当前状态。

本报告对应 `S2_小病灶消融对比执行计划.md` 的首轮三条候选：A-1、E（Focal CE）和 A-1+E。
它记录 2026-07-23 timeout 后拉回的远端快照，不构成最终模型选择或官方兼容评估结果。

## 历史结论（已被正式评估 supersede）

截至 2026-07-23，当时不能在三者之间宣布优胜者。A-1 是唯一完成 200 epoch、生成固定 103 例预测和
`validation/summary.json` 的候选；E 与 A-1+E 在 epoch 175 有完整、可加载且有限的
checkpoint，但尚未完成训练或生成相同验证集预测。三者尚未在固定 103 例上完成
BraTS_evaluation mets 官方兼容评估及 tiny/small/large 分层，因此不满足计划中的选择条件。

阶段性训练信号提示 Focal 可能改善 RC（标签 4）的学习，但组合方案的 RC 信号波动明显，
不能据此宣称组合优于单项。继续保留 B 作为冻结基准，等待两条续跑完成后再进行正式比较。

## 快照完整性

| 候选 | 根作业 | 当前状态 | 可用 checkpoint | 验证产物 |
|---|---:|---|---|---|
| A-1 | `3128521` | `COMPLETED 0:0`，200 epoch | `checkpoint_final.pth`，SHA256 已核验 | 103 个预测、`summary.json`、完成标记 |
| E | `3128522` | `TIMEOUT 0:0`，日志进入 epoch 182 | epoch 175 `checkpoint_latest.pth`，SHA256 已核验 | 无预测、无 summary |
| A-1+E | `3128523` | `TIMEOUT 0:0`，日志进入 epoch 182 | epoch 175 `checkpoint_latest.pth`，SHA256 已核验 | 无预测、无 summary |

E 的续跑为 `3141629`，A-1+E 的续跑为 `3141630`。二者均使用 `S2_CONTINUE=auto`，
应从已审计的 epoch 175 checkpoint 接续，而不是新开训练。

## 同 epoch 训练态信号

下表全部来自 epoch 175 的 nnU-Net patch-level pseudo Dice。标签 4 是 RC。它们用于观察
训练趋势，不能替代逐病例的全量验证；尤其 E 的 Focal loss 与 A-1 的 Dice+CE loss 不在同一数值尺度。

| 候选 | epoch 175 pseudo Dice `[1, 2, 3, RC]` | RC 相对 A-1 | 最佳 EMA pseudo Dice | 最后完成 epoch 的 RC pseudo Dice |
|---|---|---:|---:|---:|
| A-1 | `[0.6958, 0.8963, 0.8198, 0.4236]` | 基准 | 0.7461 | 0.4409（epoch 199） |
| E | `[0.6147, 0.8867, 0.8250, 0.6389]` | +0.2153 | 0.7528 | 0.5695（epoch 181） |
| A-1+E | `[0.5905, 0.9039, 0.8457, 0.6035]` | +0.1799 | 0.7600 | 0.1644（epoch 181） |

可观察到：

- E 和 A-1+E 在保存 checkpoint 的 RC pseudo Dice 高于 A-1，值得完成续跑后做正式验证。
- A-1+E 的最佳 EMA 最高，但其最后一个已完成 epoch 的 RC 值降至 0.1644，显示波动风险；
  这不能作为“组合互补”的证据。
- E 的损失绝对值更低是 Focal 定义改变后的正常现象，不能拿它与 A-1 loss 直接比较。

## A-1 的完整内部验证结果

这是 A-1 的 nnU-Net `validation/summary.json`，不是 BraTS_evaluation mets 官方兼容结果：

| 指标 | 数值 |
|---|---:|
| 前景平均 Dice | 0.518328 |
| 标签 1 Dice | 0.481498 |
| 标签 2 Dice | 0.719954 |
| 标签 3 Dice | 0.688012 |
| RC（标签 4）Dice | 0.183849 |
| 前景平均 FN | 1713.6553 |
| 前景平均 FP | 1438.4199 |

RC Dice 很低，说明 A-1 即使完成，也仍存在明确的小病灶/RC 风险。该风险必须与 E、A-1+E
在同一 103 例的官方兼容 tiny/small/large、FN/FP 分层中一起复核，不能仅凭 A-1 的内部
summary 推断 Focal 会解决它。

## 已拉回产物

快照目录 `remote_snapshot_timeout_20260723T1435/` 包含：

- A-1：最终 checkpoint、SHA256、partial warm-start 审计、训练日志、103 个预测和 summary；
- E：epoch 175 checkpoint 和训练日志；
- A-1+E：epoch 175 checkpoint、partial warm-start 审计和训练日志；
- 三个根作业的 Slurm stdout。

未同步 Dataset264 cache、原始影像、在线 Diffusion 产物、官方 179 例推理或提交文件。

## 后续判定门槛（当时记录，现已完成）

只有在 E 与 A-1+E 完成 200 epoch、各自生成 103 个预测和 `summary.json` 后，才执行三候选
与原版 B 的固定 103 例 BraTS_evaluation mets 官方兼容评估。届时比较整体、RC、tiny/small/large、
FN/FP，并按执行计划的决策规则选择基座；在此之前不启动 Deep Supervision D。

正式门槛现已满足。最终结果：E 作为小病灶后续基座，B 作为总体保守基线；D 仅设计，未启动。
