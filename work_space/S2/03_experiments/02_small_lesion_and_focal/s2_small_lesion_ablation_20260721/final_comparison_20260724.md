# S2 小病灶消融正式比较与归档

生成时间：2026-07-24

> **证据绑定纠正（2026-08-14）**：旧版把 `B` 的 checkpoint 标为 Dataset264
> completion 普通 loss，却误引用了 Dataset263 real-only 的 `s2_eval_results` 指标。
> 本文现已将 `B` 重新绑定到
> `s2_completion_dataset264_t2w_20260720/official_style_eval`；原 Dataset263
> real-only 结果单独记为 `R`。本次只纠正既有结果的身份和汇总，不重跑训练或评估，
> 也不改写任何原始 CSV、Panoptica JSON 或 checkpoint。

## 结论

本轮在 Dataset264 的同一固定 103 例上完成了 B、A-1、E 和 A-1+E 的 BraTS-evaluation
`mets` 官方兼容评估。

- **总体分割的保守基线：B。** 在四个 Dataset264 候选中，B 的 WT lesionwise
  DSC/NSD 和 all-instance F1 最高，分别为 `0.593539`、`0.593146` 和 `0.723753`。
- **S2 小病灶方向的后续基座：E（Focal CE）。** “E 的 WT all-instance F1 最高”
  的旧表述不成立；纠正后 E 的主要优势是 WT small-instance F1 从 B 的 `0.330570`
  小幅提升到 `0.333083`，RC lesionwise DSC/NSD 分别从 `0.235939/0.212300`
  提升到 `0.377022/0.273118`，RC all-instance F1 从 `0.155556` 提升到
  `0.421053`，且 RC FP 从 `0.679612` 降至
  `0.038835`。
- **A-1 不选。** 相对 B，A-1 的 WT DSC/NSD、all-instance F1、small-instance F1
  均下降且 WT FP 增加；虽然 RC DSC/NSD 改善，但 RC F1 仍明显低于 E，整体收益不平衡。
- **A-1+E 不选。** 它的 WT small-instance F1 最高 (`0.342887`)，但 WT/RC 的总体
  分割指标明显下降，WT FP 升至 `1.310680`；其 RC 改善也不及 E 的 F1/FP 组合，
  不能证明两项改动具有互补净收益。

因此，后续 S2 小病灶实验应以 **E checkpoint** 为基座，同时保留 **B checkpoint** 作为
总体分割的对照。E 仍有明确的 RC/小病灶瓶颈，故只登记 Deep Supervision D 的二阶段设计，
本轮不启动 D 训练。

## 统一评估合同

| 项目 | 固定值 |
|---|---|
| 数据集 | `Dataset264_BraTS2026_MET_Completion` |
| 划分 | train/validation/locked test = `1035/103/104` |
| 评估病例 | 固定 validation `103` 例，四组 subject ID 完全一致 |
| 评估器 | `BraTS-evaluation==0.0.8`，`panoptica==2.1.0`，独立 `brats_eval` 环境 |
| 配置 | `mets` |
| instance 体积阈值 | `27 voxels` |
| overlap 阈值 | `0.2` |
| 输出 | 每组 prediction/reference `103/103`，`missings=[]`，完成标记存在 |

官方 `mets` 输出只提供 `small_instance` 和 `large_instance` 两档；本轮没有独立的
`tiny` 指标，因此不把 `small_instance` 改名为 `tiny`，也不伪造 tiny 数值。

## 官方兼容均值

以下 `WT` 作为总体病灶区域，`RC` 为标签 4。DSC/NSD 是 lesionwise 均值；FN/FP 是
all-instance 的每病例均值。四列均绑定 Dataset264 completion 候选；Dataset263
real-only 不再冒充 `B`。

### WT 总体

| 指标 | B | A-1 | E | A-1+E |
|---|---:|---:|---:|---:|
| lesionwise DSC | **0.593539** | 0.580821 | 0.587819 | 0.578306 |
| lesionwise NSD | **0.593146** | 0.569390 | 0.585794 | 0.567496 |
| all-instance F1 | **0.723753** | 0.683197 | 0.712277 | 0.657227 |
| small-instance F1 | 0.330570 | 0.292404 | 0.333083 | **0.342887** |
| large-instance F1 | 0.656773 | 0.662586 | **0.672686** | 0.660221 |
| all FN | 3.029126 | 2.873786 | 2.912621 | **2.834951** |
| all FP | **0.893204** | 1.097087 | 0.970874 | 1.310680 |

### RC

| 指标 | B | A-1 | E | A-1+E |
|---|---:|---:|---:|---:|
| lesionwise DSC | 0.235939 | **0.421768** | 0.377022 | 0.380900 |
| lesionwise NSD | 0.212300 | **0.386169** | 0.273118 | 0.332156 |
| all-instance F1 | 0.155556 | 0.248505 | **0.421053** | 0.194070 |
| small-instance F1 | 0.000000 | 0.000000 | 0.000000 | **0.166667** |
| large-instance F1 | 0.055016 | **0.103098** | 0.080906 | 0.086916 |
| all FN | 0.097087 | 0.067961 | 0.116505 | **0.058252** |
| all FP | 0.679612 | 0.611650 | **0.038835** | 0.689320 |

RC small-instance F1 的可比较病例很少：固定 103 例中只有 3 例存在可用于该分层的
RC small-instance reference。A-1+E 的 `0.166667` 由其中一个病例的有限改善贡献，
不能视为已解决 RC 小病灶问题。

## 风险复核

逐病例复核见 [risk_review_20260724.md](risk_review_20260724.md)。主要发现：

1. WT 漏检集中在同一组困难病例。`BraTS-MET-01351-002` 的 WT FN 为
   `98/92/94/92`（B/A-1/E/A-1+E），`BraTS-MET-00014-000` 为 `32/32/32/33`。
   这说明均值改善没有消除高负荷病例的漏检风险。
2. E 的 RC 变化仍包含 precision-recall 交换：RC FP 从 B 的 `0.679612` 降到
   `0.038835`，RC FN 从 `0.097087` 升到 `0.116505`；但纠正绑定后，E 的 RC
   lesionwise DSC/NSD 和 all-instance F1 均高于 B。
3. A-1+E 的 WT FP 为四组最高，WT 主指标明显退化；其 RC F1/FP 又劣于 E，不能以
   WT/RC small-instance F1 单项最高为由采用组合。
4. E 相对 B 的 WT small-instance F1 在可比较的 46 例中改善 7 例、持平 32 例、下降
   7 例；均值提升仅 `0.002513`。RC small-instance 只有 3 例可比较且 B/A-1/E 均为
   0，不能单凭该分层支持 E。

## Real-only 外部训练数据对照

`R` 使用 Dataset263 的 823 个真实训练病例；`B/E` 使用 Dataset264 的 823 个真实病例
加 212 个缺失模态补全病例。三组评估病例 ID 均为同一固定 103 例，评估器和阈值一致，
但训练数据不同，因此 `R` 是数据消融对照，不属于上述四个 Dataset264 损失/架构候选。

| 指标 | R：real-only | B：completion + 普通 loss | E：completion + Focal |
|---|---:|---:|---:|
| WT lesionwise DSC | **0.625729** | 0.593539 | 0.587819 |
| WT lesionwise NSD | **0.621359** | 0.593146 | 0.585794 |
| WT all-instance F1 | 0.710078 | **0.723753** | 0.712277 |
| WT small-instance F1 | 0.277048 | 0.330570 | **0.333083** |
| RC lesionwise DSC | **0.423506** | 0.235939 | 0.377022 |
| RC lesionwise NSD | **0.363712** | 0.212300 | 0.273118 |
| RC all-instance F1 | 0.236232 | 0.155556 | **0.421053** |
| RC FP/病例 | 0.776699 | 0.679612 | **0.038835** |

原始指标路径：

- R：`work_space/S2/results/s2_eval_results/leaderboard_metrics.csv`
- B：`work_space/S2/results/s2_completion_dataset264_t2w_20260720/official_style_eval/leaderboard_metrics.csv`
- E：`work_space/S2/results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/focal/leaderboard_metrics.csv`

## Checkpoint 选择与产物

选择、路径、SHA256 和完整性合同已写入
`checkpoint_selection.json`。本轮有效训练和评估作业为：

| 候选 | 训练作业 | 评估作业 | 200 epoch/产物 |
|---|---:|---:|---|
| A-1 | 3128521 | 3154513 | 通过，103 预测、完成标记、partial warm-start 审计 |
| E | 3141629 | 3154514 | 通过，103 预测、完成标记 |
| A-1+E | 3141630 | 3154517 | 通过，103 预测、完成标记、partial warm-start 审计 |

旧失败作业、旧 checkpoint、Dataset264 cache 和所有本轮预测均保留；未启动在线
Diffusion、官方 179 例推理或提交。

## D 二阶段设计，仅设计不启动

由于选定的 E 仍有 RC lesionwise/小病灶瓶颈，允许进入 D 的设计阶段，但本轮不写入新的
训练结果：

1. **基座固定为 E。** 继续使用原版 6-stage architecture、`gamma=2.0`、RC class
   weight `3.0`、同一 `1035/103/104` split 和同一 103 例评估合同；不得把 A-1 架构
   再混入 D。
2. **只改 loss 层级权重。** 在 `_build_loss` 中先构造与 E 相同的 Dice + Focal CE，
   再用 `DeepSupervisionWrapper` 包装。不要把 loss 权重写入
   `_get_deep_supervision_scales`，后者只负责标签下采样尺度。
3. **输出数量严格匹配。** 原版 6-stage decoder 有 5 个 deep-supervision outputs，
   因此 D 的权重向量必须长度为 5，并与预测输出/下采样 target 一一对应。建议预注册
   `[0.40, 0.30, 0.15, 0.10, 0.05]`，总和为 1；启动前必须用单元测试检查长度、归一化和
   forward shape。
4. **隔离与验收。** 新 trainer、新结果目录和新评估目录均不得覆盖 E。至少完成一个
   200 epoch 对照、103 个预测、官方兼容评估和同样的 RC/small/large/FN/FP 风险复核后，
   才能决定 D 是否替换 E。

## 归档状态

- 正式比较：完成。
- checkpoint 选择：完成，E 为小病灶后续基座，B 为总体保守基线。
- D：已设计，未启动。
- S2 自动监控：本次归档后删除。
