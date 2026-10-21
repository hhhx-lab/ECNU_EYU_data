# BraTS2026 S2 RC Segmentation

nnU-Net v2 `3d_fullres` segmentation with RC-aware loss weighting.

## Current Baseline

```text
mode                   current
dataset                Dataset263_BraTS2026_MET_RealOnly_Current
train/val/test         823/103/104
nnU-Net key            fold_0 (API/storage only)
cross-validation       disabled
```

Run from the project root:

```bash
mkdir -p logs
PREP_JOB=$(sbatch --parsable --export=ALL,S2_EXPERIMENT_MODE=current \
  work_space/S2/slurm/legacy_realonly/04_s2_realonly_prepare_nyu.slurm)
sbatch --dependency=afterok:${PREP_JOB} --export=ALL,S2_EXPERIMENT_MODE=current \
  work_space/S2/slurm/legacy_realonly/04_s2_realonly_nyu.slurm
```

Do not add `--array`. Preparation checks the count contract, patient-group isolation, raw/preprocessed ID equality, and label integrity before training.

Historical Dataset260 recovery is available only through `S2_EXPERIMENT_MODE=legacy`. See `docs/S2_服务器运行手册.md`.

---

# 小病灶分割改进方案

## 背景

在无样本增强的 baseline 训练中，小病灶（< 50 voxels）分割结果明显差于大病灶。根因有两个层面：

1. **架构分辨率限制**：nnUNet 3d_fullres 默认 5 次下采样（32× 压缩），3³ (27 voxels) 的病灶到达 bottleneck 时只剩约 0.1 个特征单元，物理上被压消失。
2. **Loss 偏向大目标**：Dice + CE 按体素平均计算，一个 100,000 voxels 的大病灶贡献的 loss 量级是一个 27 voxels 小病灶的约 3700 倍。小病灶的梯度信号被物理淹没。

以下三个方案分别从架构层、层级层、体素层解决这两个问题。三者相互独立，可以叠加使用。

> 改动位置均为 `custom_nnunet/nnUNetTrainerBraTS2026RC.py`。

---

## 原始配置

strides `[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [1,2,2]]`，6 个 stage，5 次下采样。

```
层        stride     累积压缩         特征图体素占比
L0        [1,1,1]     1,   1,   1      100%
L1        [2,2,2]     2,   2,   2      12.5%
L2        [2,2,2]     4,   4,   4      1.6%
L3        [2,2,2]     8,   8,   8      0.2%
L4        [2,2,2]    16,  16,  16      0.024%
L5 (btl)  [1,2,2]    16,  32,  32      0.003%
```

3³ voxels (27) 的病灶到 bottleneck：各向 ≈ 0.1 个特征单元，物理消失。

实际是 4 层各向同性 + 1 层各向异性（`[1,2,2]`，x 轴保留 16× 不再压缩，y,z 压到 32×）。L5 占全图体素的 0.003%，计算量几乎为零。

---

## 方案 A-1：四层各向同性下采样

砍掉最深一层，末尾改为 `[2,2,2]`，三轴均等。

**strides**: `[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]`，5 个 stage，4 次下采样。

```
层        stride     累积压缩         特征图体素占比
L0        [1,1,1]     1,   1,   1      100%
L1        [2,2,2]     2,   2,   2      12.5%
L2        [2,2,2]     4,   4,   4      1.6%
L3        [2,2,2]     8,   8,   8      0.2%
L4 (btl)  [2,2,2]    16,  16,  16      0.024%
```

3³ voxels → bottleneck：三轴 ≈ 0.2 个特征单元。感受野 ≈ 125 voxels (ERF ≈ 31-42mm)。

**配置变更**：

```
n_stages              6 → 5
features_per_stage    [32, 64, 128, 256, 320, 320] → [32, 64, 128, 256, 320]
n_conv_per_stage      [2, 2, 2, 2, 2, 2] → [2, 2, 2, 2, 2]
n_conv_per_stage_dec  [2, 2, 2, 2, 2]    → [2, 2, 2, 2]
strides               砍掉末尾 [1,2,2]，其余不变
kernel_sizes          对应切片
```

**优点**：三轴对称、逻辑简单、y/z 轴分辨率各多得 2×。计算量 ≈ 0%（砍掉的是占比最小的最深一层）。

**代价**：y/z 轴感受野从 32× 降到 16×，但转移瘤 ERF 仍在 80mm 病灶的有效覆盖范围内。> 80mm 的病灶仅占 3.6%，且浅层跳跃连接保留了局部细节，分割结果退化大概率测不出来。

**实现**：覆写 `build_network_architecture`，建网络前截断所有 list 字段至 5 个 stage。约 15 行。

---

## 方案 A-2：四层下采样，保留末尾各向异性

保留 L4 的 `[1,2,2]` 作为 bottleneck，x 轴只到 8× 而非 16×。

**strides**: `[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [1,2,2]]`，5 个 stage。

```
层        stride     累积压缩         特征图体素占比
L0        [1,1,1]     1,   1,   1      100%
L1        [2,2,2]     2,   2,   2      12.5%
L2        [2,2,2]     4,   4,   4      1.6%
L3        [2,2,2]     8,   8,   8      0.2%
L4 (btl)  [1,2,2]     8,  16,  16      0.012%
```

3³ voxels → bottleneck：x=0.4, y/z=0.2 个单元。x 轴比 A-1 多保留 2× 分辨率。

| | A-1 (均等) | A-2 (保留各向异性) |
|---|---|---|
| 累积压缩 | `16, 16, 16` | `8, 16, 16` |
| x 轴分辨率 | 基准 | 高 2× |
| y/z 轴分辨率 | 相同 | 相同 |
| 三轴对称 | 是 | 否 |
| 感受野分布 | 三轴均等 | x 轴略小 |

A-2 保留了 nnUNet planner 的原始意图（x 轴体素最多、最后不压缩），两者差异不大。**拿不准优先选 A-1**，更简单。

---

## 方案 D：Deep Supervision 权重倾斜

**什么是 Deep Supervision**：UNet decoder 的 L0-L4 各输出一个分割预测，分别计算 loss 再加权求和。L0 是 patch 全分辨率（128×160×112），L4 是 bottleneck 附近的分辨率（16× 下采样）。5 个层级各有一个 loss，加权后一起反向传播。

**当前问题**：小病灶在 L4 的输出里只剩 1-2 个体素，该层对小病灶的监督几乎失去空间信息。当前实现的默认几何权重（非 DDP）约为
`[0.533, 0.267, 0.133, 0.067, 0.000]`，但 D 需要作为独立的浅层监督权重对照注册，不能把它误写成标签缩放尺度。

**改为浅层重、深层轻**：

```
当前（代码默认，几何衰减）:
  L0:w=0.533, L1:w=0.267, L2:w=0.133, L3:w=0.067, L4:w=0.000

D 候选（预注册，浅层监督）:
  L0:w=0.40, L1:w=0.30, L2:w=0.15, L3:w=0.10, L4:w=0.05
```

**效果**：梯度优先走浅层 encoder-decoder → 小病灶的体素级 loss 被放大 → 模型更快学会小病灶。大病灶不受影响 —— 它们在所有层级都有大量体素，浅层权重增高不会导致深层信号消失。

**实现约束**：保留 `_get_deep_supervision_scales` 返回的真实标签下采样尺度；在 `_build_loss`
中先构造与基座相同的 Dice + Focal CE，再使用
`DeepSupervisionWrapper(base_loss, [0.40, 0.30, 0.15, 0.10, 0.05])`。原版 6-stage
decoder 必须对应 5 个输出，权重长度、归一化和 forward shape 需用单元测试锁定。

---

## 方案 E：Focal Loss

**是什么**：在 CE loss 中，每个体素乘 `(1-pt)^γ`。`pt` 是 softmax 输出中 ground truth 类别的概率值（无需额外网络分支、无需额外标注），γ 通常取 2。

**怎么工作**：

```
pt = 0.99  (模型很确定这是对的)  →  (1-0.99)² = 0.0001  →  loss 压到万分之一
pt = 0.50  (模型拿不准)          →  (1-0.50)² = 0.25    →  loss 保留四分之一
pt = 0.01  (模型判反了)          →  (1-0.01)² = 0.98    →  loss 几乎完整保留
```

Focal loss 不包含任何"病灶尺寸"概念，只看置信度：pt 低 → 权重高，pt 高 → 权重低。小病灶之所以更受益，不是因为 focal "知道"它小，而是因为小病灶碰巧更难 —— 体素少、信号弱、训练初期被背景淹没，pt 长时间偏低，被 focal 持续加权。

**对大病灶的影响**：不退化。两个原因：

1. **Dice loss 不受影响**：Focal 只作用于 CE 侧。Dice 是全图 mask 的交并比，大病灶内数百体素的 Dice 梯度始终存在，确保了覆盖完整性不受损。两者叠加：CE+focal 管细节和难点，Dice 管整体轮廓。
2. **大病灶内部不需要反复学**：大病灶中心区域纹理均匀、类间对比大，几个 epoch 后 pt→0.95，focal 权重趋零。后面 900 多个 epoch 的梯度不再被这些已学会的体素占据，而非遗忘它们。Dice 还在维护整体轮廓，CE focal 权重虽趋零但不是严格为零。最坏情况反而是不放 focal —— 大病灶中心体素贡献 loss 的 99%，小病灶梯度被物理淹没。

**实现**：`_build_loss` 中加 `'focal': True`。约 1 行。

---

## 三者关系

```
方案A (架构层)   ─→ 小病灶在 bottleneck 从"物理消失"变为"物理存在"
方案D (层级层)   ─→ 浅层 decoder 的 loss 权重放大，梯度不再被深层稀释
方案E (体素层)   ─→ 已学会区域自动降权，难分体素（小病灶+边界）获得更多梯度
```

互不依赖、可以叠加、改动收敛在同一个 trainer 文件内。

**首轮实测结论（2026-07-24）**：E（Focal CE，`gamma=2`）冻结为小病灶方向的后续基座；B 保留为总体分割保守基线。A-1 的总体指标退化，A-1+E 未显示组合互补净收益。E 仍有 RC/小病灶漏检瓶颈，D 只作为 E 基座上的二阶段对照设计，尚未训练。

## 固定消融对比矩阵

不以理论推断直接冻结方案。先在同一 Dataset264、同一 `1035/103/104`
固定划分和相同超参数下完成下列对比：

| 候选 | Trainer | 变更 |
|---|---|---|
| B | `nnUNetTrainerBraTS2026RCCompletionFineTune` | 已完成的原版 baseline |
| C-A1 | `nnUNetTrainerBraTS2026RCA1CompletionFineTune` | 仅 A-1 |
| C-E | `nnUNetTrainerBraTS2026RCFocalCompletionFineTune` | 仅 E，`gamma=2` |
| C-A1E | `nnUNetTrainerBraTS2026RCA1FocalCompletionFineTune` | A-1 + E |

四者必须在同一固定 103 例上执行官方兼容评估，同时报告整体、RC、小病灶和
大病灶分层。首轮已完成，最终选择和 SHA256 见
`results/s2_small_lesion_ablation_20260721/checkpoint_selection.json`。
D 只能以 E 为基座追加：`_get_deep_supervision_scales` 控制标签缩放尺度，不是 loss 权重；
loss 权重必须在 `_build_loss` 中通过 `DeepSupervisionWrapper` 设置，并严格匹配原版
6-stage decoder 的 5 个输出。
