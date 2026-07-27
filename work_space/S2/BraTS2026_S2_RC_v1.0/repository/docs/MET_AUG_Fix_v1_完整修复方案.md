# BraTS 2026 S2 MET-AUG Fix-v1 完整修复方案

## 0. 文档状态

| 项目 | 值 |
|---|---|
| 文档类型 | 技术设计与验收方案 |
| 版本 | Fix-v1 design v1 |
| 日期 | 2026-07-27 |
| 当前状态 | 仅设计，未授权实施 |
| 当前比赛主线 | 保持 MET-AUG 停止，继续 B / 原 E / E-continue 比较与官方 179 例推理 |
| 实施边界 | 必须使用独立 root，不得修改、删除或覆盖 R4/R5 及其人工决策 |

> 本文档定义赛后或拥有独立 39–54 小时窗口时的修复路线。在新路线通过全部门禁前，不得使用任何 MET-AUG 产物训练当前 S2 模型。

## 阅读导航

- [失败证据与不可变基线](#2-背景与失败证据)
- [当前硬替换机制与根因](#4-当前机制与根因)
- [Fix-v1 详细算法](#6-详细算法)
- [代码改造范围](#8-代码改造范围)
- [测试计划](#10-测试计划)
- [重新门禁流程](#11-重新门禁流程)
- [训练对照和选模原则](#12-训练对照设计)
- [验收标准总表](#13-验收标准总表)
- [实施清单](#18-实施清单)
- [核心伪代码](#附录-a核心伪代码)

## 1. 执行摘要

R4 和 R5 均完成了自动数值检查，但在人工视觉复核中发现了明显的支撑区边界接缝、黑白极端信号和块状伪影。失败的根因不是单个供体组件，而是当前插入逻辑对四个 MRI 通道执行二值支撑区硬替换。

Fix-v1 采用以下组合修复：

1. 将标签区域和图像过渡区域分离。
2. 标签仍保持离散整数并硬写入。
3. 图像使用物理尺度定义的余弦 alpha 羽化融合。
4. 融合前进行逐通道局部鲁棒强度对齐。
5. 提交前执行边界、极值、块状和跨模态自动 QC。
6. 任意 QC 失败时记录 `NO_OP`，不重试、不强制提交。
7. 使用训练集校准阈值，固定 103 例验证集和官方 179 例严禁参与阈值选择。
8. 通过 100,000 次确定性 Gate-1、全新 48 例 Gate-2、48/48 人工通过和 training smoke 后，才允许训练。

## 2. 背景与失败证据

### 2.1 R4

- 路由：`s2_met_aug_route_a_20260726_r4`
- Gate-2 自动 QC：24/24 通过
- 人工复核：19 accept / 5 reject
- 拒绝病例：`004` / `009` / `016` / `021` / `023`
- 典型问题：支撑区边界接缝、极端块状信号，以及 T1n 三平面的近黑硬边界。

### 2.2 R5 compact-support

- 路由：`s2_met_aug_route_a_20260726_r5`
- 额外限制：`total_support_voxels <= 4096`
- 额外限制：`total_support_voxels / core_voxels <= 20`
- 组件池：4015 总组件，3554 合格，461 排除
- Gate-2 自动 QC：24/24 通过
- 人工复核：18 accept / 6 reject
- 拒绝病例：`007` / `009` / `010` / `012` / `015` / `021`
- 典型问题：支撑区对齐的近黑块、饱和白带、T2f 极端亮区和矩形/棱柱状多模态信号。

### 2.3 已得结论

R5 已排除 R4 的已知失败类型，但仍出现了 6 个新失败病例。因此：

- 不能通过黑名单删除 5 或 6 个组件解决。
- 不能仅依赖 compact-support 阈值。
- 不能将 18 个已通过固定样本等同于整个在线随机分布已通过。
- 必须修复插入和提交机制本身。

### 2.4 不可变证据与基线

本文档所依赖的失败证据和基线必须保持不变。

#### 远端证据

```text
R4: /root/brats2026/runs/s2_met_aug_route_a_20260726_r4
R5: /root/brats2026/runs/s2_met_aug_route_a_20260726_r5
true-1mm cache: /root/brats2026/data/s2_dataset264_true1mm_20260726_r1
E-continue: /root/brats2026/runs/s2_e_continue_fallback_20260726_r1
```

#### 本地视觉与人工决策证据

```text
work_space/S2/results/s2_met_aug_route_a_20260726_r4/gate2_run/
work_space/S2/results/s2_met_aug_route_a_20260726_r5/gate2_run/
```

| 证据 | SHA256 |
|---|---|
| R4 `manual_review_decisions.csv` | `323183f3fc15a9992547dfafd41437ef5a542d99241f77d37617618dcd27f817` |
| R4 `manual_review_template.csv` | `aea3c1f010769efc57090afc074a2b1933a273882427e114e5b637399f60f9a2` |
| R5 `manual_review_decisions.csv` | `efcf272a66da89ed36704b52b5af9e2e0a0e08e16c749cdb60014e51c8a4e510` |
| R5 `manual_review_template.csv` | `b3d46f7b68e21ec1647524f4776059b15964e8988ccf243c38d5c53a4359ac35` |

#### 模型基线

| 模型 | SHA256 | 角色 |
|---|---|---|
| 原 E | `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267` | 始终可部署回退 |
| E-continue final | `535e89644121a0c0f1f591f0c1a211581d6d3dd6c1df334a7ccb1bb7825328b1` | 当前无增强继续训练候选 |

Fix-v1 不得原地写入上述任何目录。若证据 SHA 与本节不一致，必须先停止并调查漂移，不得继续开发或门禁。

## 3. 目标与非目标

### 3.1 目标

1. 消除人眼可见的支撑区边界接缝。
2. 拒绝近黑、饱和白带、块状和平面状异常信号。
3. 保持标签为离散整数 `{-1,0,1,2,3,4}`。
4. 保持训练 split、seed、G1/G2 checkpoint、EDM-Heun/18、FP32 和 `p=0.20` 的可比性。
5. 保持事件可确定重放、可审计和串/并行等价。
6. 任何无法证明合格的增强事件必须安全退化为 `NO_OP`。

### 3.2 非目标

1. 不修改 R4/R5 人工决策。
2. 不用固定 103 例验证集校准 QC 阈值。
3. 不使用官方 179 例无标签数据参与任何开发或选模。
4. 不通过降低人工门禁、改写 reject 或只选有利样本获得通过。
5. 不对标签做线性、三次或任何连续值插值。
6. 不在当前比赛主线中边评估边调参。

## 4. 当前机制与根因

### 4.1 当前硬替换

当前实现位于 [`custom_nnunet/met_aug_core.py`](../custom_nnunet/met_aug_core.py)：

```python
support = placement.support
for channel in range(4):
    updated = draft_image[(channel,) + slices]
    updated[support] = generated[channel][support]

seg_crop = draft_segmentation[(0,) + slices]
seg_crop[support] = placement.label_cube[support]
```

`support` 是一个二值 mask。支撑区内的图像来自生成器，支撑区外的图像来自目标病例，两者之间没有过渡。

### 4.2 当前自动 QC 为何未捕获

当前 `_validate_commit()` 主要验证：

- 图像全部 finite。
- 图像和标签不在 `support` 外变化。
- 标签值仍在合法集合内。
- 写入标签与计划标签逐体素一致。

这些检查能防止数值和几何错误，但不能判断 MRI 信号是否自然。因此“自动 pass + 人工 reject”不矛盾。

## 5. Fix-v1 总体设计

```text
选择供体与放置位置
  -> 构造离散标签区 label_support
  -> 扩张得到图像过渡区 image_support
  -> 调用固定 Diffusion backend
  -> 逐通道局部强度对齐
  -> 余弦 alpha 羽化融合图像
  -> 硬写入离散标签
  -> 边界/极值/块状/跨模态 QC
  -> COMMITTED 或 NO_OP
  -> 追加不可覆盖事件审计
```

## 6. 详细算法

### 6.1 分离标签支撑区和图像支撑区

定义：

- `label_support = placement.label_cube != 0`
- `image_support = dilate(label_support, blend_radius_mm)`
- `context_ring = dilate(label_support, context_outer_mm) - dilate(label_support, context_inner_mm)`

推荐初始几何参数：

| 参数 | 初始值 | 说明 |
|---|---:|---|
| `blend_radius_mm` | 3.0 mm | 图像过渡带宽度 |
| `context_inner_mm` | 4.0 mm | 强度对齐环带内径 |
| `context_outer_mm` | 8.0 mm | 强度对齐环带外径 |
| `minimum_context_voxels` | 512 | 局部对齐最小样本数 |

虽然当前 true-1mm cache 的体素间距为 1 mm，实现仍必须用物理距离构造 mask，不得将“3 个体素”硬编码成普遍约定。

### 6.2 余弦 alpha 羽化

对于体素 `x`，令 `d(x)` 为其到 `label_support` 的物理距离，`r` 为 `blend_radius_mm`：

```text
alpha(x) = 1                                      x in label_support
alpha(x) = 0.5 * (1 + cos(pi * d(x) / r))         0 < d(x) < r
alpha(x) = 0                                      d(x) >= r
```

图像逐通道融合：

```python
candidate[channel] = (
    original[channel]
    + alpha * (harmonized_generated[channel] - original[channel])
)
```

标签始终使用硬写入：

```python
candidate_segmentation[label_support] = label_cube[label_support]
```

不允许对标签使用 alpha、线性插值或任何浮点过渡。

### 6.3 逐通道局部强度对齐

对每个 MRI 通道独立处理。在 `context_ring` 内计算：

```text
gain = MAD(original) / MAD(generated)
offset = median(original) - gain * median(generated)
harmonized_generated = gain * generated + offset
```

安全限制：

- `gain` 非 finite 时拒绝事件。
- `MAD(generated)` 过小时拒绝事件。
- 环带体素不足时拒绝事件。
- `gain` 超过训练集校准范围时拒绝，不做无限制 clamp。
- 不根据固定 103 例或官方 179 例调整对齐参数。

`gain` 的建议初始开发范围为 `[0.67, 1.50]`，但最终值必须由训练集校准并写入不可改的 calibration JSON。

### 6.4 放置安全条件

不再仅检查 `label_support`。放置必须同时满足：

1. `label_support` 完全位于有效脑组织内。
2. `image_support` 完全位于有效脑组织内。
3. `label_support` 不覆盖目标病例已有病灶。
4. `image_support` 不与已有病灶或禁止区域重叠。
5. `context_ring` 具有足够有效体素。
6. 扩张区域不超出 crop 边界。

任一条件不满足时，返回 `NO_OP / BLEND_PLACEMENT_INVALID`。

### 6.5 提交前 QC

提交前 QC 使用融合后 candidate，不得仅检查原始 generated crop。

#### A. 基本数值和几何

- 四通道全部 finite。
- 形状、dtype 和坐标合同不变。
- 标签仅位于合法集合。
- 图像变化仅位于 `image_support`。
- 标签变化仅位于 `label_support`。

#### B. 边界连续性

逐通道计算：

- 过渡区外边界梯度的 p95 / p99。
- 边界内外的鲁棒强度差。
- 候选图像与原图在过渡区外缘的最大绝对差。
- 与真实训练病灶边界分布归一化后的 seam score。

#### C. 内部信号合理性

- `label_support` 内强度的鲁棒 z-score 分布。
- 近黑体素占比。
- 近饱和白体素占比。
- 连续平面状极值区域的面积和厚度。
- 边界对齐块状区域的体积、矩形度和表面积/体积比。

#### D. 跨模态合理性

- T1n / T1c / T2w / T2f 的支撑区内效应量。
- 任一通道与其他通道出现方向相反且超出真实训练分布的极端信号。
- 跨模态块状边界的空间重合率。

### 6.6 QC 决策

```text
基本合同失败          -> NO_OP / COMMIT_CONTRACT_FAIL
强度对齐失败        -> NO_OP / HARMONIZATION_FAIL
边界连续性失败      -> NO_OP / BOUNDARY_QC_FAIL
内部极值或块状失败  -> NO_OP / INTENSITY_QC_FAIL
跨模态失败          -> NO_OP / CROSS_MODAL_QC_FAIL
全部通过              -> COMMITTED
```

QC 失败后不更换供体重试。重试会改变有效采样分布、隐藏失败率并导致不可预测的训练延迟。

## 7. 配置合同

新路由配置必须升级 schema，不得在 R5 配置上就地添加默认值。示例：

```json
{
  "schema_version": 4,
  "route": "met_aug_fix_v1",
  "seed": 20260725,
  "probability": 0.20,
  "diffusion": {
    "sampler": "EDM-Heun",
    "steps": 18,
    "precision": "FP32"
  },
  "blend": {
    "policy": "label_hard_image_cosine_v1",
    "blend_radius_mm": 3.0,
    "context_inner_mm": 4.0,
    "context_outer_mm": 8.0,
    "minimum_context_voxels": 512
  },
  "harmonization": {
    "policy": "per_channel_median_mad_v1",
    "gain_min": 0.67,
    "gain_max": 1.50,
    "on_failure": "no_op"
  },
  "commit_qc": {
    "policy": "seam_intensity_cross_modal_v1",
    "calibration_path": "calibration/FROZEN_COMMIT_QC_CALIBRATION.json",
    "on_failure": "no_op",
    "retry_count": 0
  }
}
```

最终运行时必须同时绑定：

- route config SHA256
- calibration JSON SHA256
- component manifest SHA256
- valid-mask manifest SHA256
- G1 checkpoint selection SHA256
- G2 parent gate SHA256
- 参与推理的所有源码 SHA256

## 8. 代码改造范围

### 8.1 `custom_nnunet/met_aug_core.py`

新增：

- `BlendGeometry`
- `HarmonizationResult`
- `CommitQCResult`
- `_build_blend_geometry(...)`
- `_harmonize_generated_crop(...)`
- `_compute_boundary_qc(...)`
- `_compute_intensity_qc(...)`
- `_compute_cross_modal_qc(...)`
- `_blend_generated_crop(...)`

修改：

- `Placement` 或提交元数据必须区分 `label_support` 和 `image_support`。
- `apply()` 必须先生成 candidate，再执行 QC，最后提交。
- `_validate_commit()` 分别校验图像变化区和标签变化区。
- `audit_mapping()` 记录 alpha/QC/强度对齐统计。

### 8.2 Gate 脚本

需要升级：

- route config 生成和 schema 验证。
- Gate-1 事件审计字段和报告汇总。
- Gate-2 NPZ 必须包含 `label_support`、`image_support`、`alpha`、融合前/后图像和 QC 指标。
- montage 必须同时显示原图、原始 generated、融合后 candidate、标签区和过渡区边界。

### 8.3 训练器

训练器仅接收 `COMMITTED` 事件。`NO_OP` 必须返回原始图像和标签，不允许部分修改后回退。

## 9. 校准数据与防泄漏约束

### 9.1 允许使用

- 固定训练 split 中的 1035 例。
- 只读复用已验收的 true-1mm train component pool。
- R4/R5 的 11 个 reject 作为已知缺陷回归集。
- R4/R5 的 37 个 accept 作为误拒率回归集。

### 9.2 禁止使用

- 固定 103 例 validation。
- 锁定 104 例 internal test。
- 官方 179 例 validation。
- 任何人工查看了评估结果后才新增的有利阈值。

### 9.3 阈值冻结

所有 QC 阈值必须写入：

```text
calibration/FROZEN_COMMIT_QC_CALIBRATION.json
calibration/FROZEN_COMMIT_QC_CALIBRATION.ok
```

内容必须包括：

- 训练病例清单和 SHA256。
- 真实病灶边界指标分布。
- 每个通道的阈值和分位数来源。
- 已知 accept/reject 回归结果。
- 校准代码 SHA256。
- 生成时间和环境版本。
- 整个校准 payload 的 identity SHA256。

阈值冻结后，不得根据新 Gate-2 结果原地调整。任何调整都必须生成全新实验 root 和新版本号。

## 10. 测试计划

### 10.1 单元测试

1. `alpha` 仅在 `[0,1]` 内。
2. `alpha=1` 覆盖全部 `label_support`。
3. `alpha=0` 覆盖 `image_support` 外部。
4. alpha 沿距离方向单调不增。
5. 标签不产生浮点或新类别。
6. 图像在 `image_support` 外逐体素不变。
7. 标签在 `label_support` 外逐体素不变。
8. 相同 seed/输入/配置产生字节一致结果。
9. 脑外、crop 边缘和已有病灶重叠放置被拒绝。
10. 无有效 context ring 时安全 `NO_OP`。
11. 审计写入失败时整个事务不提交。

### 10.2 合成极端测试

构造并确认以下 candidate 会被拒绝：

- 支撑区全黑。
- 支撑区全白。
- 单一模态饱和白带。
- 多模态矩形块。
- 边界对齐的平面状信号跳变。
- 包含 NaN / Inf。
- 跨模态空间错位。

### 10.3 R4/R5 回归门禁

- 11/11 已知 reject 必须被自动拒绝或修复后达到无可见伪影。
- 已知 reject 不得在没有新人工复核的情况下被自动重标为 accept。
- 37 个已知 accept 的保留率建议至少 90%。
- 若“11/11 捕获”与“合格保留率”无法同时满足，必须返回算法设计，不得选择性降低门禁。

### 10.4 确定性与并行等价

- 使用真实资产先运行小规模串行/并行等价。
- 相同事件索引必须派生相同 seed。
- 事件 JSONL 顺序和字节必须一致。
- 汇总报告必须字节一致。
- 并行 worker 不得直接竞争写入最终 JSONL。

### 10.5 性能测试

- 单事件羽化+强度对齐+QC 的 CPU 时间。
- 训练 step 墙钟和 GPU 利用率。
- 峰值 RAM/VRAM。
- `NO_OP` 率和实际 `COMMITTED` 率。
- 相对已有 Route A 的额外墙钟不得超过 15%。

## 11. 重新门禁流程

### 11.1 Gate-0：静态回归与配置冻结

验收：

- 全部单元测试通过。
- R4/R5 回归报告通过。
- calibration JSON 已冻结并绑定 SHA256。
- 代码和运行时快照已归档。

### 11.2 Gate-1：100,000 次确定性事件

保持：

- 相同固定 seed。
- 恰好 100,000 个事件。
- 确定性多进程分片。
- 最终 JSONL 按事件索引合并。
- 无 worker failure、无非有限统计、无残留 shard。

Gate-1 主要验证采样、放置、mask 几何、配置绑定和审计完整性，不用便宜替代模型伪装 Diffusion 视觉质量。

### 11.3 Gate-2：全新 48 例 Diffusion 视觉门禁

采样设计：

- 48 个唯一 target。
- 48 个唯一 donor component。
- 三个 core-volume 分档各 16 例。
- 目标和供体必须全部来自训练 split。
- 不使用 R4/R5 的固定 24 例 manifest。
- 正式 EDM-Heun / 18 steps / FP32。

自动验收：

- 48 NPZ / 48 montage / 48 COMMITTED events。
- 四通道 finite。
- 标签、shape、几何、mask 和 SHA 绑定全部合法。
- 图像只在 `image_support` 中变化。
- 标签只在 `label_support` 中变化。
- 自动 QC 无 violation。

人工验收：

- 必须全部 48/48 真实逐例复核。
- 必须 48/48 accept。
- 任意接缝、脑外生成、标签错位、通道错误、极值或块状伪影均为 reject。
- 任一 reject 即本版本失败，不得原地修改决策或阈值。

### 11.4 Training smoke

要求：

- 不少于 32 个真实 training step。
- 至少 4 个 `COMMITTED` 增强事件。
- 所有 loss finite。
- 无 OOM / NaN / Inf / Traceback。
- 无 validation。
- 无 checkpoint 写入。
- 无未审计生成资产落盘。
- 显存、GPU 利用率和吞吐正常。
- 根据 smoke 实测重新估算 200 epoch 墙钟。

## 12. 训练对照设计

### 12.1 两臂

| 训练臂 | 初始 checkpoint | 增强概率 | 作用 |
|---|---|---:|---|
| Control | 同一冻结 E | 0 | 排除继续训练本身的收益 |
| Fix-v1 | 同一冻结 E | 0.20 | 测量修复后增强的净收益 |

其他必须完全一致：

- 固定 split：1035 / 103 / 104。
- seed。
- trainer 中除 augmentation 外的源码。
- lr = 0.001。
- focal gamma = 2。
- 200 epoch。
- save every 25 epoch。
- compile = 0。
- 单 GPU，不使用 DDP。

两张 H20 各运行一臂，两臂输出 root 完全隔离。

### 12.2 固定 103 例评估

使用同一版本的：

- `BraTS-evaluation == 0.0.8`
- `panoptica == 2.1.0`
- `NumPy == 1.26.4`
- `config = mets`
- `vol_threshold = 27`
- `overlap_threshold = 0.2`

汇报：

- ET / RC / TC / WT lesion-wise DSC。
- ET / RC / TC / WT lesion-wise NSD。
- all-instance F1。
- small-instance F1。
- large-instance F1。
- FN / FP。
- 103 例逐例配对差。
- paired bootstrap 置信区间。

官方 evaluator 不提供 tiny bin 时，必须记录 `tiny_metric_available=false`，不得自行构造 tiny 指标。

### 12.3 选模原则

预先冻结下列决策规则：

1. Fix-v1 在 WT/TC 主要 DSC、NSD 或 all-instance F1 上不得出现大于 0.01 的绝对下降。
2. Fix-v1 必须在至少两个预注册目标指标上改善。
3. 目标指标优先包括 small-instance F1、RC DSC/NSD/F1 和 FN。
4. 不得用单一有利指标掩盖整体性能下降。
5. 选择结论必须绑定两个 checkpoint、评估 CSV、参考清单和环境 SHA。

0.01 是建议的初始安全界值。若研究负责人要修改，必须在训练启动前写入选模合同，不得看到结果后更改。

## 13. 验收标准总表

| 阶段 | 必须满足 | 失败处置 |
|---|---|---|
| 代码测试 | 全部单测通过，无决定性漂移 | 停止，修代码 |
| R4/R5 回归 | 已知 reject 11/11 捕获，accept 保留率 >=90% | 停止，重做设计 |
| Gate-1 | 100000/100000，0 violation，串/并行等价 | 保留证据，不提升 |
| Gate-2 自动 | 48 产物完整，0 violation | 保留证据，不人工硬通过 |
| Gate-2 人工 | 48/48 accept | 本版本失败 |
| Training smoke | >=4 COMMITTED，loss finite，无 checkpoint/validation | 停止训练 |
| 正式训练 | 200 epoch，全 finite，合同无漂移 | 保留证据，不在同 root 重跑 |
| 103 例评估 | 覆盖完整，0 missing/error | 不选 Fix-v1 |
| 选模 | 满足预注册净收益规则 | 回退 Control/原 E |

## 14. 审计和目录规划

实施时使用全新独立 root，示例：

```text
/root/brats2026/runs/s2_met_aug_fix_v1_YYYYMMDD_r1/
  config/
  calibration/
  regression/
  gate1/
  gate2_run/
    artifacts/
    montages/
    manual_review_template.csv
    manual_review_decisions.csv
  training_smoke/
  control_train/
  fix_v1_train/
  evaluation/
  selection/
  logs/
  runtime/
```

每个阶段必须产生：

- 启动合同。
- 完成 marker。
- 输入和输出 manifest。
- 源码、配置、checkpoint 和资产 SHA256。
- 运行环境版本。
- PID、启动时间、完成时间和墙钟。
- 完整的失败原因和未被覆盖的旧证据路径。

R1–R5、原缓存、true-1mm 缓存和人工决策必须保持只读语义。不得为节省磁盘删除失败证据。

## 15. 失败、回滚与停止规则

1. 任一阶段失败时停在当前门禁。
2. 不在同一 root 删除输出后重跑。
3. 不改写人工 reject。
4. 不在看到新 Gate-2 后原地改阈值。
5. 修改算法或阈值时，必须新建版本和 root。
6. 新版本连续两次 Gate-2 人工失败时，停止该技术路线，不继续逐例打补丁。
7. Fix-v1 评估不达标时回退 Control/原 E，不用官方 179 例进行二次选模。

## 16. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 羽化后图像/标签不匹配 | 标签边缘训练噪声 | `label_support` 内 alpha=1，过渡只向标签外扩展 |
| 强度对齐削弱病灶特征 | 增强价值降低 | 仅用外围环带估计仿射参数，异常参数直接 NO_OP |
| QC 过严 | 有效增强率过低 | 报告 attempted/committed/no-op 率，在冻结前用 train-only 数据校准 |
| QC 过松 | 伪影进入训练 | 已知 reject 11/11 回归 + 全新 48/48 人工门禁 |
| 过拟合 R4/R5 | 新样本仍失败 | R4/R5 仅作回归，最终依赖全新分层 Gate-2 |
| 训练吞吐下降 | 训练时间不可控 | smoke 实测和 15% 额外墙钟上限 |
| 重试导致采样偏差 | 归因失效 | QC 失败直接 NO_OP，retry=0 |
| 继续查看验证集调参 | 信息泄漏 | 校准只使用 1035 训练例，阈值 SHA 冻结 |

## 17. 工期和资源估算

| 阶段 | 估计墙钟 | 主要资源 |
|---|---:|---|
| 代码改造和单元测试 | 6–8 h | CPU |
| train-only 校准与 R4/R5 回归 | 2–4 h | CPU，可少量使用 GPU |
| 100,000 次并行 Gate-1 | 1–2 h | 16+ CPU process |
| 48 例 Gate-2 | 5–10 min 计算 + 人工时间 | 1 H20 |
| Training smoke | 0.5–1 h | 1 H20 |
| 两臂 200 epoch 并行训练 | 25–31 h | 2 H20 |
| 固定 103 例评估与选模 | 3–5 h | GPU 推理 + CPU MET evaluator |
| 审计归档 | 1–2 h | CPU |

总计预留：39–54 小时。该估算不包含官方 179 例最终推理。

## 18. 实施清单

### 设计与代码

- [ ] 建立独立 Fix-v1 root，不覆盖 R4/R5。
- [ ] 冻结输入资产和当前源码 SHA。
- [ ] 实现 label/image support 分离。
- [ ] 实现物理距离 alpha 羽化。
- [ ] 实现逐通道 median/MAD 强度对齐。
- [ ] 实现边界、极值、块状和跨模态 QC。
- [ ] 更新事务提交验证和审计 schema。
- [ ] 更新 Gate-2 NPZ 和 montage。

### 测试与校准

- [ ] 完成单元、属性、极值和审计失败测试。
- [ ] 使用 1035 train-only 病例生成真实边界分布。
- [ ] 生成并冻结 calibration JSON。
- [ ] 通过 R4/R5 已知 accept/reject 回归。
- [ ] 证明串行/并行逐事件字节一致。

### 门禁

- [ ] Gate-0 通过。
- [ ] Gate-1 100000/100000 通过。
- [ ] 全新 48 例 Gate-2 自动 QC 通过。
- [ ] 48/48 真实人工复核通过。
- [ ] Training smoke 通过。
- [ ] 重新核对 200 epoch ETA 和存储余量。

### 训练、评估和选模

- [ ] 只启动一次 Control。
- [ ] 只启动一次 Fix-v1。
- [ ] 验收两臂 200 epoch 和 finite loss。
- [ ] 固定 103 例推理和官方兼容评估通过。
- [ ] 生成配对差、bootstrap 和分层指标。
- [ ] 根据预注册规则选模。
- [ ] 归档最终 checkpoint SHA 和选择依据。

## 19. 备选方案与不采用原因

### 19.1 只删除已知失败样本

不采用。R5 已证明旧失败黑名单不能防止新失败。

### 19.2 只使用 18 个已通过固定样本

不作为当前主方案。18 例相对 1035 例真实训练数据仅约 1.7%，正常采样时收益可能过小，过度重复又会导致过拟合和选择偏差。

### 19.3 只继续收紧 compact-support

不采用。R5 已在组件池仍充足时出现 6/24 视觉失败，根因不是支撑区体积单因素。

### 19.4 只做强度 clip

不采用。直接 clip 可能删除真实病灶的高信号特征，且不能消除边界空间不连续。

### 19.5 失败后不断重抽供体

不采用。会隐藏原始失败率、改变供体分布并使训练 step 时间不可预测。

### 19.6 对标签一起羽化

禁止。分割标签必须保持离散类别，不得引入部分体积浮点标签。

## 20. 最终决策

Fix-v1 在技术上可行，但必须被视为新的、独立的、未批准的实验路线。

在当前比赛时间窗口内：

- 不启动 Fix-v1 开发或训练。
- 不打断 B / 原 E / E-continue 的固定 103 例评估。
- 选定原 E 或 E-continue 后优先完成官方 179 例最终推理。

只有在拥有完整 39–54 小时独立窗口，并且明确授权新路线后，才按本文档从 Gate-0 开始实施。

## 附录 A：核心伪代码

```python
result = plan(segmentation, valid_mask, context)
if result.state != "PLACEMENT_VALID":
    return original_image, original_segmentation, result

generated = backend.generate(original_crop, label_cube, seed=result.event_seed)
geometry = build_blend_geometry(
    label_support=label_cube != 0,
    valid_mask=valid_crop,
    forbidden_mask=forbidden_crop,
    spacing_mm=context.spacing_mm,
    blend_radius_mm=config.blend_radius_mm,
)
if not geometry.valid:
    return no_op("BLEND_PLACEMENT_INVALID")

harmonized = harmonize_generated_crop(
    original=original_crop,
    generated=generated,
    context_ring=geometry.context_ring,
    config=config.harmonization,
)
if not harmonized.valid:
    return no_op("HARMONIZATION_FAIL")

candidate_image = original_crop + geometry.alpha * (
    harmonized.image - original_crop
)
candidate_seg = original_segmentation_crop.copy()
candidate_seg[geometry.label_support] = label_cube[geometry.label_support]

qc = compute_commit_qc(
    original=original_crop,
    generated=generated,
    candidate=candidate_image,
    candidate_segmentation=candidate_seg,
    geometry=geometry,
    frozen_calibration=config.calibration,
)
if not qc.pass_all:
    return no_op(qc.reason, metadata=qc.metrics)

if not validate_commit(
    before_image=original_crop,
    before_segmentation=original_segmentation_crop,
    after_image=candidate_image,
    after_segmentation=candidate_seg,
    image_support=geometry.image_support,
    label_support=geometry.label_support,
):
    return no_op("COMMIT_CONTRACT_FAIL")

append_audit("COMMITTED", qc.metrics, geometry.metadata)
return candidate_image, candidate_seg, committed_result
```

## 附录 B：人工 Gate-2 复核清单

每例必须检查：

- [ ] T1n 三平面无黑块、白带和硬边。
- [ ] T1c 三平面无平面状饱和信号。
- [ ] T2w 三平面无支撑区对齐接缝。
- [ ] T2f 三平面无块状极亮信号。
- [ ] 四模态病灶位置一致。
- [ ] 标签与图像空间对齐。
- [ ] 无脑外生成。
- [ ] 无方块、棱柱或支撑区轮廓可见。
- [ ] 过渡区与周围组织自然衔接。
- [ ] 放大后仍无可见接缝。

决策只允许：

```text
accept
reject
```

不允许使用“大概可以”、“轻微问题但通过”或自动将 pending 改为 accept。
