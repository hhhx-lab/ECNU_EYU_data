# BraTS 2026 S2 MET-AUG Fix-v2 完整修复与验证方案

## 0. 文档状态

| 项目 | 值 |
|---|---|
| 文档类型 | 技术设计、实施合同与验收方案 |
| 文档版本 | Fix-v2 design v3 |
| 文件名说明 | 为保持既有链接沿用 `Fix_v1` 文件名；任何实现/config/root 必须命名为 Fix-v2 |
| 修订日期 | 2026-07-27 |
| 当前状态 | 仅设计，未授权实施 |
| 当前比赛主线 | 原 E 已锁定；官方 179 例推理优先，禁止被本方案打断 |
| MET-AUG 状态 | R4/R5 均人工失败，保持停止；不建 R6 |
| 实施边界 | 只能使用全新独立 root；不得删除、覆盖或续跑 R1-R5 |

> **复核结论：上一版并不是可直接实施的最优方案。** 当前 backend 只返回
> `label_support` 内的生成值，因此上一版在标签外定义的 3 mm 羽化区没有生成残差可供融合，
> `context_ring` 上的 generated 也与 original 完全相同。Fix-v2 不再默认羽化有效，而是先验证
> “现有 backend + 严格 QC”这一最小修复；只有它不能满足盲法门禁时才启用真正的 halo 生成。

> **“最优”的限定：** Fix-v2 是在“冻结现有 G1 checkpoint、EDM-Heun/18、FP32、
> 训练 split 和标签语义，不重新训练生成器”约束下，当前证据支持的首选风险收益方案；
> 它不是未经实验即可证明的全局最优解。真正的最终结论仍由 train-only 校准、盲法门禁、
> 对照训练和固定 103 例评估决定。

## 阅读导航

- [二次复核结论](#1-二次复核结论)
- [失败证据和根因](#2-失败证据和根因)
- [Fix-v2 核心架构](#4-fix-v2-核心架构)
- [扩张 inpainting 与融合算法](#5-扩张-inpainting-与融合算法)
- [自动 QC](#7-自动-qc)
- [校准和防泄漏](#8-校准和防泄漏)
- [重新门禁](#11-重新门禁)
- [训练、评估和选模](#12-训练评估和选模)
- [停止规则和备选路线](#15-停止规则和备选路线)
- [实施清单](#18-实施清单)
- [核心伪代码](#附录-a核心伪代码)

## 1. 二次复核结论

### 1.1 上一版方案的决定性缺口

当前 G1 backend 的实际数据路径是：

1. `sample_tumour_diffusion_inpaint()` 使用标签非零区作为二值 tumour mask。
2. `sample_edm()` 在 tumour mask 外逐步投影回已知原图。
3. `G1FourModalityInpaintingBackend.generate()` 先复制原 crop。
4. 返回前仅执行 `generated_g1[index][support] = rebuilt[support]`。

对应源码：

- `custom_nnunet/met_aug_diffusion.py:267-313`
- G1 `diffusion_inference_utils.py:160-202`
- G1 `model.py:478-614`

所以 backend 的输出满足：

```text
generated[x] == original[x],  x outside label_support
```

上一版定义：

```text
image_support = dilate(label_support, 3 mm)
alpha = 1 inside label_support, then decays outside
candidate = original + alpha * (generated - original)
```

由此直接得到：

- 标签外：`generated - original = 0`，所谓外侧羽化不产生任何变化。
- 标签内：`alpha = 1`，仍是原来的硬替换。
- 标签外 `context_ring`：generated 和 original 相同，median/MAD 对齐退化为恒等变换。

因此，上一版“3 mm 外侧羽化 + 4-8 mm context ring”在现有 backend 合同下不能解决硬边。

### 1.2 固定 3 mm 也不能简单移到标签内

使用 R5 不可变 NPZ 对 6 个 reject 做了只读离线反事实检查。若把 3 mm 余弦渐隐直接
移入 `label_support`，两个小样本的平均残差保留率仅为：

| 病例 | support voxels | 内切半径 | 3 mm 平均残差保留率 |
|---|---:|---:|---:|
| `route-a-smoke-007` | 391 | 3.00 mm | 0.149 |
| `route-a-smoke-015` | 181 | 2.45 mm | 0.163 |

这会显著抹弱小病灶图像信号，但标签仍完整写入，产生新的图像/标签错配。并且 009、010、
012、021 的内部黑块、白带和棱柱信号即使渐隐边界仍然存在。因此“只把羽化改到内侧”
不作为主方案。

### 1.3 Fix-v2 的判断

当前最合理的修复必须同时处理三个独立问题，但不应预设每个安全事件都需要改 backend：

1. **生成内容质量：** 在提交前拒绝黑块、饱和带、平面和棱柱状生成失败。
2. **边界质量：** 对实际 candidate 检查 support-aligned seam，不能只检查 finite/shape。
3. **必要时修复边界合同：** 若严格 QC 后仍有接缝或通过率太低，再让模型真正生成无标签 halo。

R4/R5 共 48 次硬替换人工决策中有 37 次 accept。两轮之间存在重复 target/donor 或相同产物，
所以这不是 37 个独立成功样本，也不能直接估计真实通过率；它只证明“硬替换并非逐例必坏”。
最小充分改动原则要求先证明旧合格事件可被可靠保留、11 个失败可被自动拒绝，再决定是否
承担 halo 的新增风险。

Fix-v2 因此采用预注册候选淘汰，而不是未经数据验证只押一个 blending 算法：

```text
所有候选共享：raw-generation QC + candidate QC + 原子 NO_OP

Candidate A: 当前 label-only inpainting/hard commit + QC
Candidate B: 扩张 halo inpainting + cosine blend + QC
Candidate C: Candidate B + 受约束 median/MAD harmonization
Candidate D: screened Poisson，只有预注册且性能可接受时参与
```

选择规则按顺序为：0 个盲法人工漏检、达到有效增强率下限、病灶信号保留合格，然后优先选择
实现更简单、运行更快、改动面更小的候选。若 Candidate A 满足全部条件，它就是 Fix-v2 的
最终 boundary policy；不能因为 halo 看起来更“高级”而强行选择更复杂方案。

### 1.4 G1 是否学过标签外健康组织

扩张 halo 的前提不是猜测。G1 的正式 EDM 训练路径实际执行：

- `tumour_main_diffusion.py:249-292` 把 clean `x_crop_pad` 传入 `edm_loss_fn()`。
- `model.py:329-405` 对整个 64³ `x_0` 使用 `torch.randn_like(x_0)` 加噪。
- `loss_per_elem` 在全部 D/H/W 体素上求均值，没有乘 tumour-only loss mask。

因此 G1 接受过标签外健康组织的完整去噪监督。数据 transform 虽会额外构造 tumour-noisy crop，
但正式 EDM loss 使用的是 clean crop 并在 loss 内全幅加噪。让 G1 在 `H\L` 生成 label=0 的
健康 halo 与其训练目标一致，比直接对当前零残差做后处理更有依据。

这仍不等于 halo 质量自动合格：inference unknown-mask 的几何与训练采样不同，所以必须通过
train-only architecture ablation、raw-QC、盲法 holdout 和新 Gate-2，不能跳过验证。

### 1.5 halo 的训练-推理一致性风险

源码复核后必须精确区分“训练去噪”和“推理 inpainting”：

- 正式 EDM 训练将 clean `x_crop_pad` 传入 `edm_loss_fn()`，对整个 64³ crop 加噪，
  并在全体素上计算 loss。
- dataloader 虽构造了 tumour-noisy crop，EDM 正式 loss 路径没有使用该 noisy crop。
- `known_mask = 1 - L` 是 `sample_tumour_diffusion_inpaint()` 在推理阶段设置的每步投影，
  不是模型训练时见过的显式 mask-conditioned 任务。

因此，“模型只在 `L` 内训练过 inpainting”不准确；更准确的风险是：
**当推理从 `unknown=L` 扩大为 `unknown=H` 时，在 label=0 的 `H\L` 中让条件扩散自由采样，
扩大了推理时投影规则与训练去噪分布的差距。**

可能后果包括：

1. `H\L` 中原有灰/白质、脑沟回或血管边界被轻微移动，与 `H` 外结构不连续。
2. label=0 生成环与目标病例的局部对比度或亮度存在系统偏移。
3. 改变 `H\L` 的每步状态会通过模型感受野影响 `L` 内的去噪，所以
   `unknown=L` 与 `unknown=H` 的 `L` 内输出不应被假定为字节一致。

这不直接否定 halo，但将 Candidate B/C 定义为必须先通过成对消融和结构保持的
实验候选，不是只因理论上有 halo 就能启用的默认方案。

## 2. 失败证据和根因

### 2.1 R4

- 自动 Gate-2：24/24 数值合同通过。
- 人工：19 accept / 5 reject。
- reject：`004`、`009`、`016`、`021`、`023`。
- 典型问题：支撑区边界硬接缝、T1n 近黑块、T2f 饱和、块状替换。

### 2.2 R5 compact-support

- 自动 Gate-2：24/24 数值合同通过。
- 人工：18 accept / 6 reject。
- reject：`007`、`009`、`010`、`012`、`015`、`021`。
- 典型问题：跨三平面的近黑硬边、饱和白带、矩形/棱柱状多模态信号。

### 2.3 不可变证据

| 证据 | SHA256 |
|---|---|
| R4 `manual_review_decisions.csv` | `323183f3fc15a9992547dfafd41437ef5a542d99241f77d37617618dcd27f817` |
| R4 `manual_review_template.csv` | `aea3c1f010769efc57090afc074a2b1933a273882427e114e5b637399f60f9a2` |
| R5 `manual_review_decisions.csv` | `efcf272a66da89ed36704b52b5af9e2e0a0e08e16c749cdb60014e51c8a4e510` |
| R5 `manual_review_template.csv` | `b3d46f7b68e21ec1647524f4776059b15964e8988ccf243c38d5c53a4359ac35` |
| 原 E checkpoint | `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267` |
| E-continue final | `535e89644121a0c0f1f591f0c1a211581d6d3dd6c1df334a7ccb1bb7825328b1` |

证据位置：

```text
远端 R4: /root/brats2026/runs/s2_met_aug_route_a_20260726_r4
远端 R5: /root/brats2026/runs/s2_met_aug_route_a_20260726_r5
本地 R4: work_space/S2/results/s2_met_aug_route_a_20260726_r4/gate2_run
本地 R5: work_space/S2/results/s2_met_aug_route_a_20260726_r5/gate2_run
```

### 2.4 根因分层

| 编号 | 根因 | 证据 | Fix-v2 处理 |
|---|---|---|---|
| R-A | 标签支撑区既是生成区又是硬提交区 | 接缝严格沿 support 边界 | 分离 `L` 和 `H`，生成过渡 halo |
| R-B | G1 偶发生成内部极端/块状信号 | R4/R5 黑块、白带、棱柱 | 融合前 raw-generation QC，失败 NO_OP |
| R-C | 自动 QC 只验证 finite/shape/变更范围 | 48 例均自动 pass，11 例人工 reject | 加边界、极值、形状和跨模态 QC |
| R-D | compact-support 仅控制几何大小 | R5 仍 6/24 reject | 不再把供体体积筛选当机制修复 |

“硬替换”是接缝的直接机制和异常的放大器，但不是黑块/白带的唯一根因。文档不得再写成
“所有失败都只由硬替换造成”。

## 3. 目标、非目标和不变量

### 3.1 目标

1. 消除可见的 support-aligned paste seam。
2. 自动拒绝近黑、饱和、平面、矩形和棱柱状异常。
3. 保持四模态空间一致和病灶信号有效。
4. 标签保持离散整数 `{-1,0,1,2,3,4}`。
5. 事件可确定重放、可审计、失败时完整回滚。
6. 报告真实 attempted / generated / QC-rejected / committed 分母。
7. 不使用固定 103、internal 104 或官方 179 调参。

### 3.2 非目标

1. 不修改 R4/R5 决策或把 reject 改成 accept。
2. 不把已通过的 18/19 个固定样本直接复制进训练集。
3. 不通过 clip 掩盖生成失败。
4. 不在当前比赛推理期间实施或占用 GPU。
5. 不在 Fix-v2 中重新训练 G1；若必须重训，另立研究路线。

### 3.3 冻结不变量

- true-1mm 数据和固定 split：1035 / 103 / 104。
- G1 checkpoint、G2 parent gate、EDM-Heun、18 steps、FP32。
- seed 和事件身份派生规则。
- 训练名义选择概率 `p_select = 0.20`。
- 原 E 为始终可部署回退。
- R1-R5、原缓存、true-1mm 缓存和所有审计证据不可覆盖。

## 4. Fix-v2 核心架构

```text
确定性选择供体和目标位置
  -> 按冻结 boundary policy 构造 image support
  -> 生成并保存 raw crop
  -> raw-generation QC
  -> Candidate A: 仅 L 内硬提交
     或 Candidate B/C: H 内真实生成并在 H\L 融合
     或预注册 Candidate D: screened Poisson
  -> 标签只在 L 内硬写入
  -> candidate boundary/content/cross-modal QC
  -> 原子 COMMITTED 或完整 NO_OP
  -> append-only 审计
```

强制核心是两阶段 QC、完整回滚和固定分母。若 Gate-0 选择 halo policy，则额外遵守：
**标签告诉模型“病灶在哪里”，inpainting mask 告诉模型“哪里允许重新生成”，两者不得混用。**

## 5. 扩张 inpainting 与融合算法

本节只适用于 Gate-0 选中 Candidate B/C 时。Candidate A 保持现有 backend 和 `H=L`，
不伪造 context ring、不运行 harmonization；它仍必须通过同一 raw/candidate QC 和后续门禁。

### 5.1 几何定义

令：

- `L = label_cube != 0`：离散标签支撑区。
- `d_out(x)`：体素 `x` 到 `L` 的物理欧氏距离。
- `r_halo`：允许生成和融合的 halo 半径。
- `H = {x | d_out(x) < r_halo} union L`：inpainting/image support。
- `C`：`H\L` 中用于强度对齐的外侧健康环带。

所有距离必须使用 `spacing_mm`，不得把毫米直接当体素。当前 true-1mm 数据仍按通用物理实现。

### 5.2 halo 半径不预先写死

候选开发网格：

```text
r_halo in {1.5, 2.0, 3.0, 4.0} mm
```

这只是 train-only Gate-0 的候选集合，不是最终默认值。最终半径必须同时满足：

1. 新生成 halo 足以让边界梯度进入真实训练病灶分布。
2. 不显著改变无标签健康组织。
3. 不导致放置成功率和有效增强率低于冻结下限。
4. 不增加脑外、crop 边缘、已有病灶或 forbidden region 风险。
5. 在盲法 train-only holdout 上优于更简单候选。

最终值写入 calibration JSON 并绑定 SHA。不得在看到新 Gate-2 或固定 103 结果后调整。

### 5.3 放置安全合同

放置必须满足：

1. `L` 和整个 `H` 均位于显式 valid brain mask。
2. `L` 和 `H` 均不与目标病例已有病灶、padding `-1` 或 forbidden region 重叠。
3. `H` 与 crop 六个边界保留至少 1 个体素保护层。
4. 强度对齐环带 `C` 具有冻结要求的有效体素数和鲁棒方差。
5. target group 与 donor patient group 不同。

任一失败：`NO_OP / HALO_PLACEMENT_INVALID`，不缩半径重试，不换供体重抽。

### 5.4 backend 合同修复

S2 侧新增独立 adapter；不修改冻结的 G1 checkpoint，也不原地改写锁定 G1 源码。

建议接口：

```python
class HaloGeneration(NamedTuple):
    image: np.ndarray
    label_support: np.ndarray
    inpaint_support: np.ndarray

def generate(
    image_crop: np.ndarray,
    label_crop: np.ndarray,
    inpaint_support: np.ndarray,
    *,
    seed: int,
) -> HaloGeneration:
    ...
```

调用 G1 时必须区分：

```text
label_cond  = labels derived from L
known_mask  = 1 - H
known_scan  = clean original crop
```

关键要求：

- `H` 内从扩散采样结果取值，而不是只取 `L`。
- `H` 外逐体素恢复原图。
- 不得继续使用 `generated_g1[index][L] = rebuilt[L]` 作为最终 backend 返回合同。
- G1 四个 checkpoint、EDM 参数和 seed 派生不变。
- 新 S2 adapter、冻结 G1 runtime 和全部 checkpoint 分别记录 SHA。

这样 `H\L` 中才有真正的无标签健康组织生成值，外侧羽化和环带强度对齐才有数学意义。

### 5.5 受约束的局部强度对齐

对每个模态 `c`，在配对的健康 halo 环带 `C` 上计算：

```text
gain_c   = MAD(original_c[C]) / MAD(generated_c[C])
offset_c = median(original_c[C]) - gain_c * median(generated_c[C])
G'_c     = gain_c * generated_c + offset_c
```

等价的中心化写法更清楚：

```text
G'_c = median(original_c[C])
       + gain_c * (generated_c - median(generated_c[C]))
```

`offset` 会抵消环带中位数差，因此不能用“`gain * 绝对强度`”的简化例子
估计真实放大量；但 `gain` 依然会放大 generated 相对其中位数的局部偏差，
所以风险真实存在。

安全约束：

- `C` 只包含 label=0、valid brain、无既有病灶的体素。
- `MAD`、gain、offset 必须 finite。
- 生成环带方差过小、体素不足或参数超出 train-only 冻结范围时直接 NO_OP。
- `gain/offset` 必须落在每模态、每体积/面积档的 train-only 冻结区间；
  超限时拒绝，不得用 clip 强行拉回。
- 同时保留未 harmonize 的 Candidate B 中间产物，计算 `H\L` 中后处理相对放大比、
  相对 `L` 的 halo 变化量和按物理距离分壳的残差包络。
- 不做无限制 clamp，不用固定 103 或官方 179 估计参数。
- 同时在 Gate-0 比较“无对齐”和“median/MAD 对齐”；只有盲法 holdout 支持时才启用对齐。

### 5.6 余弦融合

定义：

```text
alpha(x) = 1                                      x in L
alpha(x) = 0.5 * (1 + cos(pi * d_out(x)/r_halo))  x in H\L
alpha(x) = 0                                      x outside H
```

逐通道：

```python
candidate_image = original + alpha * (harmonized_generated - original)
```

此时与上一版不同：`H\L` 中存在真实生成残差，因此 alpha 能形成实际过渡。

### 5.7 标签合同

标签仅在 `L` 内硬写入：

```python
candidate_segmentation[L] = label_cube[L]
```

禁止对标签羽化、插值或写入浮点部分体积。`H\L` 是无标签图像过渡区，标签保持目标原值。

本项目标签语义必须在 config、calibration 和报告中显式绑定：

| 标签 | 语义 | Route A 插入合同 |
|---:|---|---|
| 0 | background | `L` 外和 halo 中的无病灶标签 |
| 1 | NETC | 允许 |
| 2 | SNFH | 允许 |
| 3 | ET | 允许 |
| 4 | RC | 禁止作为 Route A 供体标签 |

`label 4` 是切除腔 RC，不是 NETC/NCR 的别名。目标原图在 image support 外可以
保留既有 RC，但 Route A 供体 `label_cube` 和新增 `L` 必须仅含 `{1,2,3}`；
若出现 `B_4`，这是 `LABEL_CONTRACT_FAIL`，不是可通过调高边界阈值放行的普通 QC 事件。

## 6. 事务状态与审计

建议状态：

```text
NOT_SELECTED                  -> NO_OP
NO_ELIGIBLE_DONOR             -> NO_OP
LABEL_INVALID                 -> NO_OP
HALO_PLACEMENT_INVALID        -> NO_OP
BACKEND_FAILURE               -> NO_OP
RAW_GENERATION_QC_FAIL        -> NO_OP
HARMONIZATION_FAIL            -> NO_OP
CANDIDATE_BOUNDARY_QC_FAIL    -> NO_OP
CANDIDATE_CONTENT_QC_FAIL     -> NO_OP
CANDIDATE_CROSS_MODAL_QC_FAIL -> NO_OP
COMMIT_CONTRACT_FAIL          -> NO_OP
全部通过                       -> COMMITTED
```

所有 NO_OP 必须返回输入图像和标签的逐体素原值。不得发生部分写入、失败后换供体或悄悄重试。

每个事件至少记录：

- event id / seed / target / donor / patient group。
- `L`、`H`、`C` 体素数和几何参数。
- raw、harmonized、candidate 的 QC 指标。
- 每个存在的 `(subregion, modality)` 边界面数、物理面积、分布距离、
  稳健分位数、最大连续异常面片和所用的标准/小样本分支。
- 每模态 gain/offset。
- 最终状态和唯一失败原因。
- config/calibration/source/runtime/checkpoint/manifest SHA。

## 7. 自动 QC

### 7.1 两阶段 QC，不得只看最终融合图

#### 阶段 A：raw-generation QC

目的：在羽化可能掩盖异常前捕获生成器本身的失败。

检查：

- shape、dtype、finite、坐标和四模态顺序。
- `L` 内近黑、近白、极端残差体素比例。
- 连续平面状极值区域的面积、厚度和三平面持续性。
- 高残差连通域的体积、bounding-box fill ratio、主轴比和表面积/体积比。
- 按标签亚区条件化的四模态效应向量和空间对应；不要求不同模态的异常区互相 1:1 重合。
- 支撑区效应量是否落在真实 train lesion 的冻结参考分布内。

R4/R5 的 009、010、012、015、021 等不能依赖羽化“修漂亮”；应优先由此阶段拒绝。

#### 阶段 B：candidate QC

目的：验证真正准备提交给 nnU-Net 的图像。

检查：

- 图像仅在 `H` 内变化，标签仅在 `L` 内变化。
- `H` 外图像和标签逐体素不变。
- 病灶外边界 `∂L` 必须按内侧亚区 `k` 与模态 `c` 分层评估，不得使用一个
  混合全部标签和四模态的 seam 阈值。
- Candidate A/D 在 `∂L` 完成亚区边界 QC；Candidate B/C 既在 `∂L` 完成亚区边界
  QC，又在 halo 外边界 `∂H` 完成 label-0 健康组织连续性 QC。
- `∂H` 的值、法向梯度、候选/原图残差和连续异常面片必须在冻结范围内。
- Candidate B/C 的 `H\L` 残差能量、原图/候选梯度方向一致性和结构保持指标必须在冻结范围内，
  防止 halo 改写脑室、皮层或血管等原有解剖边界。
- Candidate C 必须通过每模态的 halo amplification QC：`H\L` 后处理放大比、
  `Q95/Q99(|candidate-original|)` 相对 `L` 的比值，以及按 `d_out/r_halo` 分壳的残差上包络
  均必须低于 train-only 冻结上限。
- 不把 `Q95(H\L) <= Q95(L)` 或 `|alpha*gain| <= 1` 写成无条件定律：
  前者会误拒低对比小病灶，后者忽略 offset 和局部原图偏差。两者可作为
  Development 诊断量，正式上限必须分层校准。
- `L` 内保留足够病灶残差，不能因融合退化为近原图但标签仍为病灶。
- 四模态的亚区条件效应向量、重合、质心偏移和组件拓扑符合真实训练病灶的联合分布。
- 无脑外生成、已有病灶覆盖、通道交换、NaN/Inf 或非法标签。

### 7.2 按亚区 x 模态的边界 QC

#### 7.2.1 边界定义

不用“外边界体素最近哪个标签”的模糊实现。对插入标签 `S` 和 `L = S != 0`，
使用 6-邻域有向面定义：

```text
F_k = {(x, y, a) | S(x) = k, x in L, y outside L,
                     y = x +/- e_a, k in {1,2,3}}
```

- `x` 是病灶内侧体素，`y` 是相邻外侧体素，`a` 是面法向轴。
- 每个边界面由内侧标签唯一归属给 `F_1/F_2/F_3`，角点和多亚区接触不做最近标签猜测。
- 物理面积由与 `a` 垂直的两个 spacing 乘积给出；各向异性数据不得按等体素处理。
- 仅统计从 `L` 指向背景的外边界面；亚区之间的内部接触面不属于 paste seam，
  仍由 raw content/cross-modal QC 检查。

`B_k` 是 `F_k` 的内侧体素表示，仅用于可视化；数值计算以有向面 `F_k` 为准。

#### 7.2.2 每亚区、每模态的法向跳变

对 `k in {1:NETC, 2:SNFH, 3:ET}` 和
`c in {T1n, T1c, T2w, T2f}`，对每个面 `f=(x,y,a)` 计算：

```text
g_signed[k,c](f) = (I_c(y) - I_c(x)) / (spacing[a] * scale_c)
g_abs[k,c](f)    = abs(g_signed[k,c](f))
```

`scale_c` 来自 image support 外、valid brain 内、不含既有病灶的冻结局部参考环 `R`
的稳健 MAD，不能使用候选生成区估计。Reference 真实病灶使用完全相同的
`F_k/R/spacing/scale` 定义。既检查 signed 分布也检查 absolute 分布，避免近黑硬边和
过亮硬边在取绝对值后丢失方向信息。

生理先验只用于解释，不用于跳过模态：ET 的锐利性主要在 T1c 上表现，SNFH 的
弥散高信号主要在 T2w/T2f 上表现，NETC 则更混杂。因此阈值必须是 `(k,c)` 级，
但每个存在的亚区仍检查全部四模态。

#### 7.2.3 分布判定

对边界面积充足的 `(k,c)` 层，计算候选事件与匹配的 train-only Reference
真实病灶边界经验 CDF 之间的 KS distance `D_signed[k,c]` 和 `D_abs[k,c]`。

这里仅使用 KS statistic，**不使用普通 KS 理论 p-value**。同一病灶的边界面在空间上
高度相关，把每个面当独立样本会人为放大显著性。硬阈值必须来自
leave-one-patient/component-out 的真实病灶事件级经验分布，并在 QC holdout 上一次性验收。

每层同时检查：

- `D_signed[k,c]` 和 `D_abs[k,c]` 的冻结上限。
- `Q50/Q90/Q95/Q99(g_signed, g_abs)` 的经验接受区间。
- 超出 Reference 局部包络的边界面比例。
- 连续异常表面组件的最大物理面积和总边界面积占比。

任一存在的 `(k,c)` 超出对应冻结限值即 `CANDIDATE_BOUNDARY_QC_FAIL`。同时必须在
train-only calibration 中用“单个事件所有可用 `(k,c)` 的最大标准化偏离”联合冻结
事件级误拒率，不得在部署后通过反复放宽某个层抵消多重判定。

#### 7.2.4 亚区缺失和小边界 fallback

- `F_k` 为空：记录 `not_present`，不要求该事件伪造该亚区的 KS 值。
- `F_k` 非空但物理面积或独立表面片数低于冻结 `min_boundary_area[k]`：
  不运行普通 KS 分支，改用事件级稳健分位数、最大绝对跳变和连续异常面积的
  小样本阈值。
- 小样本 fallback 的参考不足或局部标尺 `scale_c` 不可靠：
  `NO_OP / BOUNDARY_QC_INSUFFICIENT_SUPPORT`，不得因样本少自动放行。
- `F_4` 非空：直接 `LABEL_CONTRACT_FAIL`，不进入 RC 边界 calibration。

### 7.3 按亚区条件化的跨模态 QC

不使用“四模态变化区域必须互相高 IoU”的统一规则。真实 ET 在 T1c 上通常最显著，
SNFH 在 T2w/T2f 上通常更显著，NETC 的方向和强度更混杂；这些范围天然不会 1:1 重合。

但也不采用硬编码的 `T1c->{3}`、`T2w->{1,2,3}` 二值对应。NETC/SNFH 在 T1c/T1n 上
仍可能与正常组织不同，ET 在 T2w/T2f 上也不是固定方向。把这些医学趋势写成硬逻辑
会产生新的误拒和漏检。

正式实现对每个 `k in {1,2,3}` 分别建立：

```text
e[k,c] = robust standardized contrast of candidate_c[label k]
         relative to the unchanged local reference ring R_c
A[k,c] = voxels/surface patches whose local-reference contrast exceeds
         the train-only (k,c) envelope
v[k]   = concatenated four-modality effect, spread, affected-fraction,
         centroid-offset and shape features for label k
```

真实训练病灶没有“插入前原图”这个反事实，所以不得用 `candidate-original` 作为与
真实病灶直接比较的生理效应。候选事件和 Reference 真实病灶都使用相同的
“亚区强度相对局部未变参考环 `R` 的稳健标准化对比”。`candidate-original` 仍作为
增强残差、harmonization amplification 和事务范围诊断量，但不冒充真实病灶参考。

判定规则：

1. `e[k,c]` 与 `A[k,c]` 分别对比真实 train-only 的同 `(k,c,体积档)` 分布。
2. `v[k]` 使用以患者为簇的 shrinkage covariance/Mahalanobis 或预注册等价稳健多元距离，
   不在小样本上直接求逆奇异协方差。
3. 空间对齐只在同一亚区内比较经验重合、质心偏移和组件拓扑；对每个
   `(k,c1,c2)` 使用真实病灶的冻结接受范围，不要求 IoU=1。
4. “效应方向矛盾”、“模态间完全错位”可作为硬失败，但矛盾区域与方向必须由
   train-only Reference 的联合分布和医学复核共同冻结，不凭单个书本表格决定。
5. 任一亚区的多元效应或空间对齐超限即 `CANDIDATE_CROSS_MODAL_QC_FAIL`。

### 7.4 指标必须可复现

对通道 `c` 定义：

```text
delta_c(x) = generated_c(x) - original_c(x)
scale_c    = max(MAD(original_c[R]), epsilon)
z_c(x)     = delta_c(x) / scale_c
```

`R` 是不参与生成/融合的冻结局部参考环，与 Candidate B/C 可选的 harmonization
ring `C` 是两个概念。Candidate A 也必须能构造 `R`。

至少冻结以下统计：

- `Q01/Q05/Q50/Q95/Q99(z_c[L])`。
- `mean(|z_c| > q_extreme)`，其中 `q_extreme` 来自 train-only 真实分布。
- 每个 `(k,c)` 的边界面数/物理面积、`D_signed/D_abs`、梯度稳健分位数和
  最大连续异常面片。
- Candidate B/C 在 `∂H` 上的 `Q95/Q99` 法向梯度、残差和连续异常面片。
- 高残差连通域的 bbox fill、平面厚度和主轴比。
- 每个 `(k,c)` 的 effect/fraction/centroid/shape 特征，以及每个 `(k,c1,c2)` 的经验
  重合和质心偏移；不报告无条件的全模态统一 IoU 门禁。
- `L` 内 candidate residual energy / raw residual energy。
- Candidate C 每模态的 gain/offset、halo amplification ratio、halo-to-lesion residual ratio 和归一化距离分壳残差包络。
- 四模态鲁棒效应向量的 shrinkage-Mahalanobis distance。

每个公式、邻域定义、连通性、分位数方法、epsilon 和阈值来源都写入 calibration JSON。
不接受“视觉上看起来差不多”的隐式实现。

### 7.5 阈值原则

- 阈值来自 train-only reference/development 数据。
- R4/R5 只作回归，不作为主要阈值拟合样本。
- 禁止根据新 Gate-2、固定 103、internal 104 或官方 179 调阈值。
- 阈值不能只保证 11 个旧 reject 被抓住；还必须报告对旧 accept 和盲法 holdout 的误拒率。
- 边界阈值必须分别绑定 `label_semantics_sha`、`modality_order_sha`、边界面定义、
  spacing 算法、体积/面积档、小样本分支和 Reference CDF SHA。
- 不用任意 clip 把异常值压回阈值内。

## 8. 校准和防泄漏

### 8.1 train-only 三分区

在固定 1035 例训练 split 内，按 patient group 和冻结 SHA hash 建立互斥分区：

| 分区 | 建议比例 | 用途 |
|---|---:|---|
| Reference | 70% | 真实病灶边界、强度和跨模态分布 |
| Development | 15% | 候选半径、是否 harmonize、QC 阈值开发 |
| QC holdout | 15% | 阈值冻结前一次性盲法验收 |

最终实际病例数、patient group 清单和 SHA 必须落盘。不得让同一 patient group 跨分区。

### 8.2 亚区 x 模态边界参考校准

只从 Reference 分区的真实病灶提取外边界，且按 patient group 保持簇独立。
对每个真实连通病灶组件：

1. 按 7.2 的相同面定义提取 `F_1/F_2/F_3`。
2. 对四模态提取 signed/absolute 法向跳变、稳健分位数和表面异常组件。
3. 按 `(k,c,core-volume stratum,boundary-area stratum)` 建立参考；分档边界只用
   Reference 数据冻结，不根据后续候选成败调整。
4. 进行 leave-one-patient/component-out：当前组件不得同时出现在待测分布与其参考 CDF 中。
5. 以病灶/患者为重采样单位做 cluster bootstrap，冻结每层上限和事件级
   max-statistic 阈值；不把边界面数当成独立样本数。
6. 在 Development 上开发后，只能在 QC holdout 上一次性验证误拒和漏检。

同一 Reference 分区还必须为 7.3 提取每个 `(k,c)` 相对同定义局部参考环 `R` 的
稳健标准化对比、受影响比例、质心/形状特征，
并为每个 `k` 建立四模态联合稳健分布。协方差收缩强度、空间对齐接受范围、
亚区/模态缺失处置和多重判定的事件级上限都在 Development 后冻结。

Harmonization 没有对应的“真实 halo 后处理”参考，所以 gain/offset、放大比、halo-to-lesion
比值和径向分壳包络只能在 train-only Development 的固定候选事件上开发，并在互斥
QC holdout 上一次验证。Candidate C 若不能显著优于 B，就淘汰 C，不为保留 harmonization
而放宽阈值。

不允许因某一细分层样本少就临时合并 ET 和 SNFH，或 T1c 和 T2f。如果细分档
不足，只能使用预注册的层级回退：先合并相邻体积/面积档，仍保留 `(k,c)`；
再不足则启用 7.2.4 的小样本稳健阈值或安全 `NO_OP`。

冻结 calibration 至少包含：

```text
label_semantics / modality_order / connectivity / spacing_rule
reference_patient_groups / reference_component_ids / reference_cdf_sha256
volume_bins / boundary_area_bins / min_boundary_area
per_(k,c,stratum)_ks_limits / quantile_intervals / surface_patch_limits
small_sample_fallback / event_level_max_statistic_limit
per_(k,c)_effect_envelopes / per_k_cross_modal_joint_model
per_(k,c1,c2)_overlap_and_centroid_limits / covariance_shrinkage
harmonization_gain_offset_limits / halo_amplification_limits
halo_to_lesion_limits / radial_shell_residual_envelopes
```

Route A 不为 RC 拟合边界参考；插入 `label 4` 始终是合同失败。

### 8.3 候选架构比较

在同一固定事件集合上比较：

1. Candidate A：当前 label-only inpainting/hard commit + 完整 QC。
2. Candidate B：halo 半径 `{1.5, 2.0, 3.0, 4.0}` mm + cosine blend + 完整 QC。
3. Candidate C：Candidate B + 受约束 median/MAD harmonization。
4. 可选 Candidate D：screened Poisson/Laplacian blending；必须在 Gate-0 前预注册实现、solver
   tolerance 和性能上限，不能看到结果后临时加入。
5. fractional-known-mask soft projection 只作预注册研究 ablation，不默认进入正式候选。

所有候选使用同一 target/donor/event seed 清单，并报告各自的漏检、误拒、信号保留、通过率、
运行时间和新增代码面。只有在更简单候选不能同时满足质量和通过率时，才允许提升复杂度。

Candidate A 与 B/C 的开发比较必须显式复用相同初始 latent/noise 张量、模型、
target/donor/seed 和采样时间表，唯一改变是 unknown mask。记录 `H\L` 生成效应、`L` 内输出漂移、
目标原图结构保持和 `∂H` 连续性。若 B/C 在盲法 holdout 上出现系统性灰/白质边界、
脑沟回或血管结构漂移，直接淘汰 halo 候选，不用 alpha 将其解释为“已修复”。

Candidate A 的目的不是忽略接缝，而是检验“严格拒绝不合格事件”是否已经是充分修复。它若在
盲法 holdout 出现任何漏检，或通过率低于冻结下限，就不能进入 Gate-1。

候选选择使用以下词典序，不做事后加权总分：

1. QC holdout 中自动放行的候选必须 0 人工 reject。
2. generation pass rate 和 effective augmentation rate 达到冻结下限。
3. 病灶残差保留、跨模态分布和结构保持全部合格。
4. 前三项并列时，选择新增代码更少、吞吐更高的候选。

### 8.4 人工标注和冻结

- Development 和 QC holdout 使用互斥 target/donor。
- 人工审核者在看不到方法名和参数的情况下比较 montage。
- 若有两名审核者，分歧按 reject 或由预注册第三方裁决。
- R4/R5 的 11 reject 必须被旧产物 raw-QC 回归捕获。
- R4/R5 的 37 accept 用于报告误拒率，不用于放松阈值。
- 所有选择在新 Gate-2 前冻结。

冻结文件：

```text
calibration/TRAIN_ONLY_PARTITIONS.json
calibration/FROZEN_FIX_V2_CALIBRATION.json
calibration/FROZEN_FIX_V2_CALIBRATION.ok
```

内容包括数据清单、全部候选、指标定义、阈值、人工决策、选择规则、源码和运行时 SHA。

## 9. 有效增强率合同

QC 失败直接 NO_OP 会降低真实增强暴露量，因此只固定 `p_select=0.20` 不够。

必须分别报告：

```text
attempted_events
selected_events
placement_valid_events
backend_attempted_events
raw_qc_rejected_events
harmonization_rejected_events
candidate_qc_rejected_events
committed_events
```

派生：

```text
selection_rate       = selected / attempted
generation_pass_rate = committed / backend_attempted
effective_aug_rate   = committed / attempted
```

规则：

1. 不通过重抽供体提高 pass rate。
2. 不在训练时动态提高 `p_select` 补偿 NO_OP。
3. 在 Gate-0 根据盲法 holdout 冻结 `minimum_generation_pass_rate` 和
   `minimum_effective_aug_rate`。
4. 建议开发警戒线为 generation pass rate 80%；最终数值必须在 Gate-2 前冻结。
5. 若有效增强率过低，方案失败；不能把几乎全 NO_OP 的训练臂称为增强训练。

## 10. 测试计划

### 10.1 单元和属性测试

1. boundary policy 必须是冻结枚举值，不能运行时静默回退。
2. Candidate A 明确满足 `H=L`，不构造伪 context ring，不运行 harmonization。
3. Candidate B/C 满足 `L proper-subset H`，label condition 只来自 `L`。
4. Candidate B/C 传给 sampler 的 unknown region 是 `H`，不是 `L`。
5. Candidate B/C backend 在 `H\L` 返回真实采样结果，不等于无条件复制原图。
6. Candidate B/C 的 `alpha=1` 覆盖 `L`，在 `H\L` 单调衰减，`H` 外为 0。
7. alpha、距离和形态学操作支持各向异性 spacing。
8. 所有 policy 的图像在其冻结 image support 外逐体素不变。
9. 标签在 `L` 外逐体素不变且类别合法。
10. 脑外、crop 边缘、既有病灶和 forbidden overlap 均 NO_OP。
11. harmonization 参数非 finite、方差过小或环带不足均 NO_OP。
12. raw/candidate QC 失败不产生部分提交。
13. 审计写入失败时事务不提交。
14. 相同输入、seed、配置和 runtime 逐数组字节一致。
15. `F_k` 由内侧标签面唯一归属，多亚区接触和角点不重复、不漏面。
16. 法向导数和物理面积在各向异性 spacing 下按轴正确缩放。
17. 亚区缺失记录 `not_present`，小边界进入冻结 fallback，两者不被混同。
18. Route A 插入 RC 或出现 `F_4` 必须在任何边界统计前合同失败。
19. Candidate C 在 gain/offset、halo amplification 或径向包络超限时完整 NO_OP。
20. 跨模态统计按亚区计算，不存在全模态异常 mask 必须相等的隐式断言。

### 10.2 合成故障注入

必须确认以下故障被自动拒绝：

- 全黑/全白 support。
- 单通道饱和白带。
- 多模态矩形或棱柱块。
- 边界对齐的平面跳变。
- halo 外写入。
- 标签和图像错位。
- 通道交换。
- NaN/Inf。
- harmonization 极端 gain/offset。
- harmonization 在任一径向 halo shell 中制造超出冻结包络的放大残差。
- 融合后病灶残差几乎消失。

正负对照还必须包含：

- train-only 真实 ET 在 T1c 上的正常锐利边界不被统一梯度阈值误拒。
- 人工在 SNFH 的 T2w/T2f 外边界注入硬跳变时，对应 `(k,c)` 层必须拒绝。
- 真实病灶的 T1c/T2w 受影响范围不同时仍能通过亚区条件跨模态 QC。
- 将单一模态的效应 mask 刚性平移、交换通道或注入超出 Reference 联合分布的效应方向时必须拒绝。

### 10.3 旧证据回归

- R4/R5 11/11 reject 的旧 candidate 必须由 raw/candidate QC 捕获。
- 已知 reject 不得只因羽化后不明显就自动重标 accept。
- 旧 accept 先按 artifact SHA 去重；自动保留率目标至少 90%，给出去重后的 Wilson 区间，
  并按 target patient group 做 cluster bootstrap，不能把跨 R4/R5 重复产物当独立样本。
- 旧 accept/reject 样本量小，只能做回归，不能证明泛化质量。

### 10.4 backend 端到端测试

- 对 Candidate A 证明当前 label-only backend 未漂移，且 QC 失败完整 NO_OP。
- 对 Candidate B/C 证明 `H\L` 在采样阶段确实为未知区，`H` 外由 clean original 投影保持。
- 对同一固定 target/donor/seed 显式重放相同初始 latent/noise：Run-L 使用
  `unknown=L`，Run-H 使用 `unknown=H`，除 mask 外不得改变任何条件。
- Run-L 在 `H\L` 必须等于原图；Run-H 在 `H\L` 必须有可审计的非零生成残差；
  两者在 `H` 外都必须逐体素等于原图。
- 不要求 Run-L 与 Run-H 在 `L` 内字节一致；必须报告 `L` 内成对漂移、病灶信号保留和 raw-QC 变化。
- 在 `H\L` 和 `∂H` 测量 candidate 相对原图的局部 NCC/SSIM、梯度方向一致性、
  3D structure-tensor 主方向偏差和原有解剖边界位移，阈值由 train-only 成对开发冻结。
- 普通标量 MRI 不直接测量白质纤维方向，不得把 structure-tensor 指标写成已验证真实纤维束连续。
- 若 Run-H 在盲法 holdout 上不满足结构/强度连续性，Candidate B/C 不得进入 Gate-1；
  回到 Candidate A 是否能以严格 QC 满足通过率的证据判定。
- 记录四模态 raw generation 和 candidate；启用 harmonization 时额外记录 harmonized generation。
- 同一 H20、同一 runtime 重放时数组和事件 JSON 字节一致。
- 不要求压缩 NPZ 容器字节一致，除非同时固定归档时间元数据；比较数组 payload 和审计 JSON。

## 11. 重新门禁

### 11.1 Gate-0：静态、校准和盲法 holdout

必须通过：

- 全部单元、属性、故障注入和旧证据回归。
- train-only 三分区审计。
- 候选架构盲法比较。
- QC holdout 一次性通过。
- boundary policy、相关几何/harmonization 参数、QC 阈值和有效增强率下限已冻结。

### 11.2 Gate-1A：100,000 次确定性规划事件

保留 100,000 次门禁，但明确其职责：

- 验证选择、供体、patient-group 隔离、冻结 policy 对应的放置/mask 几何和审计。
- 使用确定性多进程分片，最终按事件索引合并。
- JSONL 恰好 100,000 行，0 violation，0 worker failure，0 残留 shard。
- 串/并行事件 JSONL 和报告字节一致。
- 不把 Gate-1A 冒充 Diffusion 视觉质量门禁。

### 11.3 Gate-1B：端到端 Diffusion 重放

新增 96 个固定 backend 事件，三个 core-volume 分档各 32：

- target 和 donor 唯一、train-only、patient group 隔离。
- 正式 EDM-Heun/18/FP32。
- 同一 runtime 重放两次。
- raw、candidate 数组逐字节一致；启用 harmonization 时该数组也必须一致。
- audit JSON 字节一致。
- 记录 GPU 墙钟、显存和全部 QC 状态。

Gate-1B 只验证端到端确定性和合同，不用人工结果调阈值。

### 11.4 Gate-2：固定分母的盲法视觉门禁

预先冻结 120 个 backend-attempted 事件：

- 三个 core-volume 分档各 40。
- 120 个唯一 target 和 donor component。
- 全部来自 train-only，且与 Development/QC holdout 互斥。
- 在分档内按冻结 seed 随机抽取，不得人工挑选 montage；同时覆盖 support size、表面积/体积比、
  近 CSF/皮层和相对均质组织等预注册 stress strata。
- manifest 还必须在抽取前冻结 `{NETC,SNFH,ET}` 的存在/组合覆盖和每亚区最小
  attempted 数；不包含 RC，不在看到 QC 结果后补齐某类。
- 不因 QC reject 补抽，不因视觉结果换样本。

自动验收：

- 恰好 120 个 raw artifacts 和事件记录。
- 每个 COMMITTED 事件具有 raw/candidate NPZ 和 montage；启用 harmonization 时额外包含该数组。
- 每个事件都必须逐 `(k,c)` 报告 boundary/cross-modal 统计、`not_present` 或
  small-sample fallback 分支；Candidate B/C 另报 `∂H` 结构保持，Candidate C 另报径向放大包络。
- generation pass rate 达到冻结下限，建议设计目标至少 80%，即至少 96 COMMITTED。
- 所有状态分母、失败原因和 SHA 完整。
- 0 非预期异常、0 missing、0 非有限输出、0 合同漂移。

人工验收：

- 对全部 COMMITTED 事件逐例、四模态、三平面、局部放大复核。
- 全部 COMMITTED 必须 accept；任一人工 reject 即 Fix-v2 失败。
- 对全部自动 reject 至少核对其原因截图；不得把明显合格样本大量误拒以换取表面 100% pass。
- 若至少 96 个 COMMITTED 且 0 人工 reject，二项分布零失败的一侧 95% 上界约为 3.1%；
  报告必须写出实际 `n` 和精确上界，不能只写“100%”。

### 11.5 Training smoke

- 至少 128 个真实 train step。
- 达到冻结的 minimum committed count，建议开发目标至少 12 个 COMMITTED。
- loss 全 finite，无 OOM/NaN/Inf/Traceback。
- 不运行 validation，不写 checkpoint，不落盘未审计生成资产。
- 显存、GPU 利用率和吞吐正常。
- 相对 control 的训练墙钟增幅不超过训练启动前按资源窗口冻结的上限。
- 根据实测更新 200 epoch ETA 和磁盘预算。

## 12. 训练、评估和选模

### 12.1 两臂对照

| 训练臂 | 初始 checkpoint | `p_select` | 作用 |
|---|---|---:|---|
| Control | 同一冻结 E | 0 | 测量继续训练本身收益 |
| Fix-v2 | 同一冻结 E | 0.20 | 测量经完整 QC 的增强净收益 |

其他完全一致：

- split 1035/103/104。
- seed、trainer、lr=0.001、focal gamma=2。
- 200 epoch、save every 25、compile=0。
- 单 GPU，不使用 DDP。
- checkpoint cadence 和 validation cadence 一致。

不动态提高 Fix-v2 的 `p_select`。训练报告必须同时给名义概率和实际 `effective_aug_rate`。

### 12.2 固定 103 例评估

环境固定：

- `BraTS-evaluation == 0.0.8`
- `panoptica == 2.1.0`
- `NumPy == 1.26.4`
- `config = mets`
- `vol_threshold = 27`
- `overlap_threshold = 0.2`

汇报：

- ET / RC / TC / WT lesion-wise DSC 和 NSD。
- all/small/large instance F1。
- FN、FP。
- 103 例逐例配对差。
- patient-level paired bootstrap 95% CI。
- 官方工具不提供 tiny 时记录 `tiny_metric_available=false`，不得伪造。

### 12.3 选模合同

训练启动前冻结：

1. Fix-v2 在 WT/TC 主指标上不得出现超过 0.01 的绝对下降。
2. Fix-v2 必须改善至少两个预注册目标指标。
3. 目标优先为 small-instance F1、RC DSC/NSD/F1 和 FN。
4. paired CI、逐例差和失败病例检查必须与总体均值一起报告。
5. 若证据不明确或净收益不足，选择 Control/原 E。
6. 官方 179 例不能用于二次选模。

## 13. 验收总表

| 阶段 | 必须满足 | 失败处置 |
|---|---|---|
| 设计复核 | policy 合同与实际 backend 一致；halo policy 必须真实生成 `H\L` | 不进入实现 |
| Gate-0 | 测试、校准、盲法 holdout、冻结合同通过 | 修改算法需新版本 |
| Gate-1A | 100000/100000、串并行等价、0 violation | 保留证据，停止 |
| Gate-1B | 96 事件端到端重放一致 | 保留证据，停止 |
| Gate-2 自动 | 固定 120 分母、pass rate 达标、0 合同错误 | 不人工硬通过 |
| Gate-2 人工 | 所有 COMMITTED accept | Fix-v2 失败 |
| Training smoke | 128 steps、committed 达标、finite、无副作用 | 不启动正式训练 |
| 正式训练 | 两臂 200 epoch、合同无漂移 | 回退原 E |
| 固定 103 | 0 missing/error、配对评估完整 | 不选 Fix-v2 |
| 选模 | 满足预注册净收益规则 | 选择 Control/原 E |

## 14. 目录和不可变审计

实施时使用全新 root，例如：

```text
/root/brats2026/runs/s2_met_aug_fix_v2_YYYYMMDD_r1/
  config/
  calibration/
  regression/
  gate1_planning/
  gate1_end_to_end/
  gate2_run/
    raw_artifacts/
    committed_artifacts/
    montages/
    rejected_montages/
    manual_review_template.csv
    manual_review_decisions.csv
  training_smoke/
  control_train/
  fix_v2_train/
  evaluation/
  selection/
  runtime/
  logs/
```

每阶段必须有：

- launch contract 和 completion/hold marker。
- 输入/输出 manifest。
- 数据、split、plans、checkpoint、配置、校准和源码 SHA。
- Python/PyTorch/CUDA/cuDNN/GPU 环境。
- PID、UTC 起止时间、墙钟和资源峰值。
- 全部分母、失败原因和旧证据引用。

## 15. 停止规则和备选路线

### 15.1 立即停止条件

1. 新 Gate-2 任一 COMMITTED 人工 reject。
2. generation pass rate 或 effective augmentation rate 低于冻结下限。
3. 旧 11 reject 无法全部被 QC 捕获。
4. QC holdout 出现未覆盖的新系统性伪影。
5. 训练吞吐、显存、墙钟或磁盘超预算。
6. 固定 103 不满足净收益合同。

同一算法/阈值不得在同一 root 清空后重跑。看到 Gate-2 后修改算法必须新版本、新 root、新冻结审计。

### 15.2 备选方案排序

| 优先级 | 方案 | 判断 |
|---:|---|---|
| 1 | 当前 backend + 两阶段 QC + NO_OP | 最小改动；满足全部门禁时优先 |
| 2 | 扩张 halo inpainting + cosine + QC | Candidate A 漏检或通过率不足时的首选升级 |
| 3 | halo + harmonization / fractional soft projection | 仅按 train-only 预注册 ablation 结果选择 |
| 4 | screened Poisson/Laplacian blending | 独立候选；solver 和吞吐验证成本更高 |
| 5 | 重训 boundary-aware G1 | 理论上限更高，但属于新模型研发，不是轻量修复 |
| 6 | 停止生成增强 | 若收益/质量门禁不成立，这是正确回退 |

### 15.3 何时必须重训 G1

若扩张 halo 后仍频繁出现内部黑块、白带或棱柱，且严格 QC 导致 pass rate 低于下限，说明
问题主要来自生成分布，不是 blending。此时不应继续叠加后处理，应另立 G1 训练路线：

- 训练时随机化 inpainting halo。
- 显式区分 label condition 和 unknown mask。
- 加入边界梯度/重建一致性目标。
- 使用独立生成质量验证和重新选择 checkpoint。

这一路线需重新评估 G1/G2，不得沿用 Fix-v2 门禁结论。

## 16. 不采用的捷径

### 16.1 删除 11 个已知失败样本或组件

不采用。R5 已证明旧失败黑名单不能防止新失败。

### 16.2 只训练 18/19 个已通过固定样本

不作为在线增强主方案。样本太少，重复使用会产生选择偏差和过拟合，且无法代表随机事件分布。

### 16.3 只收紧 compact-support

不采用。R5 在池仍充足时出现 6/24 reject，几何大小不是唯一根因。

### 16.4 只做强度 clip

不采用。clip 会隐藏而非修复生成失败，并可能删除真实高信号病灶特征。

### 16.5 只做支撑区内 3 mm 羽化

不采用。会严重削弱小病灶残差，且无法去掉内部块状极值。

### 16.6 QC 失败后不断重抽

不采用。会隐藏失败率、改变供体分布并使训练 step 时间不可预测。

### 16.7 对标签羽化

禁止。分割标签必须保持离散类别。

## 17. 工期和资源估算

| 阶段 | 估计墙钟 | 资源 |
|---|---:|---|
| 两阶段 QC、候选 policy 和测试 | 8-14 h | CPU + 少量 H20 |
| train-only reference/dev/holdout 校准 | 6-10 h | CPU + 1 H20 + 人工复核 |
| Gate-1A 100,000 | 1-2 h | 16+ CPU process |
| Gate-1B 96 事件双重放 | 15-30 min | 1 H20 |
| Gate-2 120 尝试 + 人工复核 | 1-2 h | 1 H20 + 人工 |
| Training smoke | 0.5-1 h | 1 H20 |
| Control 200 epoch | 25-31 h | 1 H20 |
| Fix-v2 200 epoch | smoke 后确定；预计 35-50 h | 1 H20，含在线生成 |
| 固定 103 评估与选模 | 3-5 h | GPU + CPU evaluator |
| 审计归档 | 1-2 h | CPU |

两臂并行时正式训练墙钟取较慢的 Fix-v2。建议完整预留 58-85 小时，不含官方 179 例最终推理。
该范围仍只是规划值；必须用 128-step smoke 的实际 backend 调用数和每次生成墙钟重新计算，
不得把 Control 的 25-31 小时直接当作 Fix-v2 ETA。

## 18. 实施清单

### 18.1 设计和代码

- [ ] 明确授权后建立全新 Fix-v2 root。
- [ ] 冻结 R1-R5、数据、split、plans、G1/G2 和源码 SHA。
- [ ] 先实现共享的 raw/candidate QC 和 Candidate A 最小基线。
- [ ] 实现冻结 boundary-policy 枚举和禁止静默回退的配置合同。
- [ ] 为 Candidate B/C 新增 `L/H/C` 几何对象和物理距离实现。
- [ ] 为 Candidate B/C 新增 S2 halo-inpainting adapter，不修改锁定 G1 源码。
- [ ] 为 Candidate B/C 验证 backend 返回 `H` 内真实生成结果。
- [ ] 实现可选的余弦 halo、median/MAD 和预注册其他候选。
- [ ] 实现 raw-generation 和 candidate 两阶段 QC。
- [ ] 实现 `F_1/F_2/F_3` 物理边界面、亚区 x 模态统计和小样本 fallback。
- [ ] 实现亚区条件的跨模态联合效应和空间对齐 QC，移除全模态统一 IoU 假设。
- [ ] 为 Candidate C 实现 gain/offset、halo amplification 和径向分壳包络门禁。
- [ ] 扩展事务状态、审计和 rollback。
- [ ] 升级 Gate-2 NPZ 和 montage。

### 18.2 校准和测试

- [ ] 创建 train-only Reference/Development/QC holdout。
- [ ] 完成单元、属性、故障注入和端到端测试。
- [ ] 完成 Run-L/Run-H 同噪声成对消融和 `H\L` 结构保持验证。
- [ ] 比较半径和 harmonization 候选。
- [ ] 完成 ET 锐边正对照、SNFH 硬边负对照和跨模态生理范围不同的正对照。
- [ ] R4/R5 11 reject 和 37 accept 回归。
- [ ] 一次性盲法 QC holdout。
- [ ] 冻结 calibration、有效增强率和性能上限。

### 18.3 门禁

- [ ] Gate-0 完整通过。
- [ ] Gate-1A 100000/100000 串并行等价。
- [ ] Gate-1B 96 事件端到端双重放一致。
- [ ] Gate-2 固定 120 分母，pass rate 达标。
- [ ] 所有 COMMITTED 真实人工 accept。
- [ ] Training smoke 128 steps 通过。

### 18.4 训练和选模

- [ ] Control/Fix-v2 各只启动一次。
- [ ] 两臂 200 epoch 和合同严格验收。
- [ ] 固定 103 官方兼容评估。
- [ ] 配对差、bootstrap CI 和分层指标完整。
- [ ] 按预注册规则选模型并归档 checkpoint SHA。
- [ ] 选定后才允许该新路线执行官方推理。

## 19. 当前决策

Fix-v2 只是一条赛后或独立窗口内的新研究路线。当前比赛主线保持：

1. MET-AUG 停止。
2. 不建 R6。
3. 不运行 R4/R5 finalize 或 Route A 训练。
4. 不打断已经锁定原 E 的官方 179 例推理。
5. 不因本文档修订自动获得任何实施授权。

## 附录 A：核心伪代码

```python
plan = plan_event(segmentation, valid_mask, context)
if plan.state != "PLACEMENT_VALID":
    return unchanged_no_op(plan.reason)

policy = frozen.boundary_policy
label_support = plan.placement.label_cube != 0

if policy == "label_only_qc_v1":
    geometry = build_label_only_geometry(label_support)
    raw = current_backend.generate(
        original_crop, plan.placement.label_cube, seed=plan.event_seed
    )
elif policy in {"halo_cosine_v1", "halo_cosine_harmonized_v1"}:
    geometry = build_halo_geometry(
        label_support=label_support,
        spacing_mm=context.spacing_mm,
        halo_radius_mm=frozen.halo_radius_mm,
        valid_mask=valid_crop,
        forbidden_mask=forbidden_crop,
    )
    if not geometry.valid:
        return unchanged_no_op("HALO_PLACEMENT_INVALID")
    raw = halo_backend.generate(
        image_crop=original_crop,
        label_crop=plan.placement.label_cube,
        inpaint_support=geometry.image_support,
        seed=plan.event_seed,
    )
elif policy == "screened_poisson_v1":
    geometry = build_label_only_geometry(label_support)
    raw = current_backend.generate(
        original_crop, plan.placement.label_cube, seed=plan.event_seed
    )
else:
    raise FrozenContractError("unsupported boundary policy")

raw_qc = compute_raw_generation_qc(
    original=original_crop,
    generated=raw.image,
    geometry=geometry,
    calibration=frozen.calibration,
)
if not raw_qc.pass_all:
    return unchanged_no_op("RAW_GENERATION_QC_FAIL", raw_qc.metrics)

pre_harmonization_candidate = None
if policy == "label_only_qc_v1":
    candidate_image = raw.image
    harmonization_metrics = {"policy": "disabled"}
elif policy == "halo_cosine_v1":
    pre_harmonization_candidate = original_crop + geometry.alpha * (
        raw.image - original_crop
    )
    candidate_image = pre_harmonization_candidate
    harmonization_metrics = {"policy": "disabled"}
elif policy == "halo_cosine_harmonized_v1":
    pre_harmonization_candidate = original_crop + geometry.alpha * (
        raw.image - original_crop
    )
    harmonized = harmonize_on_generated_halo(
        original=original_crop,
        generated=raw.image,
        context_ring=geometry.context_ring,
        calibration=frozen.calibration,
    )
    if not harmonized.valid:
        return unchanged_no_op("HARMONIZATION_FAIL", harmonized.metrics)
    candidate_image = original_crop + geometry.alpha * (
        harmonized.image - original_crop
    )
    harmonization_metrics = harmonized.metrics
elif policy == "screened_poisson_v1":
    candidate_image = screened_poisson_blend(
        original_crop, raw.image, geometry.label_support, frozen.poisson
    )
    harmonization_metrics = {"policy": "screened_poisson_v1"}

candidate_seg = original_segmentation_crop.copy()
candidate_seg[geometry.label_support] = plan.placement.label_cube[
    geometry.label_support
]

candidate_qc = compute_candidate_qc(
    original=original_crop,
    raw_generated=raw.image,
    candidate=candidate_image,
    pre_harmonization_candidate=pre_harmonization_candidate,
    candidate_segmentation=candidate_seg,
    geometry=geometry,
    calibration=frozen.calibration,
)
if not candidate_qc.pass_all:
    return unchanged_no_op(candidate_qc.reason, candidate_qc.metrics)

if not validate_transaction(
    before_image=original_crop,
    before_segmentation=original_segmentation_crop,
    after_image=candidate_image,
    after_segmentation=candidate_seg,
    image_support=geometry.image_support,
    label_support=geometry.label_support,
):
    return unchanged_no_op("COMMIT_CONTRACT_FAIL")

append_audit_before_return(
    state="COMMITTED",
    boundary_policy=policy,
    geometry=geometry.audit,
    raw_qc=raw_qc.metrics,
    harmonization=harmonization_metrics,
    candidate_qc=candidate_qc.metrics,
)
return commit(candidate_image, candidate_seg)
```

## 附录 B：Gate-2 montage 最低内容

每个 COMMITTED 事件必须显示：

- 四模态 x 三平面。
- original / raw generated / final candidate；启用 harmonization 时额外显示 harmonized。
- 显示冻结 boundary policy；Candidate B/C 显示 `L`、`H` 和 context ring，Candidate A/D 显示 `L`。
- 用固定且色盲可读的不同颜色叠加 `B_1(NETC)`、`B_2(SNFH)`、`B_3(ET)`；
  Route A 不得出现 `B_4`。
- 相同 window/level，不能分别自动拉伸掩盖极值。
- difference map、按亚区分色的 `∂L` 法向梯度图，以及 Candidate B/C 的 `∂H` 结构/残差图。
- support 体素数、halo 半径、gain/offset、径向放大曲线和每个 `(k,c)` 的关键 QC 指标与判定。

人工逐例检查：

- [ ] 无黑块、白带、硬边、矩形或棱柱。
- [ ] halo 外边缘自然衔接。
- [ ] `L` 内病灶信号未被抹除。
- [ ] 四模态病灶位置和形态一致。
- [ ] 标签与图像空间一致。
- [ ] 无脑外生成、已有病灶覆盖或通道错误。
- [ ] 三平面局部放大后仍无接缝。

人工决定只允许 `accept` 或 `reject`。不得使用“轻微问题但通过”，不得自动把 pending/reject 改成 accept。

## 附录 C：为什么 Fix-v2 比上一版更优

| 问题 | 上一版 | Fix-v2 |
|---|---|---|
| 最小修复基线 | 没有 | 先验证现有 backend + 两阶段 QC |
| halo 是否有生成值 | 没有，backend 外部复制原图 | 若选择 halo，`H` 是显式 unknown region |
| 外侧 alpha 是否有效 | `alpha * 0`，实际无效 | halo policy 融合真实 residual |
| context ring 是否可对齐 | generated=original，恒等变换 | 仅 halo policy 使用真实生成环带 |
| 内部黑块处理 | 主要依赖融合后 QC | 融合前 raw-QC 直接拒绝 |
| 病灶边界 QC | 全边界/全模态统一解释 | 按 NETC/SNFH/ET x 模态分层，事件级经验校准 |
| halo 训练-推理风险 | 只以全体素 loss 证明可行 | Run-L/Run-H 同噪声消融，结构不合格则淘汰 halo |
| harmonization | 仅检查 gain/offset | 额外检查放大比、halo-to-lesion 比和径向包络 |
| 跨模态 QC | 容易退化为统一重合要求 | 按亚区学习真实联合效应，只拒绝经验矛盾/错位 |
| 3 mm 参数 | 预设固定 | train-only 候选比较后冻结 |
| NO_OP 影响 | 只报告 | 冻结有效增强率下限 |
| Gate-2 分母 | 只要求 48 COMMITTED | 固定 120 backend attempts，禁止补抽 |
| 统计解释 | 48/48 即通过 | 报告零失败精确上界和实际分母 |

因此，Fix-v2 不是“更复杂所以更好”。它先选择最小充分修复；只有证据要求时才增加 halo 或
gradient-domain blending，并把边界问题、生成内容问题和 QC 失效分别处理。
