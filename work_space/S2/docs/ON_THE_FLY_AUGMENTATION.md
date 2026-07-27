# On-the-Fly 肿瘤增强策略

> **阅读边界（2026-07-25）**：本节前半部分复述 BraTS 2025 论文的在线调度思想及本项目早期适配设想，
> 不是当前可执行参数。真正用于 BraTS-MET 的唯一执行规范从“转移瘤 Route A-F 分期消融路线”开始。
> 当前只实现并准备放行 `MET-AUG-A`；其冻结值为 `p_select=0.20`、单个真实组件、`scale=1.0`、
> 不改类别，不得把下面的 `0.6/0.75`、激进缩放或第二病灶带入 Route A。

论文快照：`docs/BraTS 2025.pdf`，标题 *On-the-Fly Data Augmentation for Brain Tumor Segmentation*，
SHA256 `18fd9192f118422cd05fcc164afb9d53a2ce3558715fc11aec35fb623d0c1c8b`。

BraTS 2025 的 Custom 模型对送入 nnU-Net 训练的样本以 `p=0.6` 动态合成病灶，验证样本保持不变。
以下五步用于说明论文策略与本项目生成后端的对应关系。

---

## 第一步 — 借入标签

以 **60%** 概率选中当前样本进行增强。

从训练集中**随机借入**一个其他病人的真实标签（seg 文件），提取其肿瘤区域作为"借入灶"。插入位置在目标样本的脑内空白区域（与原有肿瘤不重叠、非 padding 区域、脑组织内）。

> 实现要点：在目标样本 z-score 归一化后的 4 通道 MRI 中搜索 crop_size³ 窗口，要求窗口内 seg 全为 0（无现有肿瘤）且 T1c 通道不全为 0（脑内）。

---

## 第二步 — 类别替换

对借入标签中的肿瘤体素，执行两类级联替换（各自以 **70%** 概率触发）：

| 步骤 | 操作 | 概率 |
|---|---|---|
| 2a | SNFH (2) → ET (3) | 70% |
| 2b | **原始** ET (3) → NETC (1) | 70% |

> 注意：步骤 2a 中 SNFH 改来的 ET **不参与**步骤 2b 的 ET→NETC 转换。步骤 2b 仅作用于借入标签中"原本就是 3"的体素。RC(4) 不做修改。

---

## 第三步 — 差分缩放

根据第二步是否移除了 SNFH，采用不同的缩放系数范围，对借入灶做以质心为锚点的各向同性缩小：

| 条件 | 缩放系数范围 |
|---|---|
| SNFH **被移除**（步骤 2a 触发且实际有 SNFH 体素被改） | **10% – 30%** |
| SNFH **未被移除**（步骤 2a 未触发，或借入灶本身无 SNFH） | **30% – 80%** |

> 缩放后每个维度至少保留 1 个体素。使用 scipy.ndimage.zoom(order=0) 保持标签为整型。

---

## 第四步 — 第二个肿瘤

对于已在第一步中插入了一个肿瘤的样本，以 **40%** 概率插入第二个肿瘤：

1. **重新独立借入**一个标签（从训练集中再次随机采样，与第一次借入的来源和插入位置均无关）
2. 对第二个借入标签**独立执行第二步**（类别替换）和**第三步**（差分缩放）
3. 插入位置同样要求与现有肿瘤（原有的 + 第一个插入的）不重叠

> 两个肿瘤的类别替换和缩放系数**各自独立随机**。最多插入 2 个肿瘤。

---

## 第五步 — 本项目以 Tumour Diffusion 替换论文 GliGAN 后端

**不再使用** label diffusion 生成标签（标签由第一至第四步决定）。

BraTS 2025 原文使用预训练 GliGAN，在每个训练 step 前按条件标签插入肿瘤；本项目不复用其 GAN 权重，
而是接入已经通过 G1/G2 评估的四模态 Diffusion。可继承的是“训练期动态生成、验证不变、未选中样本原样
返回”的调度原则，不是生成器实现。

对每个插入的肿瘤，使用 **tumour diffusion model**（已训好的 4 模态扩散模型）在标签确定的肿瘤轮廓内部生成 MRI 组织：

1. 提取目标样本中插入位置周围的 crop_size³ 区域（4 通道 z-score 数据）
2. 将借入并修改后的肿瘤标签对应区域取出（crop_size³ 二值或多类 mask）
3. 对该区域的肿瘤体素加高斯噪声 → 得到 noisy scan
4. 以肿瘤标签为条件，送入 tumour diffusion 模型执行 inpainting 采样
5. 论文/旧 G1 实现曾做零值哨兵式背景修正和强度校正；当前 Route A **不调用**该背景修正，
   只按显式病灶 support 写回，support 外逐体素保持原图

> 每个模态（T1c, T1n, T2w, T2f）使用各自独立的 diffusion checkpoint。生成仅在肿瘤轮廓内部进行，轮廓外保持不变。

---

## 流程总览

```
Batch 中的某个样本
  │
  ├── [40%] 不增强 → 原样返回
  │
  └── [60%] 增强路径
        │
        ├─ Step 1: 随机借入一个其他病人的 seg
        │     ├─ 提取其肿瘤区域作为借入灶
        │     └─ 在目标样本脑内找不冲突的插入位置
        │
        ├─ Step 2: 类别替换（借入灶内部）
        │     ├─ [70%] SNFH → ET
        │     └─ [70%] 原始 ET → NETC
        │
        ├─ Step 3: 差分缩放
        │     ├─ SNFH 被移除 → 缩至 10%-30%
        │     └─ SNFH 保留   → 缩至 30%-80%
        │
        ├─ Step 4: [40%] 第二个肿瘤
        │     ├─ 重新借入、独立执行 Step 2+3
        │     └─ 插入位置避开所有已有肿瘤
        │
        └─ Step 5: Tumour Diffusion 生成
              ├─ 每个模态独立 inpainting
              ├─ 在标签轮廓内生成组织
              └─ 轮廓外保持原始 MRI
```

---

## 关键参数汇总

| 参数 | 值 | 说明 |
|---|---|---|
| `p_select` | 60% | 选中某样本进行增强的概率 |
| `p_snfh_to_et` | 70% | SNFH → ET 替换概率 |
| `p_et_to_netc` | 70% | 原始 ET → NETC 替换概率 |
| `scale_snfh_removed` | [0.1, 0.3] | SNFH 被移除时的缩放范围 |
| `scale_snfh_kept` | [0.3, 0.8] | SNFH 保留时的缩放范围 |
| `p_second_tumour` | 40% | 插入第二个肿瘤的概率 |
| `max_tumours` | 2 | 每个样本最多插入的肿瘤数 |
| `crop_size` | 64 | 扩散模型 inpainting 的立方体尺寸 |

---

## 与现有代码的对应关系

| 步骤 | 现有代码位置 |
|---|---|
| 1-4 | `Segmentation_Tasks/GliGAN/src/utils/label_modifier.py` — `modify_borrowed_label()` |
| 5 | `Segmentation_Tasks/GliGAN/src/infer/diffusion_inference_utils.py` — `sample_tumour_diffusion_inpaint()` |
| 整体调度 | `Segmentation_Tasks/GliGAN/src/infer/on_the_fly_augmentation.py` — `OnTheFlyTumourAugmenter.augment_sample()` |

---

## BraTS 2025 与当前 Route A 的取舍

| BraTS 2025 做法或结论 | 当前处理 | 依据 |
|---|---|---|
| 仅在训练 step 动态插入；validation 不变 | 继承 | 合成分布不能进入模型选择集。 |
| `(1-p)` 样本逐元素不修改 | 继承并强化为 bit-identical no-op | 保留真实分布，并可审计失败回滚。 |
| donor 从其他患者选择 | 继承并扩展为不同 `patient_group` | 同一患者不同时间点也必须隔离。 |
| 在 nnU-Net 训练增强链中执行 | 继承；在单 patch、空间/强度增强之前执行 | 标签、四模态与 valid mask 仍处于同一标准空间。 |
| Regular `p=0.75`、Custom `p=0.60` | 不照搬；Route A 固定 `p=0.20` | 先限制伪影暴露，再由独立 Route C 检验剂量。 |
| `SNFH->ET`、原始 `ET->NETC`，概率 0.7 | Route A 禁用 | 胶质瘤类别拓扑不能直接迁移到转移瘤。 |
| 线性缩放 `0.3-0.8`，Custom 可到 `0.1-0.3` | Route A 禁用，`scale=1.0` | 小转移灶会被压到不可学习区；尺度只能在 Route B 单独消融。 |
| 第二病灶概率 0.4 | Route A 禁用 | 多灶变量只能在 Route D 单独检验。 |
| GliGAN 作为生成后端 | 替换为冻结的 G1 四模态 150k Diffusion | 当前可用父证据是 z-score、64^3、EDM-Heun 18 steps 的 G1/G2 gate。 |
| Custom 单模型内部 lesion-wise 平均排名 5.67；论文承认概率/尺度随机选取且训练未完全收敛 | 作为风险证据，不作为最优参数证据 | 论文支持在线增强的可行性，但不证明激进 Custom 参数适合 BraTS-MET。 |

因此当前最稳路线不是一次整合论文全部技巧，而是先用 `MET-AUG-A` 证明“真实组件平移 + 已验收四模态
Diffusion + 原子写回”相对同预算 `E-continue` 有净收益；原始 E 只作为部署锚点。只有 A 通过 Gate 3，
才逐项开放 B-F。

---

# 转移瘤适配方案

## 转移瘤 vs 胶质瘤：根本差异

以下基于 1296 例 BraTS-MET 的病灶池统计（9118 个病灶）分析。

### 病灶数量与大小分布

| 指标 | 转移瘤 | 胶质瘤 |
|---|---|---|
| 每患者病灶数 | 中位 3 / 均值 7 / P95=24 / max=312 | 通常 1 个 |
| 病灶体积 (voxels) | 中位 99 / P5=4 / P25=23 / P75=679 / P95=41723 | 通常数万 ~ 数十万 |
| Bbox 最大维度 | 中位 8 / P95=68 / >64: 5.5% | 通常 > 100 |

74% 的患者是多发病灶，分布严重右偏。转移瘤病灶体积中位值只有胶质瘤的几十分之一，甚至百分之几。

### 类别分布

| 指标 | 转移瘤 | 胶质瘤 |
|---|---|---|
| ET 出现率 | **96.6%** | > 95% |
| SNFH 出现率 | **46.5%** | > 95% |
| NETC 出现率 | **19.2%** | ~ 90% |
| RC 出现率 | 2.3% | 仅 post-treatment |

### 类别组合模式

| 模式 | 病灶数 | 占比 | 中位体积 |
|---|---|---|---|
| **纯 ET** | 4569 | **50.1%** | 27 |
| SNFH + ET | 2380 | 26.1% | 307 |
| NETC + SNFH + ET | 1485 | 16.3% | 4966 |
| NETC + ET | 234 | 2.6% | 220 |
| 纯 SNFH | 224 | 2.5% | — |
| 其他 | 226 | 2.5% | — |

### 胶质瘤级联为什么对转移瘤反效果

胶质瘤中 ET、NETC、SNFH 形成严格的嵌套洋葱结构（NETC ⊂ ET ⊂ SNFH）。原方案的 SNFH→ET→NETC 级联是一个**向内收紧的类均衡操作**：把过剩的 SNFH 挤入 ET、把 ET 挤入稀缺的 NETC，同时借助天然的嵌套拓扑避免产出不合理的组织排列。本质是从外向内借壳充填。

转移瘤不仅没有这个嵌套结构，三者的空间关系完全不同：

- **纯 ET 占 50.1%**：没有 SNFH 包围、没有 NETC 核心，ET 就是肿瘤的全部。级联对它完全空转（无 SNFH 可改、无 NETC 可替换），借来的纯 ET 原样插入，在增强数据中继续放大已经 96.6% 的 ET 占比。
- **SNFH+ET 占 26.1%**：SNFH 通常偏在一侧而非完整包围，层间也没有严格的包含关系。SNFH→ET 把 SNFH 转成 ET，ET 已经过剩，增了个寂寞。ET→NETC 把部分 ET 转成 NETC → NETC+ET（2.6% 的稀有模式），方向不差但规模太小。
- **NETC+SNFH+ET 占 16.3%**：三层出现在同一病灶，但坏死可以偏心、可以是裂隙状而非中心球形。原级联把 SNFH→ET→NETC，产出含大量 NETC 的模式，但纯 NETC（0.2%）和 NETC 为主的形态在转移瘤中极其罕见。

核心结论：原方案假设了"从外层过剩标签向内层稀缺标签借壳"的置换逻辑，转移瘤必须改为"从简单模式向复杂模式构造"。不是再分配已有的组织，而是从无到有新增缺失的组织。

---

## 第一步改进 — 选中概率按原生病灶数自适应

**问题**：固定 60% 概率下，只有 1 个病灶的患者和已有 50 个病灶的患者获得同样的增强概率。病灶极少的患者最需要增强，病灶极多的患者再增强 1-2 个几乎没有增量价值。

**改进**：

| 原生病灶数 N | p_select | 理由 |
|---|---|---|
| 0 – 1 | **80%** | 24% 的患者仅 0-1 个病灶，增强边际价值最高 |
| 2 – 4 | **50%** | 中位 3 个，这组患者正常偏少 |
| 5 – 9 | **30%** | 已超过中位，适度减少增强 |
| ≥ 10 | **15%** | 8% 的患者原发病灶已足够多，增强接近无意义 |

N 在标签池构建阶段（一次性离线）与患者 ID 一起索引，训练时查表，零在线开销。

---

## 第二步改进 — 定向类别转换（从置换到构造）

**问题**：胶质瘤的级联置换依赖嵌套拓扑，转移瘤中不成立。原级联对纯 ET（50.1%）完全空转，SNFH→ET 方向与真实分布相反（ET 已 96.6%），ET→NETC 方向对但产出模式不真实。

**原则**：不再在已有类别间做和为零的替换，而是从简单模式向复杂模式**新增加缺失的组织**。目标是把各模式的产出分布校准到接近真实分布。

### 模式鉴别

借入标签的 seg 文件本身带完整类别标注。取病灶非零区域做 `np.unique()` 即可判定：

```python
labels_present = set(np.unique(lesion_mask[lesion_mask != 0]))
# {3}         → 纯 ET
# {2, 3}      → SNFH+ET
# {1, 2, 3}   → NETC+SNFH+ET
# {1, 3}      → NETC+ET
# {2}         → 纯 SNFH
```

一次 unique 约 0.3ms，数据量最多 60³ = 216,000 体素。

### 各模式转换规则

| 借入灶模式 | 占比 | 转换路径 | 产出分布 |
|---|---|---|---|
| **纯 ET** | 50.1% | ① 50% 保留纯 ET | 维持基准多样性 |
| | | ② 35% 外围 **加 SNFH 边缘** | binary dilation (球形, 半径 1-3 voxel)，新增体素设为 SNFH(2)，产出 SNFH+ET |
| | | ③ 15% 核心 ET **加 NETC** + 外围加 SNFH | 先对 ET mask 做 1-2 voxel erosion 产生 NETC(1)（保证被 ET 完全包围、不强制几何中心），再 dilation 加 SNFH 边缘，产出 NETC+SNFH+ET |
| **SNFH+ET** | 26.1% | ① 60% 保留 | 不变 |
| | | ② 40% 核心 ET **加 NETC** | erosion 1-2 voxel → NETC(1)，SNFH 保留，产出 NETC+SNFH+ET |
> 去掉了"削除 SNFH → 纯 ET"的退化路径。纯 ET 已占真实分布 50.1%，模型不缺这类样本；SNFH 仅 46.5% 出现率，每一例都更珍贵。
| **NETC+SNFH+ET** | 16.3% | ① 80% 保留 | 维持三层结构 |
| | | ② 10% 去 NETC → SNFH+ET | NETC 体素设为 0 |
| | | ③ 10% 去 NETC+SNFH → 纯 ET | NETC、SNFH 体素设为 0 |
| **纯 SNFH** | 2.5% | ① 60% SNFH + 内部 ET 核心 | 对 SNFH mask erosion 产生内层 ET(3)（模拟增强灶），产出 SNFH+ET |
| | | ② 40% 保留 | 纯 SNFH |
| **NETC+ET** | 2.6% | 保留 | 已是真实存在的模式 |

> RC 类（2.3%）不做任何类别转换，保持原样。

### NETC 生成方式：形态学腐蚀（非距离变换）

对于"核心 ET → NETC"操作，使用 binary erosion 而非 distance transform：

| 方法 | 行为 | 对转移瘤的适用性 |
|---|---|---|
| distance transform + 取内层百分比 | 坏死集中在几何中心 | 假设中心性坏死（胶质瘤模式），对偏心、多灶、不规则坏死不真实 |
| **binary erosion (1-2 voxel)** | 均匀削掉 ET 边缘，坏死跟随轮廓分布 | 哑铃形病灶两个叶各有坏死、不规则形坏死跟随边界走，匹配转移瘤多灶坏死机制 |

转移瘤的坏死是血供不均导致的部分组织缺血，通常不是单中心的规整球体，而是裂隙状、偏心状、多个小灶。erosion 均匀从外向内削，形状不规则的区域削几层后自然留下不规则的残留，与坏死的物理形成过程同构。

```python
from scipy.ndimage import binary_erosion
et_mask = (label == 3)
depth = rng.integers(1, 3)  # 1-2 voxel，随机
struct = np.ones((3, 3, 3))
eroded = binary_erosion(et_mask, structure=struct, iterations=depth)
netc_mask = et_mask & ~eroded
label[netc_mask] = 1
```

剩余 ET 是包绕在 NETC 外围的薄壳（深度 1-2 voxel），对应 T1C 增强环——与环形强化转移瘤的影像一致。

### SNFH 边缘的生成方式

```python
from scipy.ndimage import binary_dilation
tumour_mask = (label != 0)
radius = rng.integers(1, 4)  # 1-3 voxel
struct = np.ones((3, 3, 3))
dilated = binary_dilation(tumour_mask, structure=struct, iterations=radius)
new_snfh = dilated & ~tumour_mask
label[new_snfh] = 2
```

SNFH 只加在肿瘤外围（不与现有肿瘤重叠），厚度 1-3 voxel 模拟转移瘤的局灶性血管源性水肿。

### 操作顺序

始终先改内部（ET→NETC erosion）、再加外部（SNFH dilation）。如果先 dilation 后 erosion，新增的 SNFH 体素会干扰 erosion 的距离判断，导致 NETC 位置偏移。

### 计算开销

所有形态学操作（unique / count_nonzero / erosion / dilation）对一个 60³ 的二值数组合计 < 3ms。扩散模型 4 模态 50 步 inpainting 约 500-2000ms。新增逻辑占总耗时 < 0.5%，同时替代了原有的 `ndimage.zoom` 缩放（约 1-2ms），净增接近零。

---

## 第三步改进 — 体积分档缩放 + 允许放大 + 硬地板 27

**问题**：胶质瘤原方案始终缩小（0.1-0.8），受 SNFH 状态调制。对转移瘤这是反效果的：

- 纯 ET 中位 27 voxels × 0.3 = 8 voxels × 0.1 = 2.7 voxels，几乎消失
- 基线实验证明，< 27 voxels 的病灶在 5 层下采样架构中物理上无法被有效学习（bottleneck < 0.1 个特征单元）
- 制造更多微型病灶不是在增加训练信号，是在注入噪声

同时转移瘤 bbox 中位 = 8，距离 64³ crop cap 有 56 voxel 的巨大余量。病灶可以放大而不触发超窗限缩。

**改进**：三档体积分档，允许放大，体积地板 = 27。

| 原病灶体积 | 缩放范围 | SNFH 微调 | 理由 |
|---|---|---|---|
| < 50 voxels | **80% – 120%** | 偏放大（80-120%） | 小灶放大给模型更多体素信号，同时保证不超窗（最大 60，远小于 64） |
| 50 – 500 voxels | **40% – 80%** | SNFH 移除偏缩小（40-60%），SNFH 保留偏中等（55-80%） | 中灶弹性最大，有缩小有放大 |
| > 500 voxels | **10% – 50%** | SNFH 移除偏小（10-35%），SNFH 保留放松（25-50%） | 大灶还是需要缩小防止超窗，但不再极端 |

**体积地板**：缩放后体积不得低于 27 voxels（3×3×3）。27 对应 5 层下采样架构下物理存在的最小方形病灶（bottleneck 约 0.2 个单元 vs < 27 时不到 0.1）。地板触发时直接钳在原体积，不缩放。

地板从原文档建议的 8 上调到 27，理由：8 = 2×2×2 在 encoder 第三层就消失了对训练无实际贡献；27 = 3×3×3 是跨过 5 层下采样的生存门槛。

**SCALE** 操作的缩放因子在工单中传为元信息，代替原有 `snfh_removed` 决定缩放区间。

---

## 第四步改进 — 第二肿瘤概率与插入后总病灶数联动

**问题**：固定 40% 不考虑上下文。原始 1 个病灶插完第一个 → 2 个（偏少），原始 50 个插完第一个 → 51 个（无增量）。

**改进**：

| 插入第一个后总病灶数 | 第二肿瘤概率 | 理由 |
|---|---|---|
| 1 – 2 | **50%** | 还不到中位 3 个，第二个边际价值高 |
| 3 – 5 | **30%** | 已达中位附近 |
| 6 – 9 | **15%** | 已偏多 |
| ≥ 10 | **5%** | 几乎无价值，大部分不触发 |

与第一步联动：第一步按原生病灶数降选中概率（病灶多→少增），第四步对已增一次的样本以递减概率考虑第二次。双重控制下，原发 50+ 病灶的患者被选中且插入两次的概率趋近于零。

---

## 第五步 — 标签池构建

**排除微病灶**：从可借入池中排除体积 < 27 voxels 的病灶。1194 个病灶体积在 1-10 voxels，借入后经第二步调整仍可能极小，第三步地板钳在原体积，整个过程浪费一次 inpainting 但产出无训练价值的微小目标。

**log 体积加权采样**：对剩余病灶按 `log1p(volume)` 加权，避免极端大病灶（P99=201k voxels）过度主导借入池，同时让借入倾向 50-5000 voxels 的有效区间。

构建在 `OnTheFlyTumourAugmenter.__init__` 时一次性完成：

```python
volumes = [max(lesion['n_voxels'], 27) for lesion in pool
           if lesion['n_voxels'] >= 27]
weights = np.log1p(volumes)
weights /= weights.sum()
# 采样: np.random.choice(len(pool), p=weights)
```

在线采样一次约 0.01ms。

---

## 第六步 — Tumour Diffusion Inpainting（不变）

逻辑不变：以修改后的肿瘤标签为条件，在每个模态独立做扩散 inpainting，在标签轮廓内生成组织，轮廓外保持原始 MRI。前提是 diffusion model 使用 z-score 归一化训练，与 nnUNet 预处理一致。

---

## 改进汇总

| 步骤 | 原有（胶质瘤） | 改进后（转移瘤） | 改进原因 |
|---|---|---|---|
| Step 1 | 固定 p=60% | 自适应 15%-80%，按原生病灶数分四档 | 病程差异巨大，固定概率浪费计算 |
| Step 2 | SNFH→ET→NETC 洋葱式内向置换 | 按借入灶类别模式定向从简到繁构造（erosion + dilation） | 转移瘤无嵌套层级，ET 过剩无需从 SNFH 转，需从无到有新增缺失组织 |
| Step 3 | 0.1-0.8 始终缩小，SNFH 状态决定 | 三档体积分档 0.1-1.2 + 允许放大 + 硬地板 27 | 小灶缩小后架构无法学习；bbox 中位 8 有大量放大空间 |
| Step 4 | 固定 p=40% | 按插入后总病灶数自适应 5%-50% | 第二肿瘤边际价值随已有病灶数递减 |
| 新增 | — | 标签池排除 < 27 voxels + log 体积加权采样 | 微尘埃对 inpainting 无训练价值还浪费推理 |
| Step 6 | Tumour diffusion | 不变 | 需 z-score 重训 |

---

# 转移瘤 Route A-F 分期消融路线

> **本节是唯一可执行规范**：上文的“转移瘤适配方案”保留为病理和机制分析，不能直接作为参数表或代码行为。若上文与本节在 `p_select`、类别转换、缩放、通道顺序或失败处理上冲突，以本节为准。
>
> **边界**：本路线只定义后续训练 patch 内的合成病灶增强；不改变 Dataset264 的固定
> `1035/103/104` 划分，不触碰验证、locked test、官方 179 例推理或提交。Route A 的事务、
> nnU-Net 2.8 接口、Gate 1/2 工具、route approval、训练/推理隔离和本地测试已经实现；ECNU 上的
> 真实组件池、Gate 结果、人工复核和 Route A 训练尚未执行。当前未授权启动真实 Diffusion 或训练。
>
> **命名**：为避免与 S2 的 Deep Supervision `D` 混淆，文中的 Route A-F 对应
> `MET-AUG-A` 至 `MET-AUG-F`。它们不是当前 S2 基座消融中的 A-1、E、D。

---

## 0. 总体原则和可归因性

目标不是“尽量造更多肿瘤”，而是在不破坏真实转移瘤的大小、组成、空间关系和四模态一致性的前提下，
检验一个新增变量是否减少小病灶漏检。每一轮只允许修改一个已冻结的维度；任何失败都保留上一个已证实
有效的配置，不以机制假设替代验证结果。

`lesion_pool.csv` 只可用于发现问题，不能作为供体池或直接冻结概率：它有 9,118 行，其中
4,085 行来自 `locked_test`，且原始体素空间混有多种 spacing。因此其中的类别比例、病灶大小和按
患者病灶数的关系只能提出假设，不能进入 Route 的训练采样器。

### 0.1 路线前置关系

```text
冻结的原始 E checkpoint（部署锚点）
                 |
                 +--> E-continue：同预算、p=0 的因果对照
                 |
                 +--> MET-AUG-A：同预算、p=0.20
                              |
               MET-AUG-B -> MET-AUG-C -> MET-AUG-D -> BestLowerRisk
                                                            |          |
                                                  MET-AUG-E            MET-AUG-F
                                               （类别拓扑支路）     （频率分配支路）
```

- `MET-AUG-A` 至 `MET-AUG-D` 是低至中风险的串行消融；若某轮不通过，回退到上一轮最佳。
- `MET-AUG-E` 与 `MET-AUG-F` 都从 `BestLowerRisk` 独立出发，不能让 F 继承 E 的人工标签构造。
- 不报告 `E+F` 的“协同”结论。只有 E、F 各自独立通过后，才可另立一个有对照的
  `MET-AUG-EF` 交互实验；它不属于 A-F 的默认路线。

这样可避免把“类别构造有效”误判为“按病灶数调概率有效”，也避免高风险 E 的失败阻断低风险 F 的检验。

---

## 1. 第零步：不可变单病灶组件池

所有 Route 共享一个版本化的 `component_manifest`，由固定 Dataset264 的 **1,035 例训练集**一次性生成。
组件标签必须从 S2 实际读取的 `nnUNetPlans_3d_fullres` 预处理分割中提取，而不是直接把 nibabel 的
`X,Y,Z` 数组当成 nnU-Net 的 `Z,Y,X` 数组；raw NIfTI 仅用于记录源图像、标签、affine 和 spacing 的哈希。
manifest 生成完成后写入组件、来源、分层和配置 SHA256；在整个 A-F 消融中不得重建、补样或从验证/测试集中
补齐稀有类型。

| 项目 | 规范 | 原因 |
|---|---|---|
| 空间 | Dataset264 的 `nnUNetPlans_3d_fullres`、`1 x 1 x 1 mm` 实际预处理空间 | 体积、轴序、裁剪坐标、边距和形态学半径必须与 S2 patch 完全一致，不能混用 raw voxel。 |
| 组件锚点 | 对 `NETC(1) union ET(3)` 做 26-connectivity | 以可确认为肿瘤核心的区域拆分多发病例，避免整例标签被整体借走。 |
| SNFH 归属 | 仅将与唯一最近核心相连/邻近的 SNFH 赋给该组件；存在并列归属或无核心关联时不作为供体组件 | 避免一片水肿跨多个病灶被错误粘成一个“假大灶”。 |
| 患者隔离 | `BraTS-MET-xxxxx-000/-001` 归为同一 `patient_group`；donor 与 target 不得同组 | 防止同一患者跨时间点的信息泄漏。 |
| 供体 split | 只能是 `train` | 验证集、locked test 和官方测试均为 0 条供体。 |
| 组件尺度 | 使用 `core_volume_mm3 = volume(NETC union ET)` 分档，同时保留总组件体积和 bbox | 大片 SNFH 不得掩盖肿瘤核心实际过小。 |

### 1.1 硬排除条件

| 排除 | 理由 |
|---|---|
| 任意 RC(4) | RC 与术后/治疗背景强绑定，随机迁移会制造缺乏病理上下文的标签。 |
| 纯 SNFH（无 NETC/ET 核心） | 不能可靠地当作转移瘤主体，也不能在 E 中凭空补出 ET。 |
| `core_volume_mm3 < 27` | 这是当前小病灶基座的最低可学习候选尺寸；超小结构先不由合成增强承担。 |
| 组件 bbox 任一维 `> 56 mm` | 为 `64^3` inpainting crop 留出上下文，禁止事后硬压进 crop。 |
| 标签不属于 `{0,1,2,3,4}`、空间/哈希异常或核心为空 | 标签语义、来源或几何不可信时宁可不用。 |

因此 Route E 中“纯 SNFH 转 ET”以及“RC 单独处理”都是不可达分支，必须删除，不能在实现中保留隐式后门。

### 1.2 manifest 最小字段

| 字段 | 用途 |
|---|---|
| `component_id`、`manifest_version`、组件标签 crop SHA256 | 每次插入可精确回溯到唯一供体。 |
| `source_case_id`、`patient_group`、`split=train` | 审计 split 与跨 timepoint 隔离。 |
| `core_volume_mm3`、`total_volume_mm3`、`bbox_mm` | 体积分档、缩放和 crop 门禁。 |
| `class_counts`、`classes_present` | 保持大小条件下的真实组成关系。 |
| `core_centroid_norm`、训练脑区有效掩膜版本 | 建立位置先验，避免全脑均匀随机。 |
| 源 seg/四模态 SHA256、spacing、affine 摘要 | 发现数据漂移、通道错配和不可复现实验。 |

---

## 2. 单病灶事务：先验证，后提交

一次“插入一个病灶”是独立事务，而不是直接在 nnU-Net 输入数组上逐通道写入。任何状态失败都返回输入的
bit-identical 副本，并记录 reason code；只有四模态和标签全部通过后才一次性提交。

| 状态 | 必须检查 | 失败行为 |
|---|---|---|
| `SELECTED` | 随机数命中该 Route 的 `p_select` | `NOT_SELECTED`，原样返回。 |
| `DONOR_VALID` | manifest、split、`patient_group`、类别和尺度门禁 | `NO_ELIGIBLE_DONOR`，原样返回。 |
| `LABEL_VALID` | Route 变换后的标签仍只含合法类、核心非空、体积/bbox/拓扑通过 | `LABEL_INVALID`，原样返回。 |
| `PLACEMENT_VALID` | 用**最终**标签做脑内、原病灶和已插入病灶 clearance 检查 | `NO_VALID_PLACEMENT`，原样返回。 |
| `FOUR_MODAL_SUCCESS` | 四个必需 checkpoint 都输出有限、同坐标、同掩膜、同形状的结果 | 任一失败即 `MODALITY_QC_FAIL`，原样返回。 |
| `COMMITTED` | 写掩膜外原值不变，seg 与四模态 QC 一致，审计写入成功 | 原子地提交 `data` 与 `seg`。 |

### 2.1 冻结的通用约束

| 参数 | 冻结规范 | 理由 |
|---|---|---|
| 通道合同 | `0000=t1n, 0001=t1c, 0002=t2w, 0003=t2f` | 必须匹配 Dataset264；不得复用旧代码的 `t1c,t1n,t2w,t2f` 顺序。 |
| 模态完整性 | 四个 checkpoint 和四个输出必须齐全；生产配置禁止 partial mode | 部分模态写回会形成跨模态自相矛盾的样本。 |
| 脑区/边界 | 使用 Dataset264 预处理产生的 brain/valid mask，不用单通道 `intensity != 0` 近似；距脑边界至少 `3 mm` | z-score 后合法脑内体素可为 0，旧强度启发式会误判。 |
| 间隔 | 距原生标签 `{1,2,3,4}` 和已提交合成灶的最终 support 至少 `5 mm` | 防止病灶粘连、计数改变和标签覆盖。 |
| crop | 最终 bbox 任一维 `<=56 mm`，inpainting crop 固定 `64^3` | 保证有真实上下文，禁止隐式压缩。 |
| 位置搜索 | 最多 50 次；每次均以最终 support mask 重算 clearance | 缩放/膨胀后再检查，不能只检查原 donor。 |
| 写入 | `tmp_data = data.copy()`、`tmp_seg = seg.copy()`；四模态都 QC 后再替换 | 消除旧实现“前几个通道已写、后一个失败”的半成品风险。 |
| 随机性 | `SHA256(global_seed, epoch, rank, worker, case_id, patch_index, route_id)` | 多 worker 下可以重放每一次事件。 |
| 审计 | 成功和失败均写 JSONL：状态、reason、donor/target、位置、transform、哈希和最终尺寸 | 真实增强率不能由配置概率代替。 |

`MET-AUG-D` 有两个候选病灶时，**每个候选病灶各自是事务**：第一灶已成功而第二灶失败时，保留第一灶并记
`SECOND_NO_OP`；绝不保留第二灶的部分模态或部分标签。Gate 和结果报告必须给出最终插入 `0/1/2` 灶的比例。

### 2.2 冻结的抽样分布

每个 Route 在 Gate 1 之前冻结一个 `Q_route(classes_present, core_volume_bin)`，并把 JSON、组件 manifest SHA256
和随机种子一起归档。`Q_route` 是该 Route **预期的最终候选分布**，不是旧 CSV 的原始行比例。

通用抽样次序为：先按 `Q_route` 选大小/组成分层，再在该层内均匀抽 `patient_group`，最后均匀抽该组的合格组件。
这既温和提高可学习小核心的暴露，也不会让多灶患者因组件行数更多而支配 donor 池。禁止：

- 用 `log1p(volume)` 直接给整行组件加权；它会将样本推向大病灶。
- 将体积与组成独立抽样；这会破坏真实的 `P(classes_present | core_volume_bin)`。
- 在 A-D 中按原生病灶数再调 `p_select`；这属于 F 的唯一变量，不能提前混入。

---

## 3. 三级 Gate：先证安全，再证有效

每个 Route 或 Route 子分支均须重新通过 Gate 1-3，且只能使用固定的 103 例验证集做最终比较。

### Gate 1：100,000 次无 Diffusion 策略模拟

固定 seed 模拟选择、分层抽样、标签变换和基于预计算有效掩膜的位置决策，不调用 Diffusion，也不修改训练数组。
正式执行允许把同一冻结事件流按连续 `patch_index` 确定性分片到多个 `fork` worker；合并时必须恢复原事件顺序，
并在真实资产前缀上证明串行、并行 `gate1_events.jsonl` 和报告逐字节一致。并行只缩短墙钟，不得减少 100,000 次事件、
改变 seed、阈值、目标 case 序列或审批格式。

必须满足：

1. `val/locked_test` donor、同 `patient_group` donor、RC donor、纯 SNFH donor、超小核心和超 bbox donor 均为 0。
2. `selected_rate` 与配置的 `p_select` 绝对偏差不超过 `0.005`。这只约束“被选中”，不把它误写为真实提交率。
3. 报告 `selected -> donor_valid -> label_valid -> placement_valid` 的逐级数量、转化率、失败 reason 和位置尝试次数。
4. 用最终标签复测 `core_volume_mm3 >=27`、bbox、类别和 clearance；不允许只验证变换前 donor。
5. 相对冻结的 `Q_route`，有效候选的每个分层频率最大绝对偏差不超过 `1` 个百分点。若某层在 train manifest 中不存在，必须显式标记为不可用，不能从其他 split 补齐。
6. Route B/D/E 的 fallback、第二灶失败和构造分支覆盖率必须单列报告；高 fallback 率意味着该 Route 尚不可解释，不能直接训练。

### Gate 2：固定分层四模态 smoke

使用预注册的 seed、target case 和 donor component 做至少 24 个离线样本：`27-49 mm3`、`50-275 mm3`、`>275 mm3`
三个核心体积分层各至少 8 个。每个在 manifest 中有足够供体的主要组成分层都至少出现 2 次；若供体不足，只能
报告缺口，不能挑选“好看”的替代样本。

每例必须保留原始与增强后的四模态三视图、seg overlay、donor/target/位置/route/seed/checkpoint/哈希和自动 QC：

- 四个输出均 finite，shape、dtype、坐标变换和写入掩膜一致；四模态均使用同一个最终标签和 crop 坐标。
- 写掩膜外逐体素保持不变；seg 只在事务的最终 support 内新增标签。
- 标签仅含 `{0,1,2,3,4}`，最终体积、bbox、边界和病灶间隔均再次通过。
- 人工复核四模态是否有接缝、脑外插入、标签-影像错位、通道顺序错误、异常强度或病灶粘连。

任一确定性错误、任何 partial-modality 提交或未解释的跨模态矛盾都阻断该 Route。修复后必须重跑同一固定 smoke 集，
不能替换为新 seed 掩盖失败。

当前 Gate 2 工具进一步冻结为：三个体积分档各 8 例，24 个 target 和 24 个 donor component 均不重复；自动 QC
通过后只能进入 `hold_for_manual_review`，不能自行产生 `pass`。人工 24/24 接受后由独立 finalizer 生成通过报告；
runner、Gate 2 contract、事务 core、Diffusion adapter、每个 NPZ 和 montage 均有 SHA256，任一证据漂移都会阻断批准。

### Gate 3：冻结基座的配对公平消融

#### S2 基座冻结结果（2026-07-24）

无合成增强的首轮 S2 消融已经在 Dataset264 固定 103 例上完成官方兼容评估。后续 Route
A-F 的 `base_candidate` 统一冻结为 **E / Focal CE**：

- trainer：`nnUNetTrainerBraTS2026RCFocalCompletionFineTune`
- Focal：`gamma=2.0`，RC class weight `3.0`
- checkpoint：见 `../results/s2_small_lesion_ablation_20260721/checkpoint_selection.json`
- 选择依据：WT all-instance F1 `0.712277`、WT small-instance F1 `0.333083`、RC all-instance F1 `0.421053`
- 保留对照：B / `nnUNetTrainerBraTS2026RCCompletionFineTune`，用于总体分割风险比较
- 评估范围：固定内部 103 例，不是官方 179 例提交结果

A-1 和 A-1+E checkpoint 继续保留用于审计，但不得作为 Route A-F 的默认基座。D 只登记为
E 基座上的 Deep Supervision 二阶段设计，尚未启动训练。

Gate 3 不是“原始 E 与继续训练后的 Route A”直接归因。原始 E 没有第二阶段的额外 200 epoch，训练预算不同；
它只能回答部署上是否值得替换，不能单独证明增强有效。必须从同一个冻结 E checkpoint 同时派生：

```text
原始 E：既有 200 epoch checkpoint，部署收益锚点，不参与增强因果归因
E-continue：E warm-start + 200 epoch + p_select=0
E+MET-AUG-A：同一 E warm-start + 同一 200 epoch + p_select=0.20
```

增强的因果效应只比较 `E-continue vs E+MET-AUG-A`。两侧固定 Dataset264 `1035/103` split、fold 0、
Focal trainer、200 epoch、LR `0.001`、每 25 epoch checkpoint、batch、patch、标准 nnU-Net 增强、训练 seed
`20260724`、每 epoch 使用 `seed+epoch` 重置 RNG、单主进程增强、0 个丢弃 train warmup batch、
`cudnn.deterministic=True`、`cudnn.benchmark=False`、`nnUNet_compile=0`、checkpoint 选择规则和 103 例
预测/评估管线。两者必须使用独立结果根，禁止从其中一方 checkpoint 续到另一方。

Gate 3 前需预注册：小/微小病灶主指标、RC/大病灶/假阳性保护指标、配对 bootstrap 方式和“实质退化”
界值，禁止看完结果再改阈值。正式结果同时报告 B、原始 E、E-continue 和 E+MET-AUG-A；其中 B/原始 E
是整体与部署锚点，只有后两者用于判断 Route A 的增强效应。

通过条件是：小/微小病灶主指标有正向、配对置信区间支持的改善；整体官方兼容主指标不实质退化；RC、大病灶和
假阳性未越过预注册保护界值。所有 103 例均保留 paired per-case 差值，重点人工复核改善最大、退化最大、RC、
tiny 和 large 病例。总体均值上升不能抵消系统性漏检或高风险亚组退化。

---

## 4. MET-AUG-A：真实组件平移 + 四模态生成

**唯一目的**：证明“真实、未编辑的组件标签 + 四模态 inpainting”本身有净收益。它是后续一切变量的对照。

| 参数 | 冻结值 | 理由 |
|---|---|---|
| `p_select` | `0.20` | 首轮限制合成样本占比，发现伪影时不会主导梯度。 |
| `max_tumours` | `1` | 每个成功样本只对应一个可追溯 donor。 |
| `scale` | `1.0` | 不引入几何变化。 |
| 类别 | 原样保留 | 不做 SNFH->ET、ET->NETC、腐蚀或膨胀。 |
| 尺寸倾向 | `27-275 mm3` 分层权重 `1.5`，更大合格核心权重 `1.0` | 温和提高可学习小核心暴露，但不制造超小或压制真实大灶。 |
| 组成关系 | 保留冻结的 `P(classes_present | core_volume_bin)` | 不把大小和组成解耦。 |
| 第二灶 | `0` | 不与多灶变量混淆。 |

**失败解释**：若 A 不能通过 Gate 2 或 Gate 3，优先怀疑扩散保真、写入事务或四模态一致性，不应通过更激进的
缩放、类别构造或频率来掩盖基础问题。

**公平对照**：`E-continue (p=0) vs E+MET-AUG-A (p=0.20)`；原始 E 只做部署锚点。

---

## 5. MET-AUG-B：按核心体积的受控缩放

**唯一新增变量**：几何尺度。类别、位置、频率和单灶上限均继承 A。

旧表中的 `0.4-0.9` 等数字如果被理解为三轴线性 scale，会使体积缩为其三次方，极易把 50-500 mm3
的病灶压回不可学习区。因此 B 必须先抽取**目标核心体积比例** `r_v`，再计算线性尺度
`s = r_v^(1/3)`；最近邻重采样后还要实测而不是相信理论值。

| 原 `core_volume_mm3` | 目标体积比例 `r_v` | 等价线性尺度范围 | 理由 |
|---|---|---|---|
| `27-49` | `[1.000, 1.728]` | `[1.00, 1.20]` | 小核心只允许原样或放大，绝不再缩小。 |
| `50-500` | `[max(27/V, 0.50), 1.25]` | `r_v^(1/3)` | 允许温和双向变化，同时用 27 mm3 地板约束。 |
| `>500` | `[max(27/V, 0.15), 0.60]` | `r_v^(1/3)` | 仅对较大核心收缩，减轻 crop 压力但不做极端压缩。 |

缩放后必须同时满足：原有每个非零类别仍非空、`core_volume_mm3 >=27`、最终 bbox `<=56 mm`、标签仍合法。
若一次抽样无效，只能在同一分层内重新抽 donor/scale；达到预注册重试上限仍失败则记 `B_SCALE_NO_OP`，不能把
无效 scale “钳回原体积”后伪装为一次缩放成功。Gate 1 还要报告 B 相对 A 的真实提交率；若缩放造成显著提交率
落差，应先修正采样范围，不能把频率差当作尺度收益。

**公平对照**：`Best(MET-AUG-A) vs Best(MET-AUG-A) + MET-AUG-B`。

---

## 6. MET-AUG-C：合成频率剂量

**唯一新增变量**：`p_select`。使用 B 的同一 manifest、`Q_route`、尺度策略和位置策略；不在此轮加入病灶数
自适应或第二灶。

| 组 | `p_select` | 目的 |
|---|---:|---|
| C-20 | `0.20` | B 的剂量对照。 |
| C-40 | `0.40` | 检验中等合成暴露是否仍有增益。 |
| C-60 | `0.60` | 检验是否出现伪影累积或训练饱和。 |

报告必须同时给出 `selected_rate` 与最终 `committed_rate`。若模型缺失、位置不足或标签无效让三组的实际提交率排序
不再随配置增加，先修正实现/采样器，不把它解释成非线性的生物学效应。若 `0.40` 与 `0.60` 在预注册标准下无可
区分净收益，选择较低剂量 `0.40`；若高剂量使小灶、RC、大灶或 FP 退化，保留通过 Gate 的最低有效剂量。

**公平对照**：C-20/C-40/C-60 内部配对比较，胜者再对 `Best(MET-AUG-B)` 复核。

---

## 7. MET-AUG-D：第二个独立病灶

**唯一新增变量**：在第一灶成功提交后，以固定 `p_second = 0.20` 尝试第二灶，`max_tumours = 2`。

固定概率而非按患者原生病灶数调制，是为了让 D 只回答“二灶相对一灶是否有增量”；按患者负荷重分配频率属于
F，不能提前混入 D。第二灶必须独立执行完整 donor、缩放、位置和四模态事务，并在**所有**缩放/类别变换完成后
再次满足：距原标签和第一灶最终 support 均至少 `5 mm`，距脑边界至少 `3 mm`，bbox 仍 `<=56 mm`。

- 第二灶失败只产生 `SECOND_NO_OP`，不回滚已经完整提交的第一灶，也不留下第二灶的任何部分结果。
- 不进入三灶或四灶。`D+` 应另立实验，因为它改变剂量、病灶数和扩散调用次数，不能借用 D 的结论。
- Gate/结果中必须报告最终 `0/1/2` 灶比例，而不是只报告配置上的 `p_second`。

**公平对照**：`Best(MET-AUG-C) vs Best(MET-AUG-C) + MET-AUG-D`。

---

## 8. MET-AUG-E：受保护的类别拓扑构造

这是最高风险支路。它从 `BestLowerRisk` 独立出发，且不得把“加 SNFH”和“造 NETC”混成一个不可归因的结论。
先分别验证下列子分支；只有二者各自通过，才允许额外检验二者同时发生的交互。

### 8.1 物理形态学和通用回退

形态学核必须是物理球/椭球，不能使用 `np.ones((3,3,3))` 立方体冒充 “1-3 mm 球”。在本路线的 1 mm
空间中先固定半径为 `1 mm`，不同时扫 1-3 mm；半径本身若要优化，必须另立消融。

```python
def physical_ball(radius_mm, spacing_mm):
    half_width = np.ceil(radius_mm / np.asarray(spacing_mm)).astype(int)
    zz, yy, xx = np.mgrid[
        -half_width[0]:half_width[0] + 1,
        -half_width[1]:half_width[1] + 1,
        -half_width[2]:half_width[2] + 1,
    ]
    distance_mm = np.sqrt(
        (zz * spacing_mm[0]) ** 2 +
        (yy * spacing_mm[1]) ** 2 +
        (xx * spacing_mm[2]) ** 2
    )
    return distance_mm <= radius_mm
```

任一构造后都要重测类别、核心体积、bbox、最终 support、脑内 clearance 和与其他病灶的 clearance。若失败，
只允许退回到**该 donor 的真实未构造标签**并记录 `E_TOPOLOGY_FALLBACK`，绝不输出畸形标签；若 fallback 比例高，
E 不具备可解释性，停止进入 Gate 3。

### 8.2 E-SNFH：真实核心外的局灶 SNFH 壳

**唯一新增变量**：仅对纯 ET donor，在最终标签外生成 `1 mm` 的 SNFH 邻接壳。

```python
support_before = label != 0
snfh_shell = binary_dilation(support_before, structure=ball_1mm) & ~support_before
candidate[snfh_shell] = 2
```

- 只允许 `pure ET -> SNFH+ET`；不得删除或重标任何真实 ET/NETC/SNFH。
- 真实含 NETC 的组件、`SNFH+ET` 组件和 `NETC+ET` 组件原样保留，不能为提高稀有类而破坏已有复杂结构。
- 构造事件占最终提交合成病灶的比例先上限为 `10%`；精确分层权重在新的 train-only manifest 上冻结为 `Q_E_SNFH`，不从旧 CSV 搬用 35% 等比例。

理由是只测试“局灶外周水肿是否有帮助”，不把内部坏死、尺度或频率一并改变。

### 8.3 E-NETC：ET 内部的 NETC 核

**唯一新增变量**：仅从真实 ET 内部构造 NETC，不加 SNFH 壳。

```python
et_before = label == 3
netc_inner = binary_erosion(et_before, structure=ball_1mm)
et_shell = et_before & ~netc_inner
candidate[netc_inner] = 1
```

这里的方向必须固定：`netc_inner`（腐蚀后**剩下的内部区域**）才是 NETC；
`et_before & ~netc_inner` 是保留的 ET 壳，绝不能反过来标为 NETC。

构造仅在以下全部条件成立时有效：

1. `netc_inner` 和 `et_shell` 均非空。
2. `dilate(netc_inner, ball_1mm)` 完全落在变换前的 ET 内，即 NETC 有连续 ET 包裹，不接触正常背景。
3. 构造后每个原有类别仍非空，最终核心体积/bbox/clearance 均通过。
4. 仅允许 `pure ET -> NETC+ET` 或 `SNFH+ET -> NETC+SNFH+ET`；纯 SNFH、RC 和已有 NETC 组件不构造。

构造事件同样上限为最终提交病灶的 `10%`，并在 train-only manifest 上冻结 `Q_E_NETC`。这样测试的是“存在一个
有 ET 包裹的坏死核”本身，而不是把 ET 大面积改成 NETC 或人为扩大水肿。

### 8.4 E-COMB：仅在两个单项均通过后验证交互

只有 E-SNFH 与 E-NETC 都独立通过 Gate 1-3，才可对纯 ET donor 以固定顺序执行：先生成合法 NETC 内核，再基于
最终 support 生成 SNFH 壳。它的对照必须同时包括 `BestLowerRisk`、E-SNFH 和 E-NETC，不能仅与组合前的基座比较。
若任一单项失败，E-COMB 不运行。

### 8.5 E 的额外 smoke 要求

每一个实际构造子分支至少 8 个成功固定-seed 样本，且必须在 montage 中并排显示“原 donor 标签、构造标签、
四模态结果、最终 seg”。人工必须确认 NETC 在原 ET 内部、ET 壳连续、SNFH 只在 support 外、没有 NETC 直接暴露在
正常脑组织中。没有成功样本或只能靠 fallback 获得样本，均不是 E 通过的证据。

---

## 9. MET-AUG-F：保持总剂量的自适应频率分配

F 默认关闭，且从 `BestLowerRisk` 独立比较，不依赖 E。它验证的不是“多做增强是否更好”（那已经是 C），而是
在**全体期望 `p_select` 与 C 的最佳值相同**时，把有限增强预算分配给不同原生病灶负荷的 target 是否更好。

执行 F 前，必须只用新的 train-only manifest 检验 `patient_native_lesion_count` 与小核心比例的关系。旧 CSV 因含
locked test、spacing 混杂，不能证明“高负荷患者应升概率”或相反方向。若该关系不稳定、方向不一致或目标病例元数据
在训练 patch 时不可可靠取得，直接跳过 F。

若证据支持一个方向，先冻结分段权重 `w(N)`，再使用：

```text
p_F(N) = clip(lambda * w(N) * p_C_best, 0, p_cap)
lambda: 令训练 target 分布上的 E[p_F(N)] = p_C_best
```

其中 `lambda` 用固定训练集患者权重求解，`p_cap` 与所有分档、分布和方向在 Gate 1 前写入配置。这样 F 与 C 的
总合成剂量相同，才可把差异解释为**分配方式**而非更多训练样本。不得在看完验证结果后切换“病灶多升概率”与
“病灶多降概率”两个方向。

**公平对照**：`BestLowerRisk vs BestLowerRisk + MET-AUG-F`。

---

## 10. 实现状态与 ECNU 放行顺序

### 10.1 旧 G1 增强器不能直接上线

现有 `label_modifier.py` 和 `on_the_fly_augmentation.py` 只可作为历史参考，尚不符合本路线：它们按整例标签采样，
仍实现 SNFH->ET->NETC 级联和激进缩小；通道遍历是旧顺序；允许缺失模态跳过；并在每个模态生成后直接写回。
因此不能仅改几个参数后接入 S2。

### 10.2 已完成的 Route A 桥接代码

逻辑接口已经按单 patch 事务实现：

```text
MetAugEngine.apply(
    image=data_4ch,
    segmentation=seg,
    valid_mask=valid_mask_patch,
    context=(epoch, rank, worker, case_id, patch_index, patch_origin, full_shape),
) -> data_4ch_out, seg_out, transaction_result
```

`case_id`、patch origin/full shape 和 valid mask 由 dataloader sidecar 显式提供；`patient_group` 与 target 是否属于
冻结 train 集由 immutable component manifest 查得，不接受 patch 调用方传入一个可漂移的 split/group 值。

| 层 | 已实现文件 | 当前能力 |
|---|---|---|
| 事务 core | `custom_nnunet/met_aug_core.py` | 组件/route 合同、patient-group 隔离、确定性选择与位置、四模态原子提交、bit-identical 回滚和 JSONL 审计。 |
| nnU-Net 2.8 接口 | `met_aug_data_loader.py`、`met_aug_transform.py` | 显式携带 case ID、patch origin、full shape 和 valid-mask patch；在空间/强度增强前处理一个训练 patch。 |
| G1 后端 | `met_aug_diffusion.py` | 只接受已批准的 z-score、64^3、EDM-Heun 18-step、四模态 150k selection；缺任一模态即失败。 |
| Route gate | `met_aug_gate.py`、`met_aug_gate2.py` | 绑定 parent gate、manifest/config/valid mask、Gate 1/2、人工决定、源码与证据 SHA。 |
| 配对训练合同 | `met_aug_paired_training.py`、Route A/control 两个 trainer | 固定 E/Focal gamma=2、RC 权重 3、200 epoch、seed+epoch、单线程、0 train warmup、确定性 cuDNN、禁用 compile；两组共享标准 nnU-Net 随机流。 |
| ECNU 工具 | `scripts/12_*.py` 至 `scripts/20_*.py` | 组件池、Route A config、valid mask、Gate 1/2、人工 finalizer、route approval，以及真实 nnU-Net+四 Diffusion 训练 step 的吞吐/显存 smoke。 |
| 入口隔离 | `train.sh`、`infer.sh`、`inference_frozen.py` | `met_aug_route_a` 锁定 Dataset264/E checkpoint；验证与推理使用纯 segmentation 路径，不导入 G1 或读取 donor。 |

本地 `g1_t2w_bbdm` 环境的 nnU-Net 版本已核验为 `2.8.0`；42 个定向测试覆盖原子事务、显式 support、
临时 G1 RNG 隔离、0 train warmup、配对 seed、预处理轴序、四返回值接口、Gate/approval、训练 smoke 报告逻辑和
推理隔离。它仍没有 ECNU 的 Dataset264/b2nd、真实四模态 checkpoint 与 CUDA 运行条件，因此不能替代服务器上的
真实 dataloader/GPU smoke。

### 10.3 仍未完成的真实产物

- 尚未从固定 1,035 train 生成 `component_manifest` 和 preprocessed valid-mask manifest。
- 尚未执行 100,000 次 Gate 1，也未预注册并生成 24 例真实四模态 Gate 2。
- 尚未完成人工 24/24 复核和 Route A approval。
- 尚未在 H20 UHost 运行 `scripts/20_run_met_aug_training_smoke.py`，因此真实联合峰值显存、提交事件吞吐和 200 epoch
  墙钟估计仍未知；Gate 2 单独加载 G1 不能替代该 Gate。
- 尚未启动 E-continue 与 E+MET-AUG-A 两个独立 200 epoch Gate 3 训练及固定 103 例正式评估。
- B-F 只有分期科学规范，未实现、未授权；A 未通过前不得提前开发或运行。

### 10.4 Route A 逻辑执行顺序

以下命令保留 Route A 的逻辑依赖说明。当前正式执行位置已迁移到双 H20 UHost，实际运行必须使用 10.6 节的
`05_met_aug_gate_h20_uhost.slurm` 封装和独立 `ROUTE_ROOT`，不得直接照抄旧 ECNU 路径或复用旧输出目录：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${S2_CONDA_ENV:?set the accepted ECNU nnU-Net 2.8 environment}"
export PROJECT_ROOT="${PROJECT_ROOT:-/public/home/${USER}/projects/ECNU_EYU_data}"
export S2_REPOSITORY="${PROJECT_ROOT}/work_space/S2/BraTS2026_S2_RC_v1.0/repository"
cd "${S2_REPOSITORY}"

export ROUTE_ROOT="${ROUTE_ROOT:?set a new, non-existing Route A result root}"
export nnUNet_raw="${nnUNet_raw:?bind the accepted Dataset264 raw root}"
export nnUNet_preprocessed="${nnUNet_preprocessed:?bind the accepted Dataset264 preprocessed root}"
export DATASET_DIR="${nnUNet_raw}/Dataset264_BraTS2026_MET_Completion"
export PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset264_BraTS2026_MET_Completion/nnUNetPlans_3d_fullres"
export TRAIN_FILE="${PWD}/data/splits/completion_warmstart/train_fixed.txt"
export MAPPING_CSV="${PROJECT_ROOT}/work_space/G2/results/manifests/nnunet_case_mapping_master.csv"
export G1_CODE_DIR="${PROJECT_ROOT}/work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN"
export G1_CHECKPOINT_ROOT="${PROJECT_ROOT}/work_space/G1/results/g1_diffusion_v3_final_20260720"
export G1_SELECTION="${PROJECT_ROOT}/work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/checkpoint_selection.json"
export G2_PARENT_GATE="${PROJECT_ROOT}/work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/g2_diffusion_qc_gate.json"
export E_CHECKPOINT="${PROJECT_ROOT}/work_space/S2/data/ecnu_completion_emergency/nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
```

严格按以下顺序执行；`12/13/14/15/17` 不调用 Diffusion：

```bash
python scripts/12_build_met_aug_component_pool.py \
  --dataset-dir "${DATASET_DIR}" --preprocessed-dir "${PREPROCESSED_DIR}" \
  --train-file "${TRAIN_FILE}" --mapping-csv "${MAPPING_CSV}" \
  --output-dir "${ROUTE_ROOT}/component_pool"

python scripts/13_make_met_aug_route_a_config.py \
  --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
  --output "${ROUTE_ROOT}/route_a_config.json"

python scripts/14_prepare_met_aug_valid_masks.py \
  --dataset-dir "${DATASET_DIR}" --preprocessed-dir "${PREPROCESSED_DIR}" \
  --train-file "${TRAIN_FILE}" --output-dir "${ROUTE_ROOT}/valid_masks"

python scripts/15_run_met_aug_gate1.py \
  --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
  --route-config "${ROUTE_ROOT}/route_a_config.json" \
  --valid-mask-manifest "${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json" \
  --output-dir "${ROUTE_ROOT}/gate1" --events 100000 --workers 16

python scripts/17_prepare_met_aug_gate2_smoke.py \
  --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
  --route-config "${ROUTE_ROOT}/route_a_config.json" \
  --valid-mask-manifest "${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json" \
  --output "${ROUTE_ROOT}/gate2_smoke_manifest.json"
```

`18_run_met_aug_gate2_smoke.py` 是本链路第一次真实加载 G1 Diffusion/GPU 的命令，必须在上述产物验收和用户明确
授权后执行：

```bash
python scripts/18_run_met_aug_gate2_smoke.py \
  --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
  --route-config "${ROUTE_ROOT}/route_a_config.json" \
  --valid-mask-manifest "${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json" \
  --smoke-manifest "${ROUTE_ROOT}/gate2_smoke_manifest.json" \
  --preprocessed-dir "${PREPROCESSED_DIR}" --g1-code-dir "${G1_CODE_DIR}" \
  --g1-checkpoint-root "${G1_CHECKPOINT_ROOT}" --g1-checkpoint-selection "${G1_SELECTION}" \
  --g2-parent-gate "${G2_PARENT_GATE}" --output-dir "${ROUTE_ROOT}/gate2_run" --device cuda
```

自动报告只能停在 `hold_for_manual_review`。runner 会生成只读证据模板
`manual_review_template.csv`；先确认决策文件不存在，再从模板创建独立的
`manual_review_decisions.csv`，逐例填写 `review_decision/reviewer/reviewed_at_utc/notes`，不得修改证据列或覆盖模板：

```bash
test ! -e "${ROUTE_ROOT}/gate2_run/manual_review_decisions.csv"
cp "${ROUTE_ROOT}/gate2_run/manual_review_template.csv" \
  "${ROUTE_ROOT}/gate2_run/manual_review_decisions.csv"
```

完成 24/24 人工复核后依次执行 `19 -> 16`；在 18 和 19 之间不得修改 Gate 2 runtime 源码：

```bash
python scripts/19_finalize_met_aug_gate2_review.py \
  --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
  --route-config "${ROUTE_ROOT}/route_a_config.json" \
  --valid-mask-manifest "${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json" \
  --smoke-manifest "${ROUTE_ROOT}/gate2_smoke_manifest.json" \
  --automatic-report "${ROUTE_ROOT}/gate2_run/gate2_automatic_report.json" \
  --review-decisions "${ROUTE_ROOT}/gate2_run/manual_review_decisions.csv" \
  --output "${ROUTE_ROOT}/gate2_final_report.json"

python scripts/16_finalize_met_aug_route_a_gate.py \
  --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
  --route-config "${ROUTE_ROOT}/route_a_config.json" \
  --valid-mask-manifest "${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json" \
  --gate1-report "${ROUTE_ROOT}/gate1/gate1_report.json" \
  --gate2-report "${ROUTE_ROOT}/gate2_final_report.json" \
  --g1-checkpoint-selection "${G1_SELECTION}" --g2-parent-gate "${G2_PARENT_GATE}" \
  --g1-code-dir "${G1_CODE_DIR}" \
  --output "${ROUTE_ROOT}/route_a_approval.json"
```

`route_a_approval.json` 的 schema 必须为 `3` 且 `decision=approve`。随后先运行真实训练路径 smoke；它使用独立
`nnUNet_results`，执行实际 `on_train_start -> dataloader -> train_step`，至少观察到 1 次四模态 `COMMITTED`，
不会运行 validation 或保存 checkpoint：

```bash
python scripts/20_run_met_aug_training_smoke.py \
  --nnunet-raw "${nnUNet_raw}" --nnunet-preprocessed "${nnUNet_preprocessed}" \
  --split-dir "${S2_REPOSITORY}/data/splits/completion_warmstart" \
  --pretrained-weights "${E_CHECKPOINT}" \
  --component-manifest "${ROUTE_ROOT}/component_pool/component_manifest.json" \
  --route-config "${ROUTE_ROOT}/route_a_config.json" \
  --valid-mask-manifest "${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json" \
  --route-approval "${ROUTE_ROOT}/route_a_approval.json" \
  --g1-code-dir "${G1_CODE_DIR}" --g1-checkpoint-root "${G1_CHECKPOINT_ROOT}" \
  --g1-checkpoint-selection "${G1_SELECTION}" --g2-parent-gate "${G2_PARENT_GATE}" \
  --output-dir "${ROUTE_ROOT}/training_smoke" \
  --min-steps 4 --max-steps 12 --min-committed-events 1 --min-gpu-memory-gib 30
```

只有 smoke report 为 `status=pass`、loss finite、无 checkpoint、GPU 至少 30 GiB，且峰值/耗时可接受后，
才可提交两条配对训练。公共合同先固定一次：

```bash
export S2_MET_AUG_COMPONENT_MANIFEST="${ROUTE_ROOT}/component_pool/component_manifest.json"
export S2_MET_AUG_ROUTE_CONFIG="${ROUTE_ROOT}/route_a_config.json"
export S2_MET_AUG_VALID_MASK_MANIFEST="${ROUTE_ROOT}/valid_masks/valid_mask_manifest.json"
export S2_MET_AUG_ROUTE_GATE="${ROUTE_ROOT}/route_a_approval.json"
export S2_MET_AUG_G1_CODE_DIR="${G1_CODE_DIR}"
export S2_MET_AUG_G1_CHECKPOINT_ROOT="${G1_CHECKPOINT_ROOT}"
export S2_MET_AUG_G1_CHECKPOINT_SELECTION="${G1_SELECTION}"
export S2_MET_AUG_G2_QC_GATE="${G2_PARENT_GATE}"
export S2_PRETRAINED_WEIGHTS="${E_CHECKPOINT}"
export S2_COMPLETION_EPOCHS=200
export S2_COMPLETION_INITIAL_LR=0.001
export S2_COMPLETION_SAVE_EVERY=25
export S2_FOCAL_GAMMA=2.0
export S2_PAIRED_TRAINING_SEED=20260724
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export S2_CONTINUE=auto
export S2_SKIP_COMPLETED=1
export nnUNet_compile=0

export CONTROL_RESULT_ROOT="${ROUTE_ROOT}/gate3_e_continue"
export nnUNet_results="${CONTROL_RESULT_ROOT}/nnUNet_results"
export S2_EXPERIMENT_MODE=met_aug_route_a_control
export S2_MET_AUG_ENABLE=0
bash train.sh

export ROUTE_A_RESULT_ROOT="${ROUTE_ROOT}/gate3_met_aug_a"
export nnUNet_results="${ROUTE_A_RESULT_ROOT}/nnUNet_results"
export S2_EXPERIMENT_MODE=met_aug_route_a
export S2_MET_AUG_ENABLE=1
bash train.sh
```

对照目录首次生成 `met_aug_control_provenance.json`、Route A 目录首次生成 `met_aug_provenance.json` 后均不得覆盖；
任何 manifest/config/gate/G1 路径或哈希、seed、Focal/compile 参数漂移都会拒绝续跑。训练完成后两组都用纯
segmentation inference 路径生成同一固定 103 例预测；validation/inference 不得加载 Diffusion。

### 10.5 必须归档的 Route A 产物

适配器消费 `component_manifest` 而不是旧的整例 `label_pool.csv`，最终至少保留：

- `component_manifest.json`、`components.jsonl`、组件 NPZ 及 SHA256；
- `route_a_config.json` 与冻结的 `Q_route`；
- `gate1_report.json`/events、固定 smoke manifest、Gate 2 automatic/final report 和人工复核表；
- smoke report/events、对照训练的 `met_aug_control_provenance.json`、Route A 的 `met_aug_events.jsonl` 与
  immutable `met_aug_provenance.json`；
- G1 四模态 checkpoint selection/parent gate、runtime code snapshot 和全部输入/输出哈希。

不得覆盖 E、B、既有 S2 消融或 G1/G2 父证据。

### 10.6 双 H20 UHost 执行层

2026-07-25 已为单机双 H20 场景增加一套不依赖集群模块的执行层：

- `slurm/05_met_aug_gate_h20_uhost.slurm`：分阶段执行 `prepare/gate2/finalize/training_smoke`；既可由
  Slurm 提交，也可在无 Slurm 的容器主机上直接用 Bash 运行。
- `slurm/06_train_met_aug_pair_h20_uhost.slurm`：同一脚本分别承载 `control` 与 `route_a`，每个进程只允许
  看到一张 GPU。
- `slurm/07_launch_met_aug_pair_h20_uhost.sh`：无调度器时使用 PID、日志和启动锁并行启动两臂，不制造重复训练。
- `slurm/08_status_met_aug_pair_h20_uhost.sh`：只读汇总 PID、GPU、磁盘和最新训练里程碑。
- `slurm/09_sync_s2_runtime_to_uhost.sh`：只同步生产代码、四个冻结 G1 checkpoint、G2 gate/mapping 和 E checkpoint，
  并在云端逐文件复核 SHA256。
- `slurm/10_relay_dataset264_via_vpn_to_uhost.sh`：本机连接 ECNU VPN 后，通过反向 SSH 隧道和可续传 rsync
  将已验收 Dataset264 raw/preprocessed 缓存送入 UHost；仅在 ECNU 无法直连 UHost 时作为回退，本机不落数据副本。
- `slurm/11_push_dataset264_direct_to_uhost_ecnu.sh`：ECNU 能直连 UHost 时的首选路径，提供 `start/status/validate`，
  使用专用临时密钥、固定 H20 主机公钥、两路可续传 rsync 和完整 Dataset264 计数/metadata/空间 Gate。

UHost 环境必须从镜像的 `py312` 克隆到独立 Conda 前缀，再安装 `requirements-h20.txt`；禁止修改基础环境或使用
`sudo pip`。因 nnU-Net 2.8 排除 torch 2.9.*，项目环境固定为已在本地和 H20 实测的
`torch 2.7.1+cu128 / torchvision 0.22.1 / torchaudio 2.7.1`，不允许 pip 自动选择其他 Torch。两臂绑定不同物理 GPU，但单个 trainer 仍是单卡、非 DDP。1TB 根盘的云端安全线固定为至少 150GiB
可用空间。

加速备用不会静默改变正式合同。默认仍为已验收 `EDM-Heun/18 steps/FP32`。若真实 training smoke 的
`estimated_200_epochs_hours > 45`，脚本写入 `TRAINING_SMOKE_TOO_SLOW.hold` 并禁止启动 Gate 3。DPM-Solver++、
UniPC、减步或 BF16 只有在同一 24 例上完成基线配对、自动指标、人工复核和新的 route approval 后才能启用；
原 G2 的 Heun-18 结果不能为新的 solver 代签质量结论。

---

## 11. 回退与当前状态

| 情形 | 处理 |
|---|---|
| MET-AUG-A 未通过 | Gate/训练失败时保留冻结 E；配对收益不成立时在原始 E 与 E-continue 中保留预注册风险更低者，不推进后续 Route。 |
| B/C/D 未通过 | 回退到上一个通过 Gate 3 的低风险配置，后续串行路线停止。 |
| MET-AUG-E 或 MET-AUG-F 未通过 | 不影响 `BestLowerRisk`；保留证据和失败 reason，不把两个高层策略强行组合。 |
| 任一 Gate 发现泄漏、partial 写入、标签拓扑错或四模态不一致 | 该 Route 无训练资格，先修实现并重跑相同 Gate。 |

**策略版本**：`MET-AUG-A-F-v2`。

**当前状态**：S2 无合成增强基座已冻结为 E。Route A 桥接代码、Gate 工具、配对 control、真实训练 smoke 入口、
训练/推理隔离和本地/远端 73/73 测试已完成；双 H20 UHost 已连通，代码、四模态 G1 checkpoint、G2
gate/mapping 与冻结 E checkpoint 已逐文件 SHA256 验收。UHost 隔离 Conda、Torch 2.7.1/cu128、nnU-Net 2.8、
`pip check`、双 H20 CUDA 实算和 `rsync 3.2.7` 均已验收。ECNU 分流 VPN、ECNU->H20 直连、临时密钥与固定
H20 主机公钥均已验收；Dataset264 raw 32GB 与 preprocessed 68GB 正由两个唯一 rsync 后台直推到 UHost，
两张 GPU 当前空闲，没有 Gate、Diffusion smoke 或训练进程。真实 `component_manifest`、
valid-mask manifest、Gate 1、24 例 Gate 2、人工复核、route approval、UHost nnU-Net 2.8 smoke、E-continue 和
Route A 训练均尚未完成。因此“桥接代码已准备”不等于“样本增强已做完”，目前还没有任何 Route A
checkpoint 或正式结果。旧 G1 结果、旧 CSV 和 E 的现有评估都不能冒充 Route A 结果。validation/test/官方提交
始终不调用 Diffusion。
