# S4 SAM 及相关变体微调方案与输入匹配性调研

生成日期：2026-05-19
重整日期：2026-05-20
项目目录：`ECNU_EYU_data` 当前仓库根目录
调研对象：MedSAM / SAM-Med2D / SAM-Adapter，以及更适合医学 3D 任务的 Medical-SAM-Adapter
目标任务：BraTS 2026 Task 1 Brain Metastases Segmentation
主要读者：S4、S1/S2 分割组、G2 数据组、组内统筹同学

## 0. 一页结论

S4 可以微调 SAM 及相关变体完成任务要求，但不能把 BraTS 2026 MET 的 `.nii.gz` 直接丢给这些模型训练。SAM 系模型的默认范式通常是：

```text
image + prompt -> binary mask
```

而我们比赛数据的范式是：

```text
[t1n, t1c, t2w, t2f] 3D NIfTI -> 0/1/2/3/4 multiclass 3D segmentation
```

因此 S4 的核心工作不是简单 fine-tune，而是补齐三层适配：

1. 数据输入适配：3D 四模态 NIfTI 转为 2D/3D SAM 可用 tensor。
2. 自动 prompt 生成：从 GT、粗分割或候选检测结果生成 bbox/point/mask prompt。
3. 输出还原与合并：把 2D/3D binary mask 还原为 BraTS 3D label，并处理多类别冲突。

最终推荐路线：

```text
先吸收 改进思路(1).ipynb 中合理部分
  -> 明确不能照搬的大改架构
  -> 构建 BraTS -> SAM-Med2D 2D slice 数据集
  -> 自动生成 bbox / center point prompt
  -> fine-tune SAM-Med2D
  -> 2D prediction stitch 回 3D
  -> 内部 validation fold 上报告 Dice / NSD / lesion recall
  -> 再做 MedSAM GT bbox 上限实验和 Medical-SAM-Adapter 3D 改造
```

推荐优先级：

| 优先级 | 模型路线 | 是否建议做 | 核心原因 |
|---:|---|---|---|
| 1 | SAM-Med2D fine-tune | 强烈建议 | 输入契约清楚，最容易跑通 2D 医学切片微调和内部验证报告 |
| 2 | MedSAM GT bbox 上限实验 | 建议 | 医学预训练强，适合快速评估“给定正确 prompt 时”的边界 refine 能力 |
| 3 | Medical-SAM-Adapter 3D 改造 | 建议作为第二阶段 | 最接近 3D 医学数据，但现有 BraTS loader 不适配 2026 MET，需要重写 |
| 4 | SAM-Adapter 原版 | 不优先 | 原版偏 2D RGB 自然图像/通用场景，医学 3D 适配收益不如前几条 |

第一版可交付目标：

```text
SAM-Med2D fine-tune
  + BraTS 2D slice 数据转换
  + 自动 bbox/center point prompt
  + 内部 validation 性能报告
```

这里的 “validation 性能报告” 指训练集内部划分出的 validation fold，不是官方 Validation。官方 Validation 没有公开标签，不能计算 Dice/NSD。

## 1. 从 `改进思路(1).ipynb` 吸收的合理部分

`改进思路(1).ipynb` 的主体是一套 “SAM-Med3D for BraTS Glioma Segmentation” 概念方案。它有启发价值，但不能原样照搬。原因是：

- 它更像重新造一个 3D SAM 分割网络，而不是微调现有 SAM 模型；
- 它偏 glioma 的 WT/TC/ET 逻辑；
- 我们是 MET 任务，更关注小病灶、RC、lesion-wise detection 和官方 `0/1/2/3/4` 标签；
- 它的部分代码设计会让工程量和显存压力迅速膨胀，不利于 S4 快速交付可验证结果。

所以本方案先吸收合理部分，再把这些思想落到可执行的 SAM-Med2D / MedSAM / Medical-SAM-Adapter 路线里。

### 1.1 多模态信息融合

notebook 里提出利用 `t1n/t1c/t2w/t2f` 四模态信息，这是合理的。对 MET 来说：

- `t1c` 对 ET 和增强病灶最关键；
- `t2f`/FLAIR 对 SNFH 和水肿样高信号更关键；
- `t2w` 对结构和部分异常信号有辅助价值；
- `t1n` 提供解剖背景和非增强结构参考。

但第一版不要实现复杂的 3D cross-modal attention。建议分阶段落地：

```text
阶段 A: 2D pseudo-RGB
  R = normalized t1c
  G = normalized t2f
  B = normalized t2w

阶段 B: 单模态/多模态消融
  t1c only
  t2f only
  t1c/t2f/t2w pseudo-RGB

阶段 C: Medical-SAM-Adapter 3D
  先 t1c 单模态 [1,H,W,D]
  再尝试四模态 [4,H,W,D]
```

不建议第一版做 notebook 中那种全体素级 `D*H*W x D*H*W` cross-modal attention，因为显存压力很大，且容易把 S4 任务从“微调 SAM”扩展成一个高风险新模型项目。

### 1.2 自动 prompt 生成

notebook 提到 Auto-Prompt Generator，这个方向非常重要。SAM 系模型若要用于比赛，必须解决 prompt 自动化，否则只能做交互式演示。

本方案采用更可控的两阶段 prompt 策略：

```text
训练/内部验证:
  GT seg -> connected components -> bbox / center point

真实无标签 Validation/Test:
  S1/S2 粗分割模型 -> candidate mask -> bbox / center point -> SAM refinement
```

第一版不建议直接训练 CAM-based Auto-Prompt 模块。理由是：

- CAM prompt 本身需要额外分类/定位监督，调参成本高；
- 小病灶 CAM 容易不稳定；
- 用 GT connected components 生成 bbox/center point 更可控，能先验证 SAM 分割上限。

后续如果 S4 第一版效果明确，再把 CAM/FPN prompt generator 作为第二阶段研究方向。

### 1.3 多类别输出包装

notebook 里提到多类别 mask decoder 和层次化输出，这个方向可以吸收，但不能直接套 glioma 的 WT/TC/ET 规则。

BraTS 2026 MET 应保持官方标签：

```text
0 background
1 NETC
2 SNFH
3 ET
4 RC
```

不要使用 notebook 中的错误 remap：

```text
1 -> 2
2 -> 3
3 -> 1
```

S4 第一版推荐从 binary/region 任务开始：

```text
whole abnormal = 1|2|3|4
lesion core = 1|3|4
ET only = 3
RC only = 4
```

如果 binary/region 结果有收益，再做 class-wise binary：

```text
NETC vs background
SNFH vs background
ET   vs background
RC   vs background
```

多类别合并时先采用简单优先级：

```text
RC > ET > NETC > SNFH > background
```

这只是工程初版规则，不是最终医学规则。最终要根据内部 validation 的 confusion matrix、false positive components 和 lesion-wise recall 调整。

### 1.4 Dice/Focal/Boundary loss 与小目标关注

notebook 强调 Dice、Generalized Dice、Focal、Boundary loss，这个方向合理。MET 小病灶和类别不平衡明显，单纯 CE 往往不够。

推荐第一版训练损失：

```text
loss = DiceLoss + 0.5 * CrossEntropyLoss
```

如果小病灶召回不足，再加入：

```text
loss = DiceLoss + 0.5 * CrossEntropyLoss + 0.3 * FocalLoss
```

Boundary loss 可作为后续增强，不建议第一版加入。理由是边界距离图计算和多类别边界定义会增加实现复杂度，且小病灶上距离图噪声可能比较明显。

实现上优先使用 MONAI 或已有成熟实现，不建议从零手写所有 loss。S4 的重点应放在数据适配、prompt 生成、微调和验证报告，而不是重写基础损失函数库。

### 1.5 必须避免的部分

| notebook 设计 | 问题 | 本方案处理 |
|---|---|---|
| 按 glioma 写 WT/TC/ET hierarchy | MET 有 RC、小病灶和 lesion-wise detection，不能完全套 glioma | 只保留 region 思想，标签按 MET 官方定义 |
| 错误 label remap：`1->2, 2->3, 3->1` | 会导致 ET/NETC/SNFH 类别错位 | 禁止 remap，保持 `1=NETC, 2=SNFH, 3=ET, 4=RC` |
| 固定 resize 到 `128x128x128` | 可能压没小病灶，破坏物理尺度 | 2D SAM 走 slice resize；3D 方案用 patch/spacing-aware 处理 |
| 只扫一层病例目录 | 会漏掉 `UCSD - Training` 的 646 个病例 | 使用 manifest 或递归扫描 |
| 从零实现 3D ViT + decoder | 工程风险大，失去 SAM 预训练优势 | 优先微调现有模型 |
| 全体素级 cross attention | 显存复杂度过高 | 第一版用 pseudo-RGB、简单融合或轻量 adapter |
| 官方 Validation 上算 Dice | 官方 Validation 无公开标签 | 只在内部 validation fold 报告指标 |

## 2. 吸收后的完整微调执行路线

本节是 S4 最重要的执行路径。第一阶段只追求把闭环跑通，不追求一次性覆盖所有 SAM 变体。

### 2.1 总体闭环

```text
真实训练 manifest
  -> corrected labels overlay
  -> 固定内部 train/val split
  -> NIfTI 转 2D SAM slices
  -> 从 GT 生成 bbox / center point prompt
  -> SAM-Med2D fine-tune
  -> val slices inference
  -> 2D mask stitch 回 3D NIfTI
  -> 内部 validation 指标
  -> 报告与错误分析
  -> 可选：MedSAM 上限实验 / Medical-SAM-Adapter 3D 改造
```

### 2.2 M0：准备 manifest 与 split

目标：建立 S4 可复用的数据索引，不让模型脚本直接扫原始目录。

输入：

```text
train_root
corrected_label_dir
real_train_manifest.csv
fixed_train_val_split.json
```

输出：

```text
S4_outputs/manifests/
  s4_case_manifest.csv
  train_val_split.json
```

要求：

- 覆盖顶层 650 个训练病例和 `UCSD - Training` 的 646 个病例。
- 应用 corrected labels。
- 排除或标记非法 label 病例。
- validation fold 只能包含真实病例。
- synthetic 数据不能泄漏到 validation。
- 每个 case 记录 `case_id/t1n/t1c/t2w/t2f/effective_seg_path/shape/spacing/affine_hash/split`。

### 2.3 M1：构建 2D SAM 数据集

目标：把 3D NIfTI 转成 SAM-Med2D/MedSAM 可用的 2D 输入。

推荐模态策略：

```text
R = normalized t1c
G = normalized t2f
B = normalized t2w
```

可做消融：

```text
t1c only
t2f only
t1c/t2f/t2w pseudo-RGB
```

mask 版本：

```text
whole_abnormal = seg in {1,2,3,4}
lesion_core = seg in {1,3,4}
et = seg == 3
rc = seg == 4
class_wise = seg == 1/2/3/4
```

输出：

```text
S4_outputs/sam_med2d_dataset/
  images/
  masks/
  image2label_train.json
  label2image_val.json
S4_outputs/manifests/
  slice_manifest.csv
```

`slice_manifest.csv` 必须记录：

```text
case_id
split
z_index
image_path
mask_path
mask_task
label_id
label_name
component_id
bbox_xyxy_original
bbox_xyxy_resized
center_xy_original
center_xy_resized
original_shape
output_shape
spacing
affine_hash
source_t1n_path
source_t1c_path
source_t2w_path
source_t2f_path
source_seg_path
```

### 2.4 M2：自动 prompt 生成

目标：把每个 binary mask 或 coarse prediction 转成 SAM prompt。

第一版 prompt：

```text
bbox prompt:
  每个 2D connected component 一个 bbox
  小病灶 margin = 3-5 px
  大病灶 margin = 5-10 px

center point prompt:
  每个 component 一个中心点
  优先选 component 内部点，不要落在背景
```

训练/验证阶段 prompt 来源：

```text
GT mask -> connected components -> prompt
```

未来真实 Validation/Test 阶段 prompt 来源：

```text
coarse segmentation from S1/S2 -> connected components -> prompt
```

输出：

```text
S4_outputs/manifests/prompt_manifest.csv
```

### 2.5 M3：SAM-Med2D fine-tune

目标：完成 S4 第一版可报告模型。

训练顺序：

1. `whole_abnormal` binary。
2. `lesion_core` binary。
3. `et` binary。
4. 如果前三个实验有效，再做 `NETC/SNFH/ET/RC` class-wise binary。

训练策略：

- 含病灶 slice 为主，保留少量背景 slice。
- 小病灶 slice 过采样。
- bbox prompt 为主，center point prompt 为补充。
- 第一版 loss 使用 `Dice + 0.5 * CE`。
- 小病灶召回不足时加入 Focal loss。
- 保留训练 config、checkpoint、日志和随机种子。

输出：

```text
S4_outputs/checkpoints/
  sam_med2d_whole_abnormal/
  sam_med2d_lesion_core/
  sam_med2d_et/
S4_outputs/logs/
  sam_med2d_train_*.log
```

### 2.6 M4：SAM-Med2D 推理与 3D 还原

目标：把 validation slices 的 2D mask stitch 回每例 3D NIfTI。

流程：

```text
val slice image + prompt
  -> SAM-Med2D checkpoint inference
  -> 2D binary mask
  -> resize to original H,W
  -> place at case_id,z_index
  -> merge classes/regions
  -> save 3D NIfTI
  -> compute metrics
```

输出：

```text
S4_outputs/predictions/
  val_2d_masks/
  val_3d_nifti/
S4_outputs/metrics/
  internal_val_results.csv
  small_lesion_results.csv
```

### 2.7 M5：MedSAM GT bbox 上限实验

目标：回答 “给定正确 bbox prompt 时，MedSAM 是否能 refine MET 病灶边界”。

实验设置：

```text
input = pseudo-RGB slice
prompt = GT bbox
mask target = binary region 或 class-wise binary
output = 2D prediction -> stitch 3D
```

注意：

- 这是上限实验，不能作为真实无标签 Validation 的全自动性能。
- 如果 GT bbox 上限都很差，不建议继续投入大量 SAM refinement 工作。
- 如果 GT bbox 上限明显好，再接入 S1/S2 粗分割 prompt 做真实 refinement 实验。

### 2.8 M6：Medical-SAM-Adapter 3D 改造

目标：测试 3D context 是否能改善 2D slice 的跨层不连续和小病灶稳定性。

第一版：

```text
image = t1c only, shape [1, H, W, D]
label = lesion_core binary 或 ET binary
prompt = lesion center click 或 bbox
```

第二版：

```text
image = [t1n, t1c, t2w, t2f], shape [4, H, W, D]
label = class-wise binary 或 region-wise binary
```

是否改 4 通道主干，取决于 Medical-SAM-Adapter 的 image encoder/adapter 是否稳定支持 `C=4`。如果需要改第一层权重，先用单模态跑通，避免把时间耗在结构调试上。

### 2.9 M7：报告与对比

S4 报告至少包含：

- 使用模型：MedSAM / SAM-Med2D / Medical-SAM-Adapter。
- 是否微调：zero-shot、few-shot fine-tune、full fine-tune。
- 输入模态：单模态或 pseudo-RGB。
- prompt 来源：GT prompt、粗分割 prompt、人工 prompt。
- 输出还原方式：2D stitch 或 3D output。
- 内部 validation fold 指标。
- 小病灶分桶表现。
- 与 S1/S2 nnU-Net baseline 的关系：独立模型、refinement、teacher/辅助裁判。

## 3. 我们的数据输入契约

当前 BraTS 2026 MET 本地训练数据结构：

```text
MICCAI-LH-BraTS2025-MET-Challenge-Training/
  BraTS-MET-xxxxx-xxx/
    BraTS-MET-xxxxx-xxx-t1n.nii.gz
    BraTS-MET-xxxxx-xxx-t1c.nii.gz
    BraTS-MET-xxxxx-xxx-t2w.nii.gz
    BraTS-MET-xxxxx-xxx-t2f.nii.gz
    BraTS-MET-xxxxx-xxx-seg.nii.gz
  UCSD - Training/
    BraTS-MET-xxxxx-xxx/
      BraTS-MET-xxxxx-xxx-t1n.nii.gz
      BraTS-MET-xxxxx-xxx-t1c.nii.gz
      BraTS-MET-xxxxx-xxx-t2w.nii.gz
      BraTS-MET-xxxxx-xxx-t2f.nii.gz
      BraTS-MET-xxxxx-xxx-seg.nii.gz
```

数据事实：

| 项目 | 当前状态 |
|---|---|
| 文件类型 | 3D NIfTI `.nii.gz` |
| 模态 | `t1n/t1c/t2w/t2f` 四模态 MRI |
| 训练标签 | `seg.nii.gz`，单个 3D multiclass mask |
| 标签值域 | `0/1/2/3/4` |
| 带标签训练病例 | 1296 个 |
| 顶层训练病例 | 650 个 |
| `UCSD - Training` 病例 | 646 个 |
| 官方 Validation | 179 个无公开标签病例 |
| shape/spacing | 跨病例不固定，不能写死 `240x240x155` 或 `128x128x128` |

官方标签含义：

| 标签 | 缩写 | 含义 | 对 SAM 适配的影响 |
|---:|---|---|---|
| 0 | background | 背景 | 输出合并时默认类 |
| 1 | NETC | 非强化肿瘤核心 | 可作为 lesion core 的一部分 |
| 2 | SNFH | 周围非强化 FLAIR 高信号 | 主要依赖 `t2f/t2w`，不适合只看 `t1c` |
| 3 | ET | 强化肿瘤 | `t1c` 最关键，可先做 ET 二分类实验 |
| 4 | RC | 切除腔 | 稀有类，建议单独统计和复核 |

S4 的所有脚本应优先读取 G2/S1 生成的 manifest，而不是自己临时 `os.listdir` 一层目录。这样可以避免漏掉 `UCSD - Training`，也能复用 corrected-label overlay、非法 label 标记和固定 train/val split。

## 4. 三类 SAM 模型的输入匹配性

### 4.1 MedSAM

官方仓库：<https://github.com/bowang-lab/MedSAM>
论文：Segment Anything in Medical Images，Nature Communications 2024

#### 官方输入形式

MedSAM 官方 inference 典型命令：

```bash
python MedSAM_Inference.py \
  -i assets/img_demo.png \
  -o ./ \
  --box "[95,255,190,350]"
```

代码层面的输入契约：

- 主 inference 读取 2D image path。
- 灰度图会复制成 3 通道。
- 图像 resize 到 `1024x1024`。
- 输入 tensor 通常是 `1 x 3 x 1024 x 1024`。
- prompt 主要是 bbox，格式为 `[x1, y1, x2, y2]`。
- 输出是当前 prompt 对应的 binary mask。

MedSAM 也提供 CT/MR NIfTI 预处理脚本，但该流程本质上仍是把 NIfTI 切成 2D slice 后保存为 `.npy` 图像和 `.npy` mask。

#### 与我们数据的匹配性

| 项目 | MedSAM | 我们数据 | 结论 |
|---|---|---|---|
| 文件类型 | 2D image 或预处理后 `.npy` | 3D `.nii.gz` | 需转换 |
| 输入维度 | `B x 3 x 1024 x 1024` | 4 个 3D volume | 需切片 |
| 模态数 | 1 灰度复制 3 通道，或 3 通道图像 | 4 MRI 模态 | 需单模态或 pseudo-RGB |
| label | binary mask per prompt | 5 值 multiclass mask | 需拆成 binary/region |
| prompt | bbox | 需要自动生成 | 可由 GT 或粗分割生成 |
| 3D 连续性 | 无内建保证 | MET 是 3D 任务 | 需 stitch 和后处理 |

#### 推荐用法

MedSAM 适合做 “GT bbox 上限实验”：

```text
BraTS 3D NIfTI
  -> axial slice
  -> pseudo-RGB: R=t1c, G=t2f, B=t2w
  -> GT seg 生成 bbox prompt
  -> MedSAM prediction
  -> stitch back to 3D
  -> 计算内部 validation 指标
```

### 4.2 SAM-Med2D

官方仓库：<https://github.com/OpenGVLab/SAM-Med2D>
论文：SAM-Med2D / SA-Med2D-20M

#### 官方输入形式

SAM-Med2D 是明确的 2D 医学图像 SAM fine-tuning 方案。官方数据组织方式是：

```text
data_demo/
  images/
  masks/
  image2label_train.json
  label2image_test.json
```

代码层面的输入契约：

- `DataLoader.py` 用 `cv2.imread` 读取 2D image。
- image 通常是 PNG/JPG。
- mask 用 `cv2.imread(mask_path, 0)` 读取单通道灰度图。
- mask 必须是 binary，值为 `0/1` 或 `0/255`。
- 一个 2D image 可以对应多个 binary mask。
- 支持 bbox prompt、point prompt、mask prompt。

`image2label_train.json` 的语义类似：

```json
{
  "image.png": [
    "mask_object_001.png",
    "mask_object_002.png"
  ]
}
```

#### 与我们数据的匹配性

| 项目 | SAM-Med2D | 我们数据 | 结论 |
|---|---|---|---|
| 文件类型 | 2D PNG/JPG + JSON | 3D NIfTI | 需完整转换 |
| 输入维度 | 2D image | 3D volume | 需切片 |
| 模态数 | 3 通道 image | 4 MRI 模态 | 建议 pseudo-RGB |
| label | 每目标 binary mask | 单个 multiclass 3D mask | 需拆分 |
| prompt | bbox / point / mask | 需自动生成 | 可实现 |
| 输出 | 2D binary mask | 3D multiclass prediction | 需 stitch + merge |

#### 推荐用法

SAM-Med2D 是 S4 第一版最推荐路线。建议构建：

```text
S4_outputs/sam_med2d_brats_slices/
  images/
    BraTS-MET-00001-000_z072.png
  masks/
    BraTS-MET-00001-000_z072_ET_000.png
    BraTS-MET-00001-000_z072_SNFH_000.png
  image2label_train.json
  label2image_val.json
  slice_manifest.csv
```

### 4.3 SAM-Adapter 与 Medical-SAM-Adapter

这里需要区分两个项目：

- SAM-Adapter 原版：<https://github.com/tianrun-chen/SAM-Adapter-PyTorch>
- Medical-SAM-Adapter：<https://github.com/ImprintLab/Medical-SAM-Adapter>

SAM-Adapter 原版主要面向 2D RGB 图像和通用下游场景，不是 S4 第一优先级。更值得看的是 Medical-SAM-Adapter。

Medical-SAM-Adapter 自定义 dataset 输出契约类似：

```python
{
  "image": image_tensor,
  "label": target_mask,
  "p_label": positive_or_negative_prompt_label,
  "pt": prompt_point,
  "image_meta_dict": {"filename_or_obj": name}
}
```

输入尺寸说明：

- 2D image tensor size 为 `[C, H, W]`。
- 3D data tensor size 为 `[C, H, W, D]`。
- 医学 CT/MRI/US 示例通常 `C=1`。
- click prompt 形式沿用 SAM 的点提示思路。
- 3D 模式需要使用相应配置，例如 `-thd True`、`-chunk` 等。

Medical-SAM-Adapter 仓库里有 `dataset/brat.py`，但不能直接用于 BraTS 2026 MET：

- 旧 loader 使用 `t1/flair/t2/t1ce` 命名，不是 `t1n/t1c/t2w/t2f`；
- 示例只返回 `raw_image[0]`，相当于只用第一个模态；
- 示例固定 `label = 4`，只做某个 label 的 binary segmentation，不是 MET 四类分割。

建议新写 `BraTS2026METDataset`：

```python
modalities = ["t1n", "t1c", "t2w", "t2f"]
image = stack([
    load_nii(case_id + "-t1n.nii.gz"),
    load_nii(case_id + "-t1c.nii.gz"),
    load_nii(case_id + "-t2w.nii.gz"),
    load_nii(case_id + "-t2f.nii.gz"),
], axis=0)  # [4, H, W, D]
label = load_nii(case_id + "-seg.nii.gz")
```

### 4.4 匹配性总表

| 维度 | MedSAM | SAM-Med2D | SAM-Adapter 原版 | Medical-SAM-Adapter |
|---|---|---|---|---|
| 原生输入 | 2D image + bbox | 2D image + binary masks + prompt | 2D RGB image + mask | 2D/3D tensor + prompt |
| 是否直接读 `.nii.gz` | 预处理可读，但主流程不是 | 否 | 否 | 可通过 dataloader |
| 是否直接支持 3D | 否，按 2D slice | 否 | 否 | 是，但需改数据集 |
| 是否直接支持 4 模态 MRI | 否 | 否 | 否 | 不确定，示例偏 `C=1` |
| 是否直接支持 `0/1/2/3/4` 多类别 | 否 | 否 | 否 | 不直接，需包装 |
| prompt 需求 | bbox | bbox/point/mask | 依任务配置 | click/box 等 |
| 第一版定位 | 上限实验/refinement | 主微调路线 | 不优先 | 第二阶段 3D 改造 |

## 5. 需要开发的脚本与伪代码

本节给出 S4 第一版需要的全部脚本。实现时建议统一放在：

```text
S4_scripts/
  s4_build_manifest_and_split.py
  s4_build_sam_slice_dataset.py
  s4_generate_prompts.py
  s4_train_sam_med2d.py
  s4_infer_sam_med2d.py
  s4_eval_sam_3d_stitch.py
  s4_run_medsam_gt_bbox.py
  s4_build_msa3d_dataset.py
  s4_train_medical_sam_adapter_3d.py
  s4_report_sam_results.py
```

### 5.1 `s4_build_manifest_and_split.py`

职责：构建 S4 使用的病例 manifest 和内部 train/val split。

输入：

```text
--train-root
--corrected-label-dir
--g2-manifest optional
--output-dir S4_outputs/manifests
--val-ratio 0.2
--seed 2026
```

输出：

```text
s4_case_manifest.csv
train_val_split.json
```

伪代码：

```python
def main(args):
    if args.g2_manifest exists:
        rows = read_csv(args.g2_manifest)
    else:
        rows = []
        for case_dir in recursive_find_dirs(args.train_root, pattern="BraTS-MET-*"):
            case_id = case_dir.name
            paths = {
                "t1n": case_dir / f"{case_id}-t1n.nii.gz",
                "t1c": case_dir / f"{case_id}-t1c.nii.gz",
                "t2w": case_dir / f"{case_id}-t2w.nii.gz",
                "t2f": case_dir / f"{case_id}-t2f.nii.gz",
                "seg": case_dir / f"{case_id}-seg.nii.gz",
            }
            if not all_exists(paths):
                write_reject(case_id, reason="missing_required_files")
                continue
            rows.append(make_row(case_id, case_dir, paths))

    corrected = index_corrected_labels(args.corrected_label_dir)
    for row in rows:
        row["effective_seg_path"] = corrected.get(row.case_id, row.raw_seg_path)
        header = read_nifti_header(row.t1c_path)
        seg_info = inspect_label(row.effective_seg_path)
        row["shape"] = header.shape
        row["spacing"] = header.spacing
        row["affine_hash"] = hash_affine(header.affine)
        row["labels_present"] = seg_info.unique_values
        row["has_illegal_label"] = not set(seg_info.unique_values) <= {0,1,2,3,4}
        row["qc_pass"] = required_files_ok(row) and not row["has_illegal_label"]

    valid_rows = [r for r in rows if r.qc_pass]
    train_ids, val_ids = stratified_case_split(
        valid_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_by=["has_label_4", "lesion_count_bin", "source_site"]
    )

    for row in rows:
        row["split"] = "val" if row.case_id in val_ids else "train"

    write_csv("s4_case_manifest.csv", rows)
    write_json("train_val_split.json", {"train": train_ids, "val": val_ids})
```

实现注意：

- 必须递归扫描，不能只扫 Training 根目录一层。
- corrected labels 要通过 `effective_seg_path` 使用，不建议直接覆盖原数据。
- `BraTS-MET-01094-002` 这类非法 label 病例应默认排除或标记 `qc_pass=false`。

### 5.2 `s4_build_sam_slice_dataset.py`

职责：把 3D BraTS NIfTI 转成 SAM-Med2D 可用的 2D image/mask/json。

输入：

```text
--case-manifest S4_outputs/manifests/s4_case_manifest.csv
--output-dir S4_outputs/sam_med2d_dataset
--task whole_abnormal|lesion_core|et|rc|class_wise
--modality-policy pseudo_rgb_t1c_t2f_t2w
--image-size 256
--keep-empty-ratio 0.05
--min-component-voxels 1
```

输出：

```text
images/*.png
masks/*.png
image2label_train.json
label2image_val.json
slice_manifest.csv
```

伪代码：

```python
def normalize_slice(x):
    brain = x[x != 0]
    if len(brain) == 0:
        return zeros_uint8_like(x)
    lo, hi = percentile(brain, [1, 99])
    x = clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return (x * 255).astype(uint8)

def make_pseudo_rgb(t1c_slice, t2f_slice, t2w_slice):
    r = normalize_slice(t1c_slice)
    g = normalize_slice(t2f_slice)
    b = normalize_slice(t2w_slice)
    return stack([r, g, b], axis=-1)

def make_task_masks(seg_slice, task):
    if task == "whole_abnormal":
        return [{"label_name": "whole_abnormal", "mask": seg_slice > 0}]
    if task == "lesion_core":
        return [{"label_name": "lesion_core", "mask": isin(seg_slice, [1,3,4])}]
    if task == "et":
        return [{"label_name": "ET", "mask": seg_slice == 3}]
    if task == "rc":
        return [{"label_name": "RC", "mask": seg_slice == 4}]
    if task == "class_wise":
        return [
            {"label_name": "NETC", "label_id": 1, "mask": seg_slice == 1},
            {"label_name": "SNFH", "label_id": 2, "mask": seg_slice == 2},
            {"label_name": "ET", "label_id": 3, "mask": seg_slice == 3},
            {"label_name": "RC", "label_id": 4, "mask": seg_slice == 4},
        ]

def main(args):
    rows = read_csv(args.case_manifest)
    image2label_train = defaultdict(list)
    label2image_val = {}
    slice_rows = []

    for case in rows:
        if not case.qc_pass:
            continue
        vols = load_modalities(case.t1n, case.t1c, case.t2w, case.t2f)
        seg = load_nifti(case.effective_seg_path)
        affine, spacing, shape = read_reference_meta(case.t1c_path)

        for z in range(shape.z):
            image_rgb = make_pseudo_rgb(vols.t1c[:,:,z], vols.t2f[:,:,z], vols.t2w[:,:,z])
            masks = make_task_masks(seg[:,:,z], args.task)
            non_empty_masks = [m for m in masks if m.mask.sum() > 0]

            if len(non_empty_masks) == 0:
                if random() > args.keep_empty_ratio:
                    continue
                save_image_only_background_slice(...)
                continue

            image_path = save_png(resize(image_rgb, args.image_size), images_dir, case.case_id, z)

            for m in non_empty_masks:
                components = connected_components_2d(m.mask)
                for comp_id, comp in enumerate(components):
                    mask = component_to_binary_mask(comp)
                    mask_resized = resize_nearest(mask, args.image_size)
                    mask_path = save_png(mask_resized * 255, masks_dir, case.case_id, z, m.label_name, comp_id)

                    if case.split == "train":
                        image2label_train[image_path.name].append(mask_path.name)
                    else:
                        label2image_val[mask_path.name] = image_path.name

                    slice_rows.append({
                        "case_id": case.case_id,
                        "split": case.split,
                        "z_index": z,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "mask_task": args.task,
                        "label_name": m.label_name,
                        "component_id": comp_id,
                        "original_shape": shape,
                        "spacing": spacing,
                        "affine_hash": case.affine_hash,
                        "source_t1c_path": case.t1c_path,
                        "source_seg_path": case.effective_seg_path,
                    })

    write_json("image2label_train.json", image2label_train)
    write_json("label2image_val.json", label2image_val)
    write_csv("slice_manifest.csv", slice_rows)
```

实现注意：

- 不要丢弃小病灶 slice。
- 背景 slice 可以抽样保留，避免训练集全是阳性。
- 保存 `slice_manifest.csv` 是 2D -> 3D 还原的关键。

### 5.3 `s4_generate_prompts.py`

职责：从 mask 或粗分割结果生成 bbox/center point prompt。

输入：

```text
--slice-manifest S4_outputs/manifests/slice_manifest.csv
--prompt-type bbox|point|bbox_point
--bbox-margin-small 3
--bbox-margin-large 8
--output S4_outputs/manifests/prompt_manifest.csv
```

输出：

```text
prompt_manifest.csv
```

伪代码：

```python
def bbox_from_mask(mask):
    ys, xs = where(mask > 0)
    return [xs.min(), ys.min(), xs.max(), ys.max()]

def add_margin(box, margin, width, height):
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width - 1, x2 + margin),
        min(height - 1, y2 + margin),
    ]

def center_point_from_mask(mask):
    dist = distance_transform(mask)
    y, x = argmax_2d(dist)
    if mask[y, x] == 0:
        y, x = centroid(mask)
    return [x, y]

def main(args):
    rows = read_csv(args.slice_manifest)
    prompt_rows = []
    for row in rows:
        mask = read_binary_mask(row.mask_path)
        if mask.sum() == 0:
            continue
        components = connected_components_2d(mask)
        for comp_id, comp in enumerate(components):
            comp_mask = component_to_mask(comp)
            area = comp_mask.sum()
            margin = args.bbox_margin_small if area < 64 else args.bbox_margin_large
            box = add_margin(bbox_from_mask(comp_mask), margin, width=mask.shape[1], height=mask.shape[0])
            point = center_point_from_mask(comp_mask)
            prompt_rows.append({
                "case_id": row.case_id,
                "split": row.split,
                "z_index": row.z_index,
                "image_path": row.image_path,
                "mask_path": row.mask_path,
                "label_name": row.label_name,
                "component_id": comp_id,
                "bbox_xyxy": box,
                "point_xy": point,
                "point_label": 1,
                "prompt_type": args.prompt_type,
            })
    write_csv(args.output, prompt_rows)
```

实现注意：

- 坐标要明确是原图坐标还是 resize 后坐标。
- SAM-Med2D 通常使用 resize 后图像坐标；stitch 回 3D 时再靠 manifest 还原。
- 小病灶 bbox margin 不要太大，否则 prompt 会吞入大量背景。

### 5.4 `s4_train_sam_med2d.py`

职责：调用或包装 SAM-Med2D 训练逻辑，完成 2D fine-tune。

输入：

```text
--sam-med2d-root /path/to/SAM-Med2D
--dataset-dir S4_outputs/sam_med2d_dataset
--prompt-manifest S4_outputs/manifests/prompt_manifest.csv
--task whole_abnormal
--config S4_configs/sam_med2d_whole_abnormal.yaml
--output-dir S4_outputs/checkpoints/sam_med2d_whole_abnormal
--seed 2026
```

输出：

```text
checkpoint.pth
train_log.csv
config_resolved.yaml
```

伪代码：

```python
def build_config(args):
    cfg = load_yaml(args.config)
    cfg["data"]["image_dir"] = args.dataset_dir / "images"
    cfg["data"]["mask_dir"] = args.dataset_dir / "masks"
    cfg["data"]["image2label_train"] = args.dataset_dir / "image2label_train.json"
    cfg["data"]["label2image_val"] = args.dataset_dir / "label2image_val.json"
    cfg["train"]["seed"] = args.seed
    cfg["output_dir"] = args.output_dir
    return cfg

def main(args):
    set_seed(args.seed)
    cfg = build_config(args)
    validate_dataset_exists(cfg)
    write_yaml(args.output_dir / "config_resolved.yaml", cfg)

    if use_original_repo_cli:
        cmd = [
            "python", args.sam_med2d_root / "train.py",
            "--config", args.output_dir / "config_resolved.yaml",
            "--work_dir", args.output_dir,
        ]
        run_subprocess(cmd, log=args.output_dir / "train_stdout.log")
    else:
        model = load_sam_med2d_model(cfg)
        optimizer = build_optimizer(model, cfg)
        train_loader = build_loader(cfg, split="train")
        val_loader = build_loader(cfg, split="val")
        best_metric = -inf
        for epoch in range(cfg.train.epochs):
            model.train()
            for batch in train_loader:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                prompt = batch["prompt"].to(device)
                pred = model(image, prompt)
                loss = dice_loss(pred, mask) + 0.5 * ce_loss(pred, mask)
                if cfg.train.use_focal:
                    loss = loss + 0.3 * focal_loss(pred, mask)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            metrics = validate(model, val_loader)
            append_log(epoch, loss, metrics)
            if metrics["dice"] > best_metric:
                best_metric = metrics["dice"]
                save_checkpoint(model, args.output_dir / "best.pth")
```

实现注意：

- 优先复用 SAM-Med2D 官方训练入口，不建议第一版重写整个训练框架。
- 训练日志必须记录 task、模态策略、prompt 类型、随机种子、checkpoint 路径。

### 5.5 `s4_infer_sam_med2d.py`

职责：对 validation slices 跑 SAM-Med2D 推理，输出 2D mask。

输入：

```text
--checkpoint S4_outputs/checkpoints/sam_med2d_whole_abnormal/best.pth
--prompt-manifest S4_outputs/manifests/prompt_manifest.csv
--split val
--output-dir S4_outputs/predictions/val_2d_masks
```

输出：

```text
pred_2d_masks/*.png
pred_manifest.csv
```

伪代码：

```python
def main(args):
    model = load_checkpoint(args.checkpoint)
    model.eval()
    prompt_rows = read_csv(args.prompt_manifest)
    pred_rows = []

    for row in prompt_rows:
        if row.split != args.split:
            continue
        image = read_image(row.image_path)
        prompt = make_prompt_tensor(row.bbox_xyxy, row.point_xy, row.prompt_type)
        with no_grad():
            pred_prob = model.predict(image, prompt)
        pred_mask = threshold_or_argmax(pred_prob)
        pred_path = save_png(pred_mask * 255, args.output_dir, row)
        pred_rows.append({
            **row,
            "pred_mask_path": pred_path,
            "checkpoint": args.checkpoint,
        })

    write_csv(args.output_dir / "pred_manifest.csv", pred_rows)
```

实现注意：

- 如果一个 slice 有多个 prompt，需要保存多个 prediction，后续 stitch 时再 merge。
- 保存 prediction 时文件名必须包含 `case_id/z_index/task/component_id`。

### 5.6 `s4_eval_sam_3d_stitch.py`

职责：把 2D predictions stitch 回 3D，并计算内部 validation 指标。

输入：

```text
--case-manifest S4_outputs/manifests/s4_case_manifest.csv
--pred-manifest S4_outputs/predictions/val_2d_masks/pred_manifest.csv
--task whole_abnormal
--output-dir S4_outputs/predictions/val_3d_nifti
--metrics-dir S4_outputs/metrics
```

输出：

```text
val_3d_nifti/*.nii.gz
internal_val_results.csv
small_lesion_results.csv
```

伪代码：

```python
def merge_slice_predictions(preds_for_slice, task):
    canvas = zeros_like_slice()
    for pred in preds_for_slice:
        mask = read_binary_mask(pred.pred_mask_path)
        mask = resize_nearest(mask, original_hw_from_manifest(pred))
        if task in ["whole_abnormal", "lesion_core", "et", "rc"]:
            canvas = logical_or(canvas, mask)
        elif task == "class_wise":
            label_id = label_name_to_id(pred.label_name)
            canvas[mask > 0] = resolve_priority(canvas[mask > 0], label_id)
    return canvas

def compute_case_metrics(pred_3d, gt_3d, spacing):
    metrics = {}
    metrics["dice"] = dice(pred_3d > 0, gt_3d > 0)
    metrics["nsd"] = normalized_surface_dice(pred_3d, gt_3d, spacing)
    metrics["lesion_recall"] = lesion_wise_recall(pred_3d, gt_3d, spacing)
    metrics["fp_components"] = false_positive_components(pred_3d, gt_3d)
    metrics.update(size_bucket_metrics(pred_3d, gt_3d, spacing, buckets=[27, 275]))
    return metrics

def main(args):
    cases = read_csv(args.case_manifest)
    preds = group_by(read_csv(args.pred_manifest), keys=["case_id", "z_index"])
    all_metrics = []

    for case in cases:
        if case.split != "val" or not case.qc_pass:
            continue
        ref_img = load_nifti(case.t1c_path)
        pred_3d = zeros(ref_img.shape)
        for z in range(ref_img.shape[2]):
            slice_preds = preds.get((case.case_id, z), [])
            if slice_preds:
                pred_3d[:,:,z] = merge_slice_predictions(slice_preds, args.task)

        if args.task == "whole_abnormal":
            gt = load_nifti(case.effective_seg_path) > 0
        elif args.task == "lesion_core":
            gt = isin(load_nifti(case.effective_seg_path), [1,3,4])
        elif args.task == "et":
            gt = load_nifti(case.effective_seg_path) == 3
        else:
            gt = load_nifti(case.effective_seg_path)

        save_nifti(pred_3d, affine=ref_img.affine, header=ref_img.header, path=out_path(case))
        metrics = compute_case_metrics(pred_3d, gt, spacing=case.spacing)
        all_metrics.append({"case_id": case.case_id, **metrics})

    write_csv(args.metrics_dir / "internal_val_results.csv", all_metrics)
    write_csv(args.metrics_dir / "small_lesion_results.csv", extract_small_lesion_rows(all_metrics))
```

实现注意：

- 2D prediction 必须 resize 回原始 H/W 后再 stitch。
- 保存 NIfTI 必须继承 affine/header。
- class-wise 合并时必须使用统一优先级，默认 `RC > ET > NETC > SNFH`。

### 5.7 `s4_run_medsam_gt_bbox.py`

职责：跑 MedSAM 的 GT bbox 上限实验。

输入：

```text
--medsam-root /path/to/MedSAM
--slice-manifest S4_outputs/manifests/slice_manifest.csv
--prompt-manifest S4_outputs/manifests/prompt_manifest.csv
--checkpoint /path/to/medsam_vit_b.pth
--split val
--output-dir S4_outputs/predictions/medsam_gt_bbox
```

输出：

```text
medsam_pred_2d_masks/*.png
medsam_pred_manifest.csv
```

伪代码：

```python
def main(args):
    medsam = load_medsam(args.checkpoint)
    prompts = read_csv(args.prompt_manifest)
    pred_rows = []

    for row in prompts:
        if row.split != args.split:
            continue
        image = read_image(row.image_path)
        image_1024 = resize(image, (1024, 1024))
        box_1024 = scale_box(row.bbox_xyxy, from_size=image.shape[:2], to_size=(1024,1024))
        with no_grad():
            pred = medsam_infer(image_1024, box_1024)
        pred_original = resize_nearest(pred, image.shape[:2])
        pred_path = save_png(pred_original * 255, args.output_dir, row)
        pred_rows.append({**row, "pred_mask_path": pred_path})

    write_csv(args.output_dir / "medsam_pred_manifest.csv", pred_rows)
```

实现注意：

- 这个脚本只用于上限实验。
- 不能把 GT bbox 实验写成真实 Validation 性能。

### 5.8 `s4_build_msa3d_dataset.py`

职责：为 Medical-SAM-Adapter 构建 BraTS2026METDataset 索引或缓存。

输入：

```text
--case-manifest S4_outputs/manifests/s4_case_manifest.csv
--task lesion_core|et|whole_abnormal|class_wise
--modality-policy t1c_only|four_modalities
--patch-size 128,128,128
--output-dir S4_outputs/msa3d_dataset
```

输出：

```text
msa3d_index.csv
cached_patches/*.npz optional
```

伪代码：

```python
def make_binary_label(seg, task):
    if task == "whole_abnormal":
        return seg > 0
    if task == "lesion_core":
        return isin(seg, [1,3,4])
    if task == "et":
        return seg == 3
    if task == "class_wise":
        return seg

def sample_patch_centers(label, patch_size, num_positive, num_negative):
    positives = coordinates_where(label > 0)
    centers = []
    for _ in range(num_positive):
        centers.append(random_choice(positives))
    for _ in range(num_negative):
        centers.append(random_background_center(label))
    return centers

def main(args):
    cases = read_csv(args.case_manifest)
    index_rows = []
    for case in cases:
        if not case.qc_pass:
            continue
        image = load_modalities_by_policy(case, args.modality_policy)
        seg = load_nifti(case.effective_seg_path)
        label = make_binary_label(seg, args.task)
        centers = sample_patch_centers(label, args.patch_size, num_positive=8, num_negative=2)
        for i, center in enumerate(centers):
            patch_bbox = bbox_from_center(center, args.patch_size, image.shape)
            if args.cache_npz:
                img_patch = crop(image, patch_bbox)
                lab_patch = crop(label, patch_bbox)
                cache_path = save_npz(img_patch, lab_patch, case.case_id, i)
            else:
                cache_path = None
            index_rows.append({
                "case_id": case.case_id,
                "split": case.split,
                "patch_id": i,
                "patch_bbox": patch_bbox,
                "cache_path": cache_path,
                "modality_policy": args.modality_policy,
                "task": args.task,
            })
    write_csv(args.output_dir / "msa3d_index.csv", index_rows)
```

实现注意：

- 第一版建议 `t1c_only`，降低结构改造风险。
- patch 采样要过采样阳性病灶，尤其小病灶。
- 3D 输入的 spacing/resize 策略要单独记录，不要无说明地全量 resize 到固定 128³。

### 5.9 `s4_train_medical_sam_adapter_3d.py`

职责：训练或微调 Medical-SAM-Adapter 3D 版本。

输入：

```text
--msa-root /path/to/Medical-SAM-Adapter
--msa3d-index S4_outputs/msa3d_dataset/msa3d_index.csv
--task lesion_core
--modality-policy t1c_only
--output-dir S4_outputs/checkpoints/msa3d_lesion_core
```

输出：

```text
best.pth
train_log.csv
config_resolved.yaml
```

伪代码：

```python
class BraTS2026METDataset(Dataset):
    def __init__(self, index_csv, split, modality_policy, task):
        self.rows = read_csv(index_csv, where={"split": split})
        self.modality_policy = modality_policy
        self.task = task

    def __getitem__(self, idx):
        row = self.rows[idx]
        if row.cache_path:
            image, label = load_npz(row.cache_path)
        else:
            case = lookup_case(row.case_id)
            image = load_modalities_by_policy(case, self.modality_policy)
            seg = load_nifti(case.effective_seg_path)
            label = make_binary_label(seg, self.task)
            image = crop(image, row.patch_bbox)
            label = crop(label, row.patch_bbox)
        point = sample_positive_point(label)
        return {
            "image": to_tensor(image),        # [C,H,W,D]
            "label": to_tensor(label),        # [H,W,D]
            "p_label": tensor([1]),
            "pt": to_tensor(point),
            "image_meta_dict": {"filename_or_obj": row.case_id},
        }

def main(args):
    train_ds = BraTS2026METDataset(args.msa3d_index, "train", args.modality_policy, args.task)
    val_ds = BraTS2026METDataset(args.msa3d_index, "val", args.modality_policy, args.task)
    model = load_medical_sam_adapter(args.msa_root, thd=True)
    optimizer = build_optimizer(model)
    for epoch in range(num_epochs):
        train_one_epoch(model, train_ds, optimizer)
        metrics = validate_3d(model, val_ds)
        log(metrics)
        if metrics["dice"] > best:
            save_checkpoint(model, "best.pth")
```

实现注意：

- 如果原 repo 不支持 `C=4`，先使用 `t1c_only`。
- 不要第一版就改 4 通道 patch embedding，先跑通数据和训练闭环。

### 5.10 `s4_report_sam_results.py`

职责：汇总 S4 所有实验，生成报告。

输入：

```text
--metrics-dir S4_outputs/metrics
--configs-dir S4_outputs/configs
--output S4_outputs/reports/report_sam_validation.md
```

输出：

```text
report_sam_validation.md
```

伪代码：

```python
def main(args):
    metrics = load_all_metrics(args.metrics_dir)
    configs = load_all_configs(args.configs_dir)
    sections = []

    sections.append(render_experiment_table(metrics, configs))
    sections.append(render_model_input_contract(configs))
    sections.append(render_prompt_summary(configs))
    sections.append(render_metric_summary(metrics, keys=[
        "dice", "nsd", "lesion_recall", "lesion_f1",
        "small_lesion_recall", "fp_components_per_case",
    ]))
    sections.append(render_failure_analysis(metrics))
    sections.append(render_recommendations(metrics))

    write_markdown(args.output, sections)
```

报告必须明确：

- 官方 Validation 无标签，不报告 Dice；
- GT bbox 是上限实验；
- 粗分割 prompt 才接近真实自动推理；
- SAM-Med2D 的 2D 结果需要关注 3D 连续性；
- 与 nnU-Net baseline 的关系是 refinement / auxiliary，不是默认替代主模型。

## 6. 推荐输出目录

```text
S4_outputs/
  manifests/
    s4_case_manifest.csv
    train_val_split.json
    slice_manifest.csv
    prompt_manifest.csv
  sam_med2d_dataset/
    images/
    masks/
    image2label_train.json
    label2image_val.json
  msa3d_dataset/
    msa3d_index.csv
    cached_patches/
  checkpoints/
    sam_med2d_whole_abnormal/
    sam_med2d_lesion_core/
    sam_med2d_et/
    msa3d_lesion_core/
  predictions/
    val_2d_masks/
    val_3d_nifti/
    medsam_gt_bbox/
  metrics/
    internal_val_results.csv
    small_lesion_results.csv
  reports/
    report_sam_validation.md
```

## 7. 评价指标与报告要求

内部 validation fold 上至少报告：

- Dice；
- NSD；
- lesion-wise recall；
- lesion-wise F1；
- `<27 mm^3` 小病灶 recall；
- `27-275 mm^3` 小病灶 recall；
- `>275 mm^3` 病灶分割指标；
- false positive components per case；
- ET/NETC/SNFH/RC 单类表现。

报告中必须区分：

| 实验类型 | 能说明什么 | 不能说明什么 |
|---|---|---|
| GT bbox 上限实验 | SAM 给定正确 prompt 时的 refine 能力 | 不能说明全自动推理能力 |
| 粗分割 prompt 实验 | SAM 作为 refinement 的实际潜力 | 受粗分割质量影响 |
| SAM-Med2D fine-tune | 2D 医学切片适配能力 | 不能天然保证 3D 连续性 |
| Medical-SAM-Adapter 3D | 3D context 价值 | 工程成本更高，需单独消融 |

## 8. S4 需要确认的信息

S4 开始实现前需要确认：

1. 第一版优先做 SAM-Med2D 还是 MedSAM 上限实验。
2. 内部 validation fold 使用哪份 split。
3. 是否已有 S1/S2 nnU-Net baseline，可作为自动 prompt 候选生成器。
4. 第一版输入模态使用 `t1c` 单模态还是 `t1c/t2f/t2w` pseudo-RGB。
5. 第一版任务做 `whole_abnormal`、`lesion_core` 还是 `ET only`。
6. 是否有 GPU 资源支持 SAM-Med2D fine-tune。
7. 是否允许安装/使用各 SAM repo 的依赖环境。
8. 是否需要对官方 Validation 输出推理 NIfTI 作为展示。

## 9. 参考资料

- MedSAM official repo：<https://github.com/bowang-lab/MedSAM>
- MedSAM inference code：<https://raw.githubusercontent.com/bowang-lab/MedSAM/main/MedSAM_Inference.py>
- MedSAM CT/MR preprocessing：<https://raw.githubusercontent.com/bowang-lab/MedSAM/main/pre_CT_MR.py>
- SAM-Med2D official repo：<https://github.com/OpenGVLab/SAM-Med2D>
- SAM-Med2D DataLoader：<https://raw.githubusercontent.com/OpenGVLab/SAM-Med2D/main/DataLoader.py>
- SAM-Med2D data demo：<https://github.com/OpenGVLab/SAM-Med2D/tree/main/data_demo>
- SAM-Adapter original repo：<https://github.com/tianrun-chen/SAM-Adapter-PyTorch>
- Medical-SAM-Adapter repo：<https://github.com/ImprintLab/Medical-SAM-Adapter>
- Medical-SAM-Adapter BraTS loader：<https://raw.githubusercontent.com/ImprintLab/Medical-SAM-Adapter/main/dataset/brat.py>
- Medical-SAM-Adapter dataset guide：<https://raw.githubusercontent.com/ImprintLab/Medical-SAM-Adapter/main/guidance/Dataset.md>
