# G1-G2 GliGAN 兼容 Diffusion 数据契约

更新日期：2026-05-30
归属工作区：`work_space/G2`
上游理想契约来源：`data_space/task1_2026/datasets/数据契约.md`
实际兼容目标：2023 年 GliGAN 代码和论文中的生成式数据增强输入输出方式
适用对象：G1 生成模型组、G2 数据生成与质量控制组、S1/S2 nnU-Net 分割组

## 0. 核心结论

1. 原始理想契约要求 G1 直接输出 `SYN-MET-000001` 形式的完整病例目录，并且每例必须有 `metadata.json`。这个设计适合作为最终数据资产规范，但不适合作为第一阶段 G1-G2 对接的最低接口。
2. 第一阶段生成模型输入输出应沿用 2023 GliGAN 的真实工程习惯：CSV 驱动、病例路径驱动、局部 `96x96x96` ROI、四模态分别或联合重建、真实病例背景插入、输出 BraTS 风格 NIfTI 病例文件夹。
3. G1 不必在第一版直接生成完整 `metadata.json`。G2 负责在接收 GliGAN-compatible raw output 后生成 `synthetic_generation_manifest.csv`、`generation_log.jsonl`、QC 表和 nnU-Net 转换清单。
4. G1 原始输出保留 GliGAN 兼容命名，例如 `BraTS-MET-01094-003_fake_label_0` 或 `BraTS-MET-01094-003_real_label_0`。G2 在下游转换时再统一改成 `SYN-MET-000001` 或 nnU-Net `DatasetXXX` 的标准命名。
5. Diffusion 在本项目中的第一版定位不是整例 MRI 生成器，而是替代 GliGAN 的 modality generator：输入“被噪声遮住的局部 MRI crop + label crop”，输出“同一 ROI 内更真实的 synthetic tumour patch”。
6. 最终进入训练的数据仍必须满足 2026 Task 1 的标签、模态、NIfTI、QC 和防泄漏要求。GliGAN 兼容是 G1-G2 中间接口，不代表可以忽略 2026 数据规范。

## 1. 参考依据

### 1.1 本地代码依据

本契约参考以下本地文件：

1. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/train/csv_creator.py`
2. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/train/tumour_main.py`
3. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/train/label_main.py`
4. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/infer/main_random_label_random_dataset_generator_multiprocess.py`
5. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/utils/data_utils.py`
6. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/utils/gaussian_noise_tumour.py`
7. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/utils/gaussian_noise_tumour_extended.py`
8. `data_space/past_code/2023/Segmentation_Tasks/GliGAN/src/utils/convert_to_multi_channel_based_on_brats_classes.py`
9. `data_space/past_code/代码解析.md`

### 1.2 本地论文依据

本契约参考以下论文：

```text
data_space/past_articles/2023/2023年第一名.pdf
```

论文关键信息：

1. GliGAN 被用于生成可随机插入健康脑区的 synthetic tumours。
2. 生成目标是减少肿瘤标签与健康脑组织的类别不平衡，并增加肿瘤属性和位置多样性。
3. modality generator 使用 `96x96x96` 局部 crop。
4. generator 输入为噪声 scan `z` 和 label `y` 的拼接。
5. scan crop 先归一化到 `[-1, 1]`，肿瘤体素被 Gaussian noise 替换，周围体素可以按距离概率被替换，形成更自然的球形过渡。
6. 生成阶段使用两类标签来源：真实病例标签随机插入形成 `G`，随机 label generator 生成标签再插入形成 `rG`。
7. 论文明确指出 GliGAN 的主要限制是只能生成尺寸不超过 `96x96x96` 的肿瘤。

## 2. 契约分层

G1-G2 不应把所有规范混成一个接口。第一阶段必须分三层：

1. `L0_ideal_asset_contract`：最终理想数据资产，面向归档、复现、团队审计。
2. `L1_gligan_compatible_raw_contract`：G1-G2 第一阶段实际接口，面向生成模型运行。
3. `L2_training_export_contract`：G2 转换给 S1/S2 的 nnU-Net 训练数据，面向分割模型训练。

三层关系：

```text
2026 official real cases
  -> G2 source manifest / GliGAN CSV
  -> G1 diffusion generator, GliGAN-compatible local ROI mode
  -> G1 raw synthetic folders
  -> G2 synthetic_generation_manifest.csv + QC
  -> accepted synthetic cases
  -> nnU-Net DatasetXXX imagesTr / labelsTr
```

## 3. 模态命名映射

2023 GliGAN 代码内部使用 glioma 命名，2026 Task 1 使用 MET 官方命名。G2 必须维护一层映射，不能让 G1、G2、S1/S2 各写各的。

| 语义 | 2023 GliGAN CSV key | 2023 论文/习惯叫法 | 2026 Task 1 文件后缀 | G2 统一含义 |
|---|---|---|---|---|
| T1 native | `scan_t1` | T1 | `t1n` | non-contrast T1 |
| T1 contrast | `scan_t1ce` | T1Gd/T1ce | `t1c` | contrast-enhanced T1 |
| T2 | `scan_t2` | T2 | `t2w` | T2-weighted |
| FLAIR | `scan_flair` | FLAIR | `t2f` | T2-FLAIR |
| segmentation | `label` | label/seg | `seg` | 0/1/2/3/4 label |

执行规则：

1. G1 代码内部可以继续使用 `scan_t1ce/scan_t1/scan_t2/scan_flair`。
2. G2 写给 G1 的 CSV 必须使用 GliGAN 兼容列名。
3. G2 写给 S1/S2 的最终 NIfTI 必须使用 2026 官方后缀 `t1n/t1c/t2w/t2f/seg`。
4. G1 原始输出若保留 `scan_t1ce` 等后缀，G2 接收；但 G2 转换时必须标准化为 `t1c/t1n/t2w/t2f`。
5. 文档和 manifest 中必须同时记录 `gligan_key` 与 `brats2026_suffix`，避免后续脚本误读。

## 4. 标签契约

### 4.1 2026 Task 1 原始标签

| label | 缩写 | 含义 |
|---:|---|---|
| 0 | background | 背景 |
| 1 | NETC | non-enhancing tumour core |
| 2 | SNFH | surrounding non-enhancing FLAIR hyperintensity |
| 3 | ET | enhancing tumour |
| 4 | RC | resection cavity |

### 4.2 G1 模型输入 label channel

2023 GliGAN 不直接把单通道 label 喂给 modality generator，而是转成 region-based multi-channel label。

2026 第一阶段推荐使用 2024 post-treatment transform 的逻辑，因为它支持 RC：

```text
channel 0: TC = label 1 or label 3
channel 1: WT = label 1 or label 2 or label 3
channel 2: ET = label 3
channel 3: RC = label 4
```

执行规则：

1. 默认 `label_channels=4`。
2. 如果第一轮只生成非术后病灶并且明确不生成 RC，可以临时使用 `label_channels=3`，但 manifest 必须写 `label_channels=3` 和 `rc_policy=ignored_in_v1`。
3. G2 不允许把 2023 adult glioma 的三通道 mapping 静默用于包含 RC 的 2026 数据。
4. 最终 `seg.nii.gz` 必须还原为单通道整数标签，值域只能是 `{0,1,2,3,4}`。
5. 随机 label generator 如果暂时只会生成 `{1,2,3}`，必须标记为 `label_source=fake_no_rc`，不能冒充完整 2026 标签分布。

## 5. G2 写给 G1 的 source CSV 契约

### 5.1 文件位置

G2 应生成：

```text
work_space/G2/data/G2_outputs/manifests/g1_gligan_source_cases_v1.csv
```

这份 CSV 是 G1 训练和推理的第一入口，兼容 2023 `csv_creator.py` 的列名。

### 5.2 必填列

```csv
id,scan_t1ce,scan_t2,scan_flair,scan_t1,label,center_x,center_y,center_z,x_extreme_min,x_extreme_max,y_extreme_min,y_extreme_max,z_extreme_min,z_extreme_max,x_size,y_size,z_size
```

列含义：

| 列名 | 类型 | 说明 |
|---|---|---|
| `id` | string | source case ID，建议用完整 `BraTS-MET-xxxxx-xxx`，不要只截取尾号。 |
| `scan_t1ce` | path | 2026 `t1c.nii.gz` 路径。 |
| `scan_t2` | path | 2026 `t2w.nii.gz` 路径。 |
| `scan_flair` | path | 2026 `t2f.nii.gz` 路径。 |
| `scan_t1` | path | 2026 `t1n.nii.gz` 路径。 |
| `label` | path | corrected 后的 `seg.nii.gz` 路径。 |
| `center_x/y/z` | int | 当前病例目标 lesion 或候选 lesion 的质心坐标。 |
| `x/y/z_extreme_min/max` | int | lesion bounding box 边界，采用 Python slice 语义时要在脚本中明确是否包含右端点。 |
| `x/y/z_size` | int | bounding box 尺寸。 |

### 5.3 G2 扩展列

2023 代码不会使用这些列，但 G2 必须保留，用于 QC、防泄漏和复现：

```csv
case_id,source_split,has_corrected_label,corrected_label_path,label_values,lesion_component_id,lesion_volume_mm3,lesion_class_set,usable_for_gligan96,allowed_as_synthetic_source,exclude_reason,shape_x,shape_y,shape_z,spacing_x,spacing_y,spacing_z,affine_hash
```

执行规则：

1. `allowed_as_synthetic_source=false` 的病例不能进入 G1 生成。
2. validation 病例必须为 `allowed_as_synthetic_source=false`。
3. 含非法标签值的病例必须在修正前 `allowed_as_synthetic_source=false`。
4. 如果某个 lesion 任一轴尺寸超过 96，`usable_for_gligan96=false`。第一阶段不把它交给 G1 作为 label crop。
5. 对于多 lesion 病例，G2 可以一病例多行，每行对应一个 lesion component；此时 `id` 可以保留 case ID，但必须用 `lesion_component_id` 区分。

## 6. G1 输入契约

### 6.1 第一阶段默认生成模式

默认模式：

```text
generation_mode = local_insertion_gligan_compatible
```

含义：

1. 从 source CSV 读取一个真实病例。
2. 选择一个待插入标签，来源可以是真实 label crop，也可以是随机 label generator。
3. 在 source case 的健康脑区选择插入中心。
4. 以插入中心为中心裁剪 `96x96x96` ROI。
5. 把 label crop 修正为不出脑、不覆盖已有 lesion。
6. 对每个模态构造 noisy crop。

7. G1 diffusion 接收 noisy crop + label channels，输出 reconstructed crop。
8. G2 或 G1 wrapper 将 reconstructed crop 融合回完整 source case。
9.  输出完整 synthetic case 的四模态和 `seg`。

### 6.2 tensor 形状

如果 G1 保持 2023 GliGAN 的单模态生成方式：

```text
input_scan_noisy: [1, 1, 96, 96, 96]
input_label:      [1, C, 96, 96, 96]
input_cat:        [1, 1 + C, 96, 96, 96]
output_patch:     [1, 1, 96, 96, 96]
```

其中 `C=4` 为默认 2026 label channels；若第一轮不含 RC，可以临时 `C=3`。

如果 G1 使用多模态联合 diffusion：

```text
input_scan_noisy: [1, 4, 96, 96, 96]
input_label:      [1, C, 96, 96, 96]
input_cat:        [1, 4 + C, 96, 96, 96]
output_patch:     [1, 4, 96, 96, 96]
```

执行规则：

1. 单模态方式更贴近 2023 GliGAN 代码，G1 可先为四个模态分别推理。
2. 多模态联合方式可以作为今年 diffusion 的改进，但必须在 manifest 写明 `generator_io=multi_modal_joint`。
3. 不管 G1 内部用哪种方式，G2 最终只接收四模态完整 NIfTI + 单通道 seg。

### 6.3 intensity 归一化

沿用 GliGAN 逻辑：

1. ROI crop 按当前模态 crop 内部强度归一化到 `[-1, 1]`。
2. label 非零体素对应区域替换为 Gaussian noise。
3. 推荐优先使用 extended noise：不仅替换 lesion 体素，也按距离概率替换 lesion 周围体素，形成自然边界过渡。
4. G1 输出 patch 后需要反归一化或线性校正，使 ROI 边界和原始 source case 强度连续。
5. 如果 G1 直接输出原始强度空间，必须在 manifest 写明 `normalization_policy=raw_intensity_output`。

### 6.4 插入位置约束

G1/G2 选择插入中心时必须满足：

1. 插入中心在脑内，不能在全零背景。
2. `96x96x96` ROI 不应越界；如越界必须 padding 后再 crop 回原始 shape。
3. 新 label crop 不能覆盖 source case 已有 lesion。
4. 插入后 `seg` 不能出现 label 相加导致的非法值。
5. 若连续多次找不到可插入位置，必须跳过并写入 log。
6. 不允许用 validation 病例作为 source case。

## 7. G1 原始输出目录契约

### 7.1 输出根目录

G1 原始输出放在：

```text
work_space/G2/data/G2_outputs/synthetic_raw/gligan_compatible/
```

每轮生成一个 run：

```text
work_space/G2/data/G2_outputs/synthetic_raw/gligan_compatible/
  run_YYYYMMDD_HHMM_g1_diffusion_vX/
    generation_config.json
    generation_log.jsonl
    synthetic_generation_manifest.csv
    BraTS-MET-01094-003_fake_label_0/
      BraTS-MET-01094-003_fake_label_0-t1n.nii.gz
      BraTS-MET-01094-003_fake_label_0-t1c.nii.gz
      BraTS-MET-01094-003_fake_label_0-t2w.nii.gz
      BraTS-MET-01094-003_fake_label_0-t2f.nii.gz
      BraTS-MET-01094-003_fake_label_0-seg.nii.gz
    BraTS-MET-01184-002_real_label_0/
      ...
```

### 7.2 兼容旧后缀

如果 G1 直接复用 2023 GliGAN 代码，可能输出以下后缀：

```text
-scan_t1.nii.gz
-scan_t1ce.nii.gz
-scan_t2.nii.gz
-scan_flair.nii.gz
-seg.nii.gz
```

G2 接收这些 raw 文件，但转换给 S1/S2 前必须重命名为：

```text
-t1n.nii.gz
-t1c.nii.gz
-t2w.nii.gz
-t2f.nii.gz
-seg.nii.gz
```

### 7.3 case 目录命名

第一阶段 raw case 目录命名采用：

```text
<source_case_id>_<label_kind>_label_<label_index>
```

示例：

```text
BraTS-MET-01094-003_fake_label_0
BraTS-MET-01094-003_real_label_0
```

字段含义：

1. `source_case_id`：被插入 synthetic lesion 的真实训练病例。
2. `label_kind=real`：使用另一个真实病例 lesion crop 作为插入标签。
3. `label_kind=fake`：使用 label generator 或 G2 synthetic label sampler 生成标签。
4. `label_index`：同一 source case 下第几个 synthetic sample。

执行规则：

1. G1 raw 输出可以复用 source case ID 作为前缀，因为这是 GliGAN 兼容层。
2. G2 最终训练导出时必须重新分配唯一 synthetic ID，不能在 nnU-Net 训练集中和真实病例 ID 混淆。
3. 同一个 source case 生成多例时，必须使用不同 `label_index` 或不同 seed。
4. 目录名和文件名前缀必须一致。

## 8. `synthetic_generation_manifest.csv` 契约

### 8.1 文件位置

每个生成 run 必须包含：

```text
synthetic_generation_manifest.csv
```

如果 G1 暂时无法生成，G2 必须在接收后补建。没有 manifest 的 synthetic raw output 不能直接进入训练。

### 8.2 必填列

```csv
synthetic_raw_id,synthetic_final_id,source_case_id,source_split,label_kind,label_source_case_id,label_component_id,label_generator_checkpoint,generation_run_id,generator_name,generator_checkpoint_t1n,generator_checkpoint_t1c,generator_checkpoint_t2w,generator_checkpoint_t2f,generator_io,label_channels,rc_policy,noise_type,seed,insert_center_x,insert_center_y,insert_center_z,roi_x_min,roi_x_max,roi_y_min,roi_y_max,roi_z_min,roi_z_max,source_shape_x,source_shape_y,source_shape_z,output_shape_x,output_shape_y,output_shape_z,output_suffix_scheme,status,error_type,error_message,qc_status,qc_reject_reason,accepted_for_training
```

字段解释：

1. `synthetic_raw_id`：GliGAN 兼容 raw case ID，例如 `BraTS-MET-01094-003_fake_label_0`。
2. `synthetic_final_id`：G2 转换后分配的最终 ID，例如 `SYN-MET-000001`；生成初期可为空。
3. `source_case_id`：真实训练 source case。
4. `source_split`：必须是 `train`，不能是 `validation`。
5. `label_kind`：`real`、`fake`、`fake_no_rc`、`manual_template`。
6. `label_source_case_id`：真实 label crop 来源；如果是 fake label 可以为空。
7. `label_component_id`：多病灶病例中的 lesion component 编号。
8. `generator_io`：`single_modal_gligan` 或 `multi_modal_joint`。
9. `label_channels`：`3` 或 `4`。
10. `noise_type`：`gaussian_tumour` 或 `gaussian_extended`。
11. `insert_center_x/y/z`：插入中心坐标。
12. `roi_*`：完整病例中 ROI 的坐标范围。
13. `output_suffix_scheme`：`brats2026` 或 `legacy_gligan`。
14. `status`：`success`、`failed`、`skipped`。
15. `qc_status`：`not_run`、`accepted`、`rejected`、`needs_review`。
16. `accepted_for_training`：只有 G2 QC 后才能为 `true`。

## 9. `generation_log.jsonl` 契约

每行一个 JSON，必须至少包含：

```json
{
  "synthetic_raw_id": "BraTS-MET-01094-003_fake_label_0",
  "source_case_id": "BraTS-MET-01094-003",
  "generation_run_id": "run_20260530_2100_g1_diffusion_v1",
  "label_kind": "fake",
  "label_index": 0,
  "seed": 42,
  "status": "success",
  "start_time": "2026-05-30T21:00:00+08:00",
  "end_time": "2026-05-30T21:04:30+08:00",
  "duration_seconds": 270,
  "error_type": null,
  "error_message": null
}
```

失败样例：

```json
{
  "synthetic_raw_id": "BraTS-MET-01094-002_real_label_0",
  "source_case_id": "BraTS-MET-01094-002",
  "generation_run_id": "run_20260530_2100_g1_diffusion_v1",
  "label_kind": "real",
  "label_index": 0,
  "seed": 43,
  "status": "failed",
  "error_type": "illegal_source_label",
  "error_message": "source label contains value 6 before corrected-label overlay",
  "duration_seconds": 3
}
```

## 10. NIfTI 输出合法性

每个 raw synthetic case 必须满足：

1. 至少包含四模态和 `seg`。
2. 四模态和 `seg` 都能被 `nibabel` 读取。
3. 四模态和 `seg` shape 一致。
4. 四模态和 `seg` affine/header 与 source case 一致，或在 manifest 记录明确转换。
5. 图像不含 NaN/Inf。
6. `seg` 必须是整数标签。
7. `seg` 值域只能是 `{0,1,2,3,4}`。
8. 默认不允许空 `seg`。
9. ROI 边界不能有肉眼明显方块断层。
10. 插入 lesion 不能出现在脑外全零背景。
11. 插入 lesion 不能覆盖 source case 原有 lesion，除非 `generation_mode=real_label_regeneration`。
12. 原始 source case 的非 ROI 区域应保持不变，除非 manifest 说明做了全局强度校正。

## 11. G1 最低交付内容

G1 第一次给 G2 smoke test 时，最低交付：

1. 10-20 个 raw synthetic cases。
2. 生成命令或脚本。
3. `generation_config.json`。
4. `generation_log.jsonl`。
5. `synthetic_generation_manifest.csv`，如果 G1 暂时不能生成，至少输出足够日志让 G2 补建。
6. G1 checkpoint 路径和版本号。
7. label channel 说明。
8. intensity normalization 说明。
9. 是否使用 `gaussian_tumour` 或 `gaussian_extended`。
10. 是否单模态四次生成，还是多模态联合生成。

不要求第一版每例有 `metadata.json`。如果 G1 已经实现，可以输出，但 G2 不把它作为第一版硬门槛。

## 12. G2 接收后的处理

G2 收到 raw output 后按以下顺序处理：

1. 扫描 run 目录。
2. 读取或补建 `synthetic_generation_manifest.csv`。
3. 检查每例四模态和 `seg` 是否齐全。
4. 兼容旧后缀并映射到 2026 后缀。
5. 读取 NIfTI header、shape、affine、dtype。
6. 检查 `seg` 值域。
7. 检查 ROI 是否在脑内。
8. 检查 ROI 边界强度连续性。
9. 抽样生成 QC overlay。
10. 写 `synthetic_candidate_manifest.csv`。
11. 通过 QC 的写入 `synthetic_accepted_manifest.csv`。
12. 拒绝的写入 `synthetic_rejected_manifest.csv`，并保留拒绝理由。
13. 分配最终 `SYN-MET-xxxxxx` ID。
14. 转换到 nnU-Net `imagesTr/labelsTr`。
15. 与 S1/S2 约定 real-only vs real+synth 消融。

## 13. G2 到 S1/S2 的训练导出契约

G2 交给 S1/S2 的数据不再使用 GliGAN raw 命名，而使用 nnU-Net v2 规范：

```text
nnunet_raw/
  DatasetXXX_BraTS2026MET_G2SynthV1/
    dataset.json
    imagesTr/
      SYNMET000001_0000.nii.gz
      SYNMET000001_0001.nii.gz
      SYNMET000001_0002.nii.gz
      SYNMET000001_0003.nii.gz
    labelsTr/
      SYNMET000001.nii.gz
```

通道顺序固定：

```text
0000 = t1n
0001 = t1c
0002 = t2w
0003 = t2f
```

执行规则：

1. validation fold 只能包含真实病例。
2. synthetic case 不能进入 validation。
3. 如果 synthetic case 来自某个 source case，则该 source case 进入 validation 时，派生 synthetic case 必须从 training 中排除。
4. 每个 nnU-Net 导出版本都必须对应一个 accepted manifest。
5. 不允许把 rejected synthetic case 混进训练。

## 14. 第一轮 smoke test 验收标准

G1 第一轮 smoke output 通过以下条件，G2 才继续扩大生成规模：

1. raw cases 数量为 10-20。
2. 每例四模态和 `seg` 齐全。
3. `generation_log.jsonl` 存在。
4. G2 能补建或读取 `synthetic_generation_manifest.csv`。
5. 每例 shape、affine、spacing 一致。
6. `seg` 值域合法。
7. 不存在 NaN/Inf。
8. 至少 80% case 通过自动 QC。
9. 人工抽查 overlay 没有明显脑外插入、方块边界、严重模态不一致。
10. G2 可以无手工改名地转出 nnU-Net 测试 dataset。

## 15. 与理想契约的差异

| 项目 | 理想契约 | 第一阶段 GliGAN 兼容契约 |
|---|---|---|
| raw case ID | `SYN-MET-000001` | `<source_case_id>_<real/fake>_label_<n>` |
| per-case metadata | 必需 `metadata.json` | 不强制，改由 manifest/log 承担 |
| 输入单位 | 完整病例或 ROI 均可 | 默认 `96x96x96` ROI |
| label 输入 | JSON/mask 可自由定义 | CSV + label crop + region channels |
| 输出单位 | 完整规范 synthetic case | GliGAN raw case folder，再由 G2 标准化 |
| 模态命名 | `t1n/t1c/t2w/t2f` | 内部兼容 `scan_t1/scan_t1ce/scan_t2/scan_flair` |
| 下游训练 | 直接可训练 | G2 QC 和 nnU-Net 转换后可训练 |

## 16. 不能妥协的底线

1. validation 病例不能作为 synthetic training source。
2. corrected label overlay 必须先应用，再进入 G1。
3. 含非法 label 的病例不能静默进入生成。
4. G1 不能只交 PNG、截图或单模态示例。
5. G1 不能只交 patch 而不说明如何融合回 source case。
6. G2 不能把没有 manifest/log 的 raw output 直接给 S1/S2。
7. 2023 GliGAN 的硬编码 shape、ID 截断和 glioma label mapping 不能原样照搬到 2026。
8. 最终训练导出必须符合 2026 Task 1 模态和标签规范。

## 17. 当前 G2 立即要做的事

1. 生成 `g1_gligan_source_cases_v1.csv`。
2. 在 CSV 中把 2026 `t1c/t2w/t2f/t1n/seg` 映射成 GliGAN 的 `scan_t1ce/scan_t2/scan_flair/scan_t1/label`。
3. 对每个 lesion component 计算 center、bbox、size。
4. 标记 `usable_for_gligan96`。
5. 标记 `allowed_as_synthetic_source`。
6. 写 `synthetic_generation_manifest.csv` 模板。
7. 写 raw output 后缀映射表。
8. 写 G1 smoke test 验收脚本的字段清单。
9. 和 G1 确认第一版 `label_channels=4` 还是临时 `label_channels=3`。
10. 和 G1 确认第一版是 `single_modal_gligan` 还是 `multi_modal_joint`。
