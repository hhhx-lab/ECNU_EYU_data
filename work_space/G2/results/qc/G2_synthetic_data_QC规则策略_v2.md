# G2 Synthetic Data QC 规则策略 v2

更新日期：2026-07-12

## 1. 目的

本策略用于检查 G1 V2 augmentation 和 G1 V3 missing-T2W completion。它回答四个问题：

1. 数据是否可追溯和可复现。
2. 数据格式、空间和标签是否正确。
3. 生成影像是否达到进入人工或 teacher 复核的最低质量。
4. 通过 QC 的数据是否能在固定真实验证/测试集上提高分割任务表现。

QC 不是单个指标，也不是“图看起来像”。正式放行需要技术硬门、质量复核和训练价值验收三个层级。

## 2. 数据身份

### 2.1 V2 augmentation

1. source 必须是 master train。
2. source T2W 必须为 authentic。
3. source 必须在 `g1_v2_source_manifest.csv` 中明确 allowed。
4. 输出建立新的稳定 synthetic ID。
5. 只可进入 train。

### 2.2 V3 completion

1. source 必须在 master mapping。
2. source 必须在 265 例 fake/broken T2W 清单。
3. 保留原病例 ID 和原 nnU-Net ID。
4. 只替换 T2W。
5. train/val/test 身份保持不变。

## 3. 当前固定数据口径

| 数据 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| master | 1035 | 130 | 130 | 1295 |
| real-only authentic T2W | 823 | 103 | 104 | 1030 |
| completion 目标 | 212 | 27 | 26 | 265 |

患者组 overlap 必须为 0。任何 V2 synthetic 出现在 val/test 都是硬拒绝。

## 4. Release 状态

| 状态 | 条件 | 用途 |
|---|---|---|
| `rejected` | 任一硬门失败 | 不使用，必要时重生成 |
| `pending_review` | 技术通过，但无人工/teacher/批次审批 | 不训练 |
| `accepted_for_training` | train 数据通过并获批 | 可进入训练 |
| `accepted_for_evaluation` | val/test completion 通过并获批 | 只用于原 split 评估 |

不得再使用 `accepted_for_ablation_only`。受控消融也必须使用正式 accepted 数据，只是实验组不同。

## 5. L0 交付完整性

### 检查内容

1. run root 存在。
2. `generation_config.json` 或 V3 `inference_run.json` 存在。
3. `generation_log.jsonl` 存在。
4. `synthetic_generation_manifest.csv` 存在。
5. run ID、generator name、seed 和 source CSV 有值。
6. V2 有四模态 checkpoint 或 checkpoint 目录。
7. V3 有 VAE、EncDec、BBDM checkpoint、`bbdm_s` 和固定 `validation_run`。
8. 当前病例在 manifest 与 JSONL log 中各有唯一 `status=success` 记录，且 source、run ID、seed 与配置一致。

### 判定

任一缺失：`rejected: metadata_incomplete`。

G2 可以从标准输出生成自己的 candidate manifest，但不能把缺失的 G1 运行证据伪装成原始交付证据。

## 6. L1 文件与命名

每病例必须有：

```text
t1n, t1c, t2w, t2f, seg
```

允许 native suffix：

```text
-t1n.nii.gz
-t1c.nii.gz
-t2w.nii.gz
-t2f.nii.gz
-seg.nii.gz
```

legacy suffix 只允许由 intake 明确记录转换，不允许一个病例混用 native/legacy。平铺 V2 多病例目录必须先过 composer，不能直接走通用 intake。

缺文件、命名冲突、混合 suffix 或一个目录出现多个病例：硬拒绝。

## 7. L2 NIfTI 与空间几何

逐例检查：

1. 文件可读。
2. 三维 shape 一致。
3. spacing 一致。
4. affine 一致。
5. orientation 一致。
6. output shape 与 source 一致。
7. affine/header 已由 composer 恢复，不保留 V2 单位阵。

任一失败：硬拒绝。

G1 V3 在进入 265 例 completion 前，其 Stage 5 paired validation 还必须提供
同一 run 的 `spatial_audit.csv`。审计行数和 ID 集合必须与 `metrics.csv`
完全一致，不允许重复 ID；每例 `foreground_outside_voxel_count` 和
`lesion_outside_voxel_count` 必须为 0。该门在生成任何 G2 QC 结果前执行，
不能用事后修 header 替代。

## 8. L3 数值与标签

### 图像

1. 不含 NaN/Inf。
2. 不是常数图。
3. 四模态均有有效脑区。

### Segmentation

1. 数值必须为整数。
2. 值域必须是 `{0,1,2,3,4}`。
3. 当前任务训练病例不得为空 mask。
4. 病灶不得触碰影像边界。
5. 病灶位于多模态非零脑区的比例至少为 0.95。

### Source seg 保护

当前 V2/V3 都不生成新 seg。输出 seg 必须与 source corrected seg 逐体素一致：

```text
source_seg_change_ratio = changed_voxels / all_voxels
要求 = 0
```

任一失败：硬拒绝。

## 9. L4 Source、身份与泄漏

### V2

要求：

```text
source_split=train
t2w_status=authentic
allowed_as_v2_source=True
synthetic_final_id != source_case_id
```

### V3

要求：

```text
source_is_fake_t2w_case=True
synthetic_final_id == source_case_id
nnunet_case_id == master nnunet_case_id
```

V3 val/test 不是泄漏，但只能 `accepted_for_evaluation`。V2 使用 val/test source 属于泄漏并硬拒绝。

## 10. L5 V2 composition 保护

V2 `generate_from_label.py` 的输出是 source-shape 数组，生成区外为 0。composer 生成 `generation_support` 后检查：

### 非生成区变化率

```text
nonroi_change_ratio = changed_voxels_outside_support / voxels_outside_support
硬门 <= 1e-4
```

正常 composer 应接近 0。超过阈值说明错误覆盖了 source。

### 边界误差

`roi_boundary_mae` 是生成体与 source 在 support shell 上的绝对误差，除以 source p99-p1 强度范围后求均值。

### 梯度不连续

`roi_boundary_gradient_jump` 比较生成体与 source 在 support shell 上的三维梯度幅值差，再按 source 强度范围归一化。

### Block artifact score

```text
artifact_block_score = max(boundary_mae, boundary_gradient_jump)
```

大于 0.25 进入高优先级人工复核，不自动 accepted。

## 11. L6 V3 受保护内容

V3 只允许改变 T2W。对 t1n/t1c/t2f 逐体素比较：

```text
protected_source_change_ratio =
  max(changed_ratio_t1n, changed_ratio_t1c, changed_ratio_t2f)
要求 = 0
```

比较使用相对于 source 强度范围的数值容差，避免浮点序列化微小误差。

以下任一情况硬拒绝：

1. 受保护模态无法比较。
2. `protected_source_change_ratio > 0`。
3. `source_seg_change_ratio > 0`。

生成 T2W 不与 fake/broken 旧 T2W 计算“真值 SSIM”。旧 T2W 不是可靠参考。

## 12. L7 病灶结构

从 `{1,3,4}` 联合病灶 mask 计算 26 邻域连通域：

1. lesion count。
2. min/p50/max lesion volume。
3. tiny `<27 mm3`。
4. small `27-275 mm3`。
5. large `>275 mm3`。
6. tiny lesion ratio。
7. RC 是否存在。

`tiny_lesion_ratio > 0.5` 自动进入人工复核。批次级分布要与真实 train 按中心和病灶大小比较，不能只看总体均值。

## 13. L8 多模态医学一致性

自动计算：

1. ET/T1C lesion-to-shell contrast。
2. SNFH/T2F lesion-to-shell contrast。
3. SNFH/T2W lesion-to-shell contrast。
4. 四模态 lesion ROI correlation。
5. label-modality alignment score。

alignment score 小于 1.0 进入高优先级复核。该阈值是最低筛查线，不是放射学诊断标准。

## 14. L9 连续性与强度

### Z 连续性

对病灶活跃切片计算：

1. `z_area_smoothness`：相邻病灶面积变化。
2. `z_intensity_smoothness`：相邻病灶平均强度变化。
3. `z_continuity_score`：上述两者均值。

小于 0.5 进入高优先级复核。少于两个活跃切片时留空，不伪填 1.0。

### V2 强度漂移

在 generation support 内，计算相对 source 强度范围归一化后的绝对差，并分别记录 p1/p50/p99。`p99 > 2.0` 进入复核。

### V2 ROI SSIM

`source_synth_roi_ssim` 使用真实图像与 composite 图像的 ROI crop 计算。它只用于发现过度复制或结构严重偏离，不单独决定放行。

V3 不对 fake/broken T2W 计算该指标。

## 15. L10 Teacher 与人工复核

### Teacher

通用自动 intake 不执行 teacher，字段必须为 `not_run`/空值，不得填伪 Dice。G1 V3 阶段 5
另有专用入口 `g2_s2_v3_teacher_eval.py`：使用冻结的 Dataset263 real-only S2 checkpoint，
在同一 103 例 fixed validation 上比较真实 T2W baseline 与生成 T2W 推理。专用报告不自动回填
通用 intake，也不跳过人工 montage 复核。

正式 teacher 应使用冻结的 real-only checkpoint，至少输出：

1. label 1/2/3/4 Dice。
2. lesion count difference。
3. missing large lesion count。
4. extra large lesion count。

teacher 训练数据、split 和 checkpoint 必须记录，避免自证循环。

### 人工复核

每例至少看：

1. 四模态三平面。
2. seg overlay。
3. V2 generation support 与边界。
4. V3 T2W 全脑覆盖和病灶区域。
5. RC、tiny lesion 和异常中心子组。

审批写入 `g2_approval_manifest.csv`。无审批一律 pending。

## 16. L11 批次级统计

每个 run 统计：

1. metadata 完整率。
2. NIfTI 可读率。
3. geometry 一致率。
4. 标签合法率。
5. rejected/pending/accepted 比例。
6. reject reason 分布。
7. 病灶数与体积分布。
8. 中心、split、RC 和小病灶分层。
9. V2 source 重复使用次数。
10. checkpoint/seed 分组异常。

以下仍是正式批次的扩展任务，不得写成已自动实现：MS-SSIM、medical FID/MMD、近似重复检测和放射科盲评。

## 17. L12 物化与 nnU-Net integrity

`g2_materialize_nnunet_dataset.py` 必须：

1. 只读取 approved manifest。
2. completion 只替换 T2W。
3. augmentation 只追加 train。
4. 使用统一通道 `t1n,t1c,t2w,t2f`。
5. 生成 nnU-Net 与 case-folder 双视图。
6. 生成 fixed split 和 materialization manifest。
7. 默认拒绝非空输出目录。
8. 做源文件预检和内置 NIfTI integrity。
9. 服务器有 nnU-Net 时运行官方 dataset integrity。

## 18. L13 训练价值验收

本层是 G2 数据是否最终采用的决定性证据。

必须比较：

```text
real-only
real + V3 completion
real + accepted V2 augmentation
real + V3 completion + accepted V2 augmentation
```

所有组共用当前 master split。历史不同 split 的 checkpoint 不能作为严格 paired baseline。

主指标：

1. ET/RC/TC/WT lesionwise DSC。
2. ET/RC/TC/WT lesionwise NSD。
3. ET/RC/TC/WT small-instance TP/FN/FP/F1。

同时报告各中心、tiny/small/large 和 RC 子组。只有真实固定验证/测试集不下降，且核心小病灶指标稳定或提升，才能推荐该批数据。

## 19. 自动放行边界

G2 脚本不会仅凭技术指标自动 accepted。逻辑为：

```text
硬门失败 -> rejected
硬门通过但无审批 -> pending_review
硬门通过 + 正确角色审批 -> accepted_for_training/evaluation
```

这避免以下错误：

1. 缺元数据却自动放行。
2. teacher 未运行却写成通过。
3. val/test completion 误进训练。
4. V2 raw output 未恢复 geometry 就进入下游。
5. 单个相似度指标替代完整质量判断。

## 20. 报告结论格式

每个 run 的最终结论必须写清楚：

1. 技术硬门是否通过。
2. 真实自动指标哪些已算、哪些未算。
3. 人工/teacher 审批依据。
4. accepted/rejected/pending 数量和原因。
5. materialization 与 integrity 结果。
6. 成对消融结果。
7. 是否进入主训练、仅保留研究记录或退回重生成。
