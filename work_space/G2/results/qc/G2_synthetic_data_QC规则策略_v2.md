# G2 Synthetic Data QC 规则策略 v2（G1 raw output 适配版）

更新日期：2026-06-15
适用对象：G2 数据报告、G1 diffusion/MET-compatible raw output、S1/S2 real+synth nnU-Net 训练入口
主结论：G2 的 QC 目标不是证明生成图像“像真的”，而是证明 G1 raw output 能被 G2 变成可追溯、无泄漏、空间合法、标签合法、病灶合理、训练可用、消融有价值的数据资产。

## 1. QC 总原则

1. 先检查能不能训练，再评价质量好不好。
2. 先硬拒绝，再人工复查，再批次分级，最后用真实验证 fold 判断是否有用。
3. G1 第一阶段可以输出 legacy raw case；G2 负责接收、标准化、QC、manifest 和训练导出。
4. 缺少 metadata 不是第一阶段直接拒绝理由，但 source、checkpoint、seed、label_kind、run_id 无法恢复时必须拒绝。
5. validation leakage、非法 label、几何错配、不可读 NIfTI、严重 ROI 方块伪影，是不可妥协底线。

## 2. 2026 Task1 对 QC 的约束

| 项目 | 2026 要求 | G2 QC 影响 |
|---|---|---|
| 模态 | `t1n/t1c/t2w/t2f` | 每例四模态必须齐全 |
| 标签 | `{0,1,2,3,4}` | `seg` 只能含合法整数值 |
| label 1 | NETC | 应与 tumour core 结构一致 |
| label 2 | SNFH | 应在 t2f/t2w 上有合理高信号语境 |
| label 3 | ET | 应在 t1c 上有增强语境 |
| label 4 | RC | 只能在治疗后/真实 RC 语境下谨慎生成 |
| tiny lesion | `<27 mm3` 有临床意义 | 不能简单删除，但要防止噪点化 |
| segmentation lesion | `>275 mm3` 影响 Dice/NSD | large lesion 边界和模态一致性必须稳 |
| 数据空间 | native/T1C/SRI24 混合 | 不要求跨病例统一 shape，只要求病例内一致 |
| validation | 不公开 label | 不可作为 source，不可进入训练 |

### 2.1 官方 leaderboard 指标口径

2026-06-15 复核官方 Task1 Evaluation 和 Results 后，G2 后续训练价值验收必须以官方 leaderboard 字段为主：

| 类别 | 官方字段 |
|---|---|
| lesionwise segmentation | `lesionwise_dsc_mean_et`, `lesionwise_nsd_mean_et` |
| lesionwise segmentation | `lesionwise_dsc_mean_rc`, `lesionwise_nsd_mean_rc` |
| lesionwise segmentation | `lesionwise_dsc_mean_tc`, `lesionwise_nsd_mean_tc` |
| lesionwise segmentation | `lesionwise_dsc_mean_wt`, `lesionwise_nsd_mean_wt` |
| small-instance detection | `small_instance_tp_et`, `small_instance_fn_et`, `small_instance_fp_et`, `small_instance_f1_et` |
| small-instance detection | `small_instance_tp_tc`, `small_instance_fn_tc`, `small_instance_fp_tc`, `small_instance_f1_tc` |
| small-instance detection | `small_instance_tp_wt`, `small_instance_fn_wt`, `small_instance_fp_wt`, `small_instance_f1_wt` |
| small-instance detection | `small_instance_tp_rc`, `small_instance_fn_rc`, `small_instance_fp_rc`, `small_instance_f1_rc` |

当前主报告不再把 `lesion-wise AUC` 或 `HD95` 作为官方主指标；它们最多作为内部辅助分析。

G2 内部临时区域映射如下，最终以官方 eval container 为准：

| 区域 | 临时映射 |
|---|---|
| ET | `label == 3` |
| RC | `label == 4` |
| TC | `label in {1,3}` |
| WT | `label in {1,2,3}` |

RC 是官方单独区域 `label == 4`，不混入 TC/WT 代理区域。G2 内部 proxy 按官方 `BraTS_evaluation` 的 MET notebook 示例优先使用 `vol_threshold=27`、`overlap_threshold=0.2`；最终提交成绩以官方 Synapse scoring 端实际参数为准。

完整说明见：`G2_official_metrics_alignment_QC_strategy_2026-06-15.md`。

## 3. 当前真实数据参考

| 项目 | 数值 |
|---|---:|
| 本地带标签训练病例 | 1296 |
| corrected overlay 后 final QC pass | 1295 |
| corrected overlay 后 final QC fail | 1 |
| 官方 validation 病例 | 179 |
| fixed fold0 train / val | 1036 / 259 |
| 真实 lesion component 总数 | 9793 |
| tiny / small / large lesion 数 | 3788 / 3922 / 2083 |
| 含 RC 真实训练病例 | 167 |
| G1 MET 96 ROI source 候选 | 472 |

已知排除：

| 病例 | 原因 |
|---|---|
| `BraTS-MET-01094-002` | corrected overlay 后仍含非法 label value `6` |

## 4. QC 分层

| 层级 | 名称 | 目的 | 失败后动作 |
|---:|---|---|---|
| L0 | raw delivery intake | 接收 G1 legacy raw output | 缺关键追踪则拒绝或修复 |
| L1 | 文件完整性 | 检查四模态 + seg | 缺文件强制拒绝 |
| L2 | NIfTI 可读性与几何 | 检查 shape/spacing/affine | 病例内不一致强制拒绝 |
| L3 | 数组与 label 合法性 | 检查 NaN/Inf/常数图/label 值域 | 非法强制拒绝 |
| L4 | source 与泄漏 | 检查 source、fold、official validation | 泄漏强制拒绝 |
| L5 | G1 ROI 插入一致性 | 检查 ROI、非 ROI、source 回填 | 严重错误强制拒绝 |
| L6 | lesion-level 结构 | 检查体积、数量、bbox、label 组合 | 异常复查或拒绝 |
| L7 | 多模态医学一致性 | 检查 t1c/t2f/t2w/t1n 对应关系 | 异常复查 |
| L8 | 扩散生成痕迹 | 检查边界、z 连续、强度漂移、模式塌缩 | 严重伪影拒绝 |
| L9 | teacher model 辅助 | 用 real-only 分割模型发现离谱样本 | 异常复查 |
| L10 | batch distribution | 检查整批 synthetic 分布 | 偏差大则降级 |
| L11 | nnU-Net integrity | 检查训练入口 | 不通过整批不可训练 |
| L12 | real validation ablation | 判断是否真正有用 | 指标受损则回滚 |

## 5. L0 raw delivery intake

### 5.1 G1 raw run 最低内容

每个 run 至少应有：

```text
run_id/
  generation_config.json            # 强烈建议
  generation_log.jsonl              # 强烈建议
  synthetic_generation_manifest.csv # 可由 G2 补建
  <raw_case_id>/
    <raw_case_id>-scan_t1.nii.gz 或 <raw_case_id>-t1n.nii.gz
    <raw_case_id>-scan_t1ce.nii.gz 或 <raw_case_id>-t1c.nii.gz
    <raw_case_id>-scan_t2.nii.gz 或 <raw_case_id>-t2w.nii.gz
    <raw_case_id>-scan_flair.nii.gz 或 <raw_case_id>-t2f.nii.gz
    <raw_case_id>-seg.nii.gz
```

对于 G1 缺 T2W 补全模式，G2 也接受下面这种 completion 目录：

```text
run_root/
  BraTS-MET-00554-000/
    BraTS-MET-00554-000-t1n.nii.gz
    BraTS-MET-00554-000-t1c.nii.gz
    BraTS-MET-00554-000-t2f.nii.gz
    BraTS-MET-00554-000-seg.nii.gz
    BraTS-MET-00554-000-t2w.nii.gz
```

如果 run root 是 G1 T2W completion 输出，`case_dir` 直接视为 `source_case_id`，`label_kind=completion`，`label_index=0`。
completion 模式下，G2 不再要求 source 出现在 `g1_met_source_cases_v1.csv`，也不再要求 `usable_for_met96=True`；但 source 必须仍能在真实数据 manifest 中追溯到，且真实数据侧 final QC 通过。

如果 run root 是 G1 diffusion augmentation 输出，目录名虽然也可能是 `BraTS-MET-xxxxx-xxx`，但 G2 将其标记为 `label_kind=full_generation`。full_generation 不按 completion 处理，必须遵守 `g1_met_source_cases_v1.csv`、`usable_for_met96=True`、train-only source、无 val/test 泄漏等 synthetic augmentation 规则。

### 5.2 G2 可自动恢复的信息

G2 可以从 raw case 目录名恢复：

1. `source_case_id`
2. `label_kind`
3. `label_index`
4. `synthetic_raw_id`

G2 可以从 NIfTI 恢复：

1. output shape
2. spacing
3. affine hash
4. dtype
5. label values

### 5.3 G2 不能可靠恢复的信息

以下信息必须来自 G1 config/log/manifest，否则不能进入 accepted：

1. generator checkpoint。
2. seed。
3. label channel 数。
4. noise type。
5. source CSV 版本。
6. generation mode。
7. 如果目录名不可解析，则 source case。

## 6. L1 文件完整性

每例必须有：

| 语义 | 接收后缀 |
|---|---|
| T1 native | `t1n` 或 `scan_t1` |
| T1 contrast | `t1c` 或 `scan_t1ce` |
| T2 | `t2w` 或 `scan_t2` |
| FLAIR | `t2f` 或 `scan_flair` |
| segmentation | `seg` |

强制拒绝：

1. 缺任一模态。
2. 缺 `seg`。
3. 同一语义模态重复且无法判断使用哪个。
4. 文件扩展名不是 `.nii.gz` 或无法读取。

## 7. L2 NIfTI 几何

同一 synthetic case 内，五个文件必须：

1. shape 完全一致。
2. spacing 逐轴一致，容差 `<=1e-5`。
3. affine `np.allclose(atol=1e-4)`。
4. orientation 一致。
5. 都是 3D volume。

跨病例不要求统一 shape，因为 2026 MET 数据本身存在不同空间。

判定：

| 问题 | 判定 |
|---|---|
| 病例内 shape 不一致 | 强制拒绝 |
| 病例内 spacing 不一致 | 强制拒绝 |
| 病例内 affine 不一致 | 强制拒绝 |
| 与 source case 几何不一致且无说明 | 强制拒绝 |
| 与 source case 几何不一致但有 resampling 说明 | 人工复查 |

## 8. L3 数组与 label 合法性

### 8.1 图像数组

每个模态记录：

1. min / p1 / p50 / p99 / max。
2. mean / std。
3. nonzero ratio。
4. NaN/Inf。
5. 是否常数图。

强制拒绝：

1. NaN/Inf。
2. 全零或常数图。
3. 维度不是 3D。
4. 图像数组和 header shape 不一致。

### 8.2 label

检查：

1. 是否整数。
2. 值域是否属于 `{0,1,2,3,4}`。
3. 是否空 mask。
4. 各 label 体积。

强制拒绝：

1. 非整数 label。
2. 出现 `5/6/8/255` 等非法值。
3. 空 mask 且 metadata/config 没有允许。

## 9. L4 source 与泄漏

source 必须：

1. 在 `real_train_manifest.csv` 中存在。
2. `final_qc_pass=True`。
3. 不在 fixed fold0 validation。
4. 不来自 official validation。
5. 不是已排除病例。
6. 如果使用 G1 MET 96 ROI 模式，应来自 `usable_for_met96=True` 候选。

强制拒绝：

| 情况 | 原因 |
|---|---|
| source 来自 fixed val fold | 训练/验证泄漏 |
| source 来自 official validation | 官方验证集泄漏 |
| source 缺失 | 不可追溯 |
| source 不在 G2 manifest | 不可控 |
| synthetic ID 复用真实训练 ID | 容易混淆训练资产 |

## 10. L5 G1 ROI 插入一致性

G1 当前是局部 ROI 生成，因此 G2 必须检查 ROI 相关指标。

### 10.1 必查指标

| 指标 | 含义 | 判定 |
|---|---|---|
| `roi_bbox_available` | manifest/log 是否给出 ROI | 缺失则复查 |
| `roi_inside_image` | ROI 是否在图像内 | 否则拒绝 |
| `lesion_inside_roi_ratio` | 新 lesion 是否落在 ROI 内 | 过低拒绝 |
| `nonroi_change_ratio` | 非 ROI 区域是否被改动 | 高于阈值复查/拒绝 |
| `source_existing_lesion_overlap` | 是否覆盖原有 lesion | 未声明则拒绝 |
| `brain_mask_overlap_ratio` | lesion 是否在脑内 | 过低拒绝 |

### 10.2 推荐阈值

第一轮 smoke 阶段：

1. `nonroi_change_ratio <= 0.001`。
2. `brain_mask_overlap_ratio >= 0.98`。
3. `lesion_inside_roi_ratio >= 0.99`。
4. `source_existing_lesion_overlap=0`，除非 `generation_mode=real_label_regeneration`。

## 11. L6 lesion-level 结构

### 11.1 连通域定义

把 `{1,2,3,4}` 的并集作为 lesion mask，做 3D connected component。

每个 component 记录：

1. lesion id。
2. labels present。
3. volume mm3。
4. tiny/small/large bucket。
5. bbox。
6. center。
7. 是否含 RC。

### 11.2 体积分档

| 档位 | 标准 |
|---|---|
| tiny | `<27 mm3` |
| small | `27-275 mm3` |
| large | `>275 mm3` |

### 11.3 单例判定

| 情况 | 判定 |
|---|---|
| bbox 超出图像 | 强制拒绝 |
| lesion 大面积在脑外 | 强制拒绝 |
| lesion 数 `>28` | 人工复查 |
| lesion 数 `>56` | 高优先级复查 |
| tiny ratio `>35%` | 人工复查 |
| 最小 lesion `<1 mm3` | 人工复查 |
| 最大 lesion `>15213 mm3` | 人工复查 |
| RC 出现在无 RC 语境 | 人工复查或拒绝 |

tiny lesion 不是噪声的同义词；G2 拒绝的是不可解释的生成噪点，而不是所有 tiny lesion。

## 12. L7 多模态医学一致性

### 12.1 自动指标

| 指标 | 目的 |
|---|---|
| `et_t1c_contrast_ratio` | ET 在 t1c 上是否有合理增强 |
| `snfh_t2f_contrast_ratio` | SNFH 在 t2f 上是否有高信号语境 |
| `snfh_t2w_contrast_ratio` | SNFH 在 t2w 上是否合理 |
| `rc_t1n_t2f_profile_score` | RC 语境是否异常 |
| `cross_modality_roi_corr` | 四模态 ROI 是否空间一致 |
| `label_modality_alignment_score` | label 与图像异常是否对齐 |

### 12.2 人工复查触发

1. label 有 ET，但 t1c 完全没有对应增强。
2. label 有 SNFH，但 t2f/t2w 与周围组织无差别或边界极硬。
3. 四模态 lesion 位置不一致。
4. RC 大量出现但 source 或 label_kind 无 RC 依据。

## 13. L8 扩散生成质量

这一层专门衡量 G1 diffusion 生成质量。

### 13.1 局部质量

| 指标 | 说明 | 趋势 |
|---|---|---|
| `roi_boundary_mae` | ROI 边界内外强度差 | 越低越好 |
| `roi_boundary_gradient_jump` | 边界梯度跳变 | 越低越好 |
| `z_continuity_score` | 相邻 slice 面积/强度连续性 | 越高越好 |
| `intensity_drift_p50` | synthetic 与 source 中位数漂移 | 越低越好 |
| `artifact_block_score` | 方块伪影疑似程度 | 越低越好 |

### 13.2 多样性与复制风险

| 指标 | 说明 |
|---|---|
| `synth_synth_ms_ssim` | 同批 synthetic 过高提示模式塌缩 |
| `source_synth_roi_ssim` | ROI 与 source 原图过高可能没生成出新病灶 |
| `label_source_synth_roi_ssim` | 借用真实 label 时检查是否复制原病例影像 |
| `nearest_real_roi_feature_distance` | synthetic 是否过度接近某真实病例 ROI |
| `duplicate_hash_hit` | 是否存在重复样本 |

### 13.3 医学 feature 级指标

可以使用 FID/MMD，但不能直接用 ImageNet Inception FID。

允许路线：

1. 使用 real-only segmentation encoder 提取 3D ROI feature。
2. 使用医学 MRI 自监督 encoder。
3. 使用团队训练的 3D feature extractor。

指标：

1. `feature_fid_medical`。
2. `feature_mmd_medical`。
3. `roi_feature_distance_p50`。

这些指标只做辅助，不作为单独放行标准。

## 14. L9 teacher model 辅助

teacher model 建议使用 real-only nnU-Net baseline。

记录：

1. `teacher_dice_label_1..4`
2. `teacher_nsd_label_1..4`
3. `teacher_lesion_count_diff`
4. `teacher_missing_large_lesion_count`
5. `teacher_extra_large_lesion_count`

判定：

1. teacher 完全找不到 large lesion，人工复查。
2. teacher 预测与 synthetic label 完全相反，人工复查。
3. teacher 分数低不能单独拒绝，因为 synthetic 可能是困难样本。
4. teacher 异常必须写入报告。

## 15. L10 batch distribution

整批 synthetic 通过标准：

| 指标 | 第一轮标准 |
|---|---|
| validation leakage | 必须为 0 |
| hard reject rate | 目标 `<=5%` |
| unresolved manual review | 必须为 0 |
| accepted synthetic 总量 | 第一轮不超过 real train 25% |
| 每个 source accepted 数 | 默认最多 1 |
| tiny lesion ratio | 默认 `<=35%`，超出需解释 |
| RC synthetic | 只来自真实 RC 语境 |
| source 分布 | 不集中于少数病例 |
| checkpoint 分布 | 能按 checkpoint 分层追溯 |

必须报告：

1. label combination 分布。
2. tiny/small/large 分布。
3. 每例 lesion 数分布。
4. source case 使用次数。
5. label_kind 分布。
6. generation_mode 分布。
7. checkpoint/seed 覆盖。
8. reject reason 分布。

## 16. L11 nnU-Net integrity

accepted synthetic 导出后必须检查：

当前 G1 T2W completion 与后续 diffusion augmentation 的物化口径不同：

- completion 输出默认不是新增病例，而是替换对应真实病例的 fake/broken `t2w` 通道。
- fake/broken `t2w` 病例如果没有 accepted completion，默认不能以原始 fake/broken `t2w` 进入主训练数据。
- 非 completion 的 diffusion augmentation 才作为额外 synthetic case 追加到 real+synth 数据集。
- `accepted_for_ablation_only=True` 只有显式加 `--include-ablation-only` 时才进入受控消融数据集。
- fake/broken T2W 清单缺失时，物化脚本默认停止，不能假设所有真实病例 T2W 都可信。

1. `imagesTr/{case_id}_0000.nii.gz = t1n`
2. `imagesTr/{case_id}_0001.nii.gz = t1c`
3. `imagesTr/{case_id}_0002.nii.gz = t2w`
4. `imagesTr/{case_id}_0003.nii.gz = t2f`
5. `labelsTr/{case_id}.nii.gz = seg`
6. `dataset.json` channel_names 正确。
7. `numTraining` 与实际一致。
8. 运行 `nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity`。

不通过 integrity check 的批次不能进入训练。

## 17. L12 real validation ablation

QC 证明数据“能用”，ablation 证明数据“有用”。

最低实验：

| 实验 | 训练数据 | 验证 |
|---|---|---|
| real-only | fixed train real | fixed real val |
| real+synth smoke | fixed train real + smoke accepted | fixed real val |
| real+synth low ratio | fixed train real + 0.25x synth | fixed real val |
| policy ablation | 不同 G1 checkpoint/label_kind/noise_type | fixed real val |

采用标准：

1. `lesionwise_dsc_mean_et/rc/tc/wt` 不明显下降。
2. `lesionwise_nsd_mean_et/rc/tc/wt` 不明显下降。
3. `small_instance_f1_et/tc/wt/rc` 不下降，目标区域最好提升。
4. `small_instance_fn_*` 不增加，尤其 ET、TC、WT 小病灶。
5. `small_instance_fp_*` 不显著增加；若 FP 增加超过 10% 且 TP 无增加，该批次不进入主训练。
6. large lesion segmentation 不被牺牲。
7. RC 指标不被 synthetic 拉坏。
8. 若 synthetic 只改善某类病灶，报告必须限制使用结论。

## 18. 单例分级标准

| 等级 | 名称 | 标准 | 动作 |
|---|---|---|---|
| A | excellent | 硬检查全过，ROI 自然，多模态一致，teacher 无异常，人工抽查优秀 | 可训练，可展示 |
| B | accepted | 硬检查全过，无严重伪影，指标可接受 | 可训练 |
| C | ablation_only | 硬检查全过，但分布/teacher/视觉质量需验证 | 只做消融 |
| D | needs_regeneration | 可追溯但质量问题可由 G1 重跑修复 | 回传 G1 |
| F | rejected | 泄漏、非法 label、几何错误、严重伪影、不可追溯 | 不使用 |

## 19. 怎么证明“质量非常优秀”

一批数据只有满足以下证据，才能在报告中写“质量非常优秀”：

1. `validation_leakage_count=0`。
2. `hard_reject_rate<=5%`。
3. `manual_review_unresolved_count=0`。
4. `excellent_or_accepted_ratio>=90%`。
5. `label_values_valid_rate=100%`。
6. `geometry_consistent_rate=100%`。
7. `nifti_readable_rate=100%`。
8. `roi_boundary_pass_rate>=95%`。
9. `z_continuity_pass_rate>=95%`。
10. `modality_consistency_pass_rate>=90%`。
11. `batch_distribution_status=controlled`。
12. `nnunet_integrity_pass=True`。
13. real+synth ablation 不降低官方核心指标，并至少在一个目标官方字段上有提升或稳定收益。

如果没有第 13 条，只能说“QC 质量优秀”，不能说“训练价值优秀”。

## 20. 人工复查规则

每批必须复查：

1. 所有 hard reject 临界病例。
2. 所有 manual review 病例。
3. accepted 病例随机抽查不少于 `max(10, accepted_count * 10%)`。
4. 所有 RC synthetic。
5. tiny lesion ratio 最高的前 20 例。
6. 最大 lesion 体积前 20 例。
7. teacher 异常前 20 例。

每例至少查看：

1. `t1c + ET` overlay。
2. `t2f + SNFH` overlay。
3. 四模态同切片。
4. axial/coronal/sagittal 三方向。
5. ROI 边界前后 slice。

## 21. 输出文件

每批 run 输出：

```text
qc/
  official_leaderboard_metrics_{experiment_id}.csv
  qc_metrics_{run_id}.csv
  diffusion_quality_metrics_{run_id}.csv
  qc_case_review_{run_id}.csv
  qc_batch_summary_{run_id}.json
  G2_synthetic_data_quality_report_{run_id}.md
manifests/
  synthetic_candidate_manifest_{run_id}.csv
  synthetic_accepted_manifest_{run_id}.csv
  synthetic_rejected_manifest_{run_id}.csv
```

## 22. 数据报告必须回答

1. 这批数据由哪个 G1 checkpoint 生成。
2. 使用哪个 G2 source CSV。
3. 生成了多少例，成功读取多少例。
4. G2 能否恢复 source、seed、label_kind、ROI。
5. 是否存在 validation leakage。
6. label 是否全部合法。
7. geometry 是否全部一致。
8. ROI 边界是否自然。
9. z 轴是否连续。
10. 多模态是否合理。
11. lesion 分布是否可控。
12. teacher 是否发现异常。
13. 人工复查结论是什么。
14. 最终 accepted/rejected/needs_regeneration 各多少。
15. 是否通过 nnU-Net integrity。
16. 是否输出官方 leaderboard 同款字段。
17. 是否允许进入 real+synth 消融。
18. ablation 后是否证明它真的有用。

## 23. 第一轮建议

1. G1 先交 10-20 个 smoke cases。
2. G2 对 smoke cases 跑 L0-L11。
3. smoke 通过后生成 100-300 个候选。
4. 第一轮 accepted synthetic 不超过真实训练病例 25%。
5. 不做 full-case 从零生成进入主训练。
6. 不凭空生成 RC。
7. 不使用 official validation 或 fixed real val source。
8. 所有 accepted synthetic 必须完整追踪 source、checkpoint、seed、generation_mode 和 QC 版本。

## 24. 一句话

G2 的 QC 不是“挑好看的图”，而是把 G1 的 raw diffusion output 变成有证据链的数据资产；没有证据链的 synthetic data，不管视觉上多像，都不能进入训练。

## 25. 2026 Task1 官方口径锁定

本文件后续全部按 2026 Task1 Brain Metastases 口径执行。

1. 官方任务：Brain metastases detection and segmentation。
2. 官方输入：单通道 semantic segmentation。
3. 官方标签：`0 background, 1 NETC, 2 SNFH, 3 ET, 4 RC`。
4. 官方区域：`ET={3}`, `RC={4}`, `TC={1,3}`, `WT={1,2,3}`。
5. 官方主字段：`lesionwise_dsc_mean_*`、`lesionwise_nsd_mean_*`、`small_instance_tp/fn/fp/f1_*`。
6. G2 内部 proxy 默认：`vol_threshold=27`、`overlap_threshold=0.2`。
7. 最终成绩：以 Synapse scorer 或官方发布 container 为准。

禁止事项：

1. 禁止把 RC 混入 TC/WT。
2. 禁止用 NETC/SNFH/ET/RC 四类均值冒充 ET/RC/TC/WT。
3. 禁止把 FID、MS-SSIM、teacher Dice 写成官方主指标。
4. 禁止只用全局 Dice 判断 synthetic 是否有效。

## 26. G2 当前脚本责任分工

| 脚本 | 责任 | 输入 | 输出 |
|---|---|---|---|
| `code/g2_pretraining_audit.py` | 真实数据扫描、source CSV、模板、基础 synthetic intake 能力 | 本地 Task1 数据根目录，可选 G1 run | results 全量基础产物 |
| `code/g2_synthetic_raw_intake_qc.py` | 正式接收 G1 raw run 并生成 QC 闭环 | G1 synthetic raw run | candidate/accepted/rejected manifest、QC CSV、batch summary、质量报告 |
| `code/g2_materialize_nnunet_dataset.py` | 将 real mapping + accepted synthetic 转为 nnU-Net raw 入口 | real mapping、accepted synthetic manifest | dataset.json、materialization manifest、可选 symlink/copy |
| `code/g2_official_mets_metrics_parser.py` | 解析或校验官方 Task1 leaderboard 字段 | Panoptica JSON 或 metrics CSV | official-style metrics CSV 或字段校验结果 |

推荐执行顺序：

```bash
# 1. 训练前刷新真实数据 manifest 和模板
python work_space/G2/code/g2_pretraining_audit.py --force

# 2. G1 交付后接收 raw synthetic run
python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root /path/to/G1/run_id \
  --synthetic-run-id run_id \
  --generation-mode full_generation

如果接收的是 G1 T2W completion 输出，则把 `--generation-mode full_generation` 改成 `--generation-mode completion`。

# 3. 训练机上生成 real+synth nnU-Net raw dataset
python work_space/G2/code/g2_materialize_nnunet_dataset.py \
  --output-root /path/to/nnUNet_raw \
  --synthetic-accepted-manifest work_space/G2/results/manifests/synthetic_accepted_manifest_run_id.csv \
  --dataset-id 261 \
  --dataset-name BraTS2026_MET_RealSynth_G1V1 \
  --channel-order g2_official \
  --mode manifest-only

# completion 默认 auto：accepted completion 替换 fake/broken T2W，不追加重复病例。

# 4. S1/S2 训练后解析官方风格指标
python work_space/G2/code/g2_official_mets_metrics_parser.py parse-json \
  --json-path panoptica_evaluation_summary.json \
  --output-csv official_metrics_fold0.csv \
  --vol-threshold 27 \
  --overlap-threshold 0.2
```

## 27. 完整检测点矩阵

| 层 | 检测点 | 怎么检测 | 通过标准 | 失败动作 |
|---|---|---|---|---|
| L0 | run 目录存在 | `g2_synthetic_raw_intake_qc.py` 检查路径 | 目录可读 | 不接收 |
| L0 | raw case 可发现 | 扫描 `*.nii.gz` 和 legacy/native suffix | 至少 1 个 case | 不生成 run 产物 |
| L0 | run_id | 读 config 或目录名 | 唯一非空 | 补目录名，报告标注 |
| L0 | checkpoint | 读 config/log/manifest | 可追踪 | 不能 accepted |
| L0 | seed | 读 config/log/manifest | 可追踪 | 不能 accepted |
| L0 | label_channels | 读 config/log/manifest | 2026 推荐 4 | 3 channel 只能 ablation/review |
| L0 | rc_policy | 读 config/log/manifest | 明确 RC 处理方式 | RC case 进入 review |
| L0 | completion source 归属 | 查 source_case 是否在 real train/validation manifest，且是否 final QC pass | 可追溯 | 不可追溯 hard reject |
| L1 | 四模态完整 | 查 `t1n/t1c/t2w/t2f` 或 `scan_t1/scan_t1ce/scan_t2/scan_flair` | 全部存在 | hard reject |
| L1 | seg 完整 | 查 `seg.nii.gz` | 存在 | hard reject |
| L1 | suffix scheme | 识别 native/legacy/mixed | native 或 legacy | mixed hard reject |
| L1 | 文件命名 | raw case 文件应可追踪到目录名 | 一致或 manifest 可解释 | review |
| L2 | NIfTI 可读 | nibabel load | 全部可读 | hard reject |
| L2 | 维度 | header/data shape | 3D volume | hard reject |
| L2 | 病例内 shape | 五个文件 shape | 完全一致 | hard reject |
| L2 | 病例内 spacing | header zooms | 容差内一致 | hard reject |
| L2 | 病例内 affine | affine hash/allclose | 一致 | hard reject |
| L2 | orientation | aff2axcodes | 一致 | hard reject/review |
| L3 | 图像有限值 | `np.isfinite` | 无 NaN/Inf | hard reject |
| L3 | 常数图 | min/max/std | 非常数 | hard reject |
| L3 | 强度统计 | min/p1/p50/p99/max/std | 记录并对 batch 检查 | 异常 review |
| L3 | label 整数 | `np.isclose(label, round(label))` | 全整数 | hard reject |
| L3 | label 值域 | unique labels | `{0,1,2,3,4}` 子集 | hard reject |
| L3 | 空 mask | `label>0` | 默认不允许 | hard reject，除非官方/source 明确允许 |
| L4 | source 可追踪 | 从 raw name/log/manifest 找 source | 存在 source_case_id | hard reject |
| L4 | source 在 train manifest | 查 `real_train_manifest.csv` | 存在 | hard reject |
| L4 | source final QC | 查 `final_qc_pass` | true | hard reject/review |
| L4 | fixed real val 泄漏 | 查 `splits_final_fold0_realval.json` | 不在 val | hard reject |
| L4 | official validation 泄漏 | 查 `real_validation_manifest.csv` | 不在 validation | hard reject |
| L4 | source 可用于 G1 MET 96 ROI | 查 `g1_met_source_cases_v1.csv` | true | hard reject/review |
| L5 | ROI bbox | 从 label 或 manifest/log 恢复 | 可恢复 | review |
| L5 | ROI 在图像内 | bbox 与 shape 对比 | 在范围内 | hard reject |
| L5 | lesion 在脑内 | mask 不贴边、脑 mask proxy | 合理 | review/hard reject |
| L5 | 非 ROI 改动 | source/synth diff outside ROI | 低于阈值 | review/hard reject |
| L5 | ROI 边界跳变 | boundary shell diff/gradient | 无明显方块 | review/hard reject |
| L5 | source lesion overlap | source seg 与 synth seg | 有说明且合理 | review |
| L6 | component count | connected components | 与真实分布相容 | review |
| L6 | tiny/small/large | `volume=voxels*spacing` | tiny 不失控 | review |
| L6 | ET/RC/TC/WT region | 按官方 mapping 生成 | region 合法 | review |
| L6 | RC 语境 | source has RC 或 config 声明 | 合理 | review/hard reject |
| L7 | ET 与 T1C | ET mask vs t1c shell contrast | 有增强语境 | review |
| L7 | SNFH 与 T2F/T2W | SNFH mask vs t2f/t2w contrast | 高信号合理 | review |
| L7 | 多模态 ROI corr | ROI 内跨模态相关 | 无模式塌缩 | review |
| L8 | z 连续性 | slice area/intensity smoothness | 无断裂 | review |
| L8 | block artifact | lesion bbox fill ratio/边界 | 无方块伪影 | reject/review |
| L8 | duplicate/similarity | hash、ROI SSIM/MS-SSIM、feature distance | 不疑似复制 | review/reject |
| L9 | teacher model | real-only teacher 预测 vs synthetic seg | 不明显冲突 | review |
| L10 | batch 分布 | accepted vs real stats | 受控 | 降级/限制比例 |
| L11 | nnU-Net 命名 | materialization manifest | imagesTr/labelsTr 可生成 | 不放行训练 |
| L11 | channel order | dataset.json 与 S1/S2 一致 | 明确固定 | 不放行训练 |
| L11 | integrity | nnUNet verify | pass | 不放行训练 |
| L12 | official metrics | Panoptica + parser | real+synth 不损害核心字段 | 不进入主方案 |

## 28. S1/S2 对齐要求

1. G2 默认 nnU-Net 通道顺序是 `0000=t1n, 0001=t1c, 0002=t2w, 0003=t2f`。
2. S2 当前脚本顺序是 `0000=t1c, 0001=t1n, 0002=t2f, 0003=t2w`。
3. S1 当前自定义 loader 顺序也是 `t1c,t1n,t2f,t2w`，并把 `seg` 拆成 tumor/RC 双头。
4. 团队训练前必须固定通道顺序，real-only 与 real+synth 不能换顺序。
5. 如果沿用 S1/S2 当前顺序，G2 物化脚本必须使用 `--channel-order s2_current`，并在报告中写明。
6. 如果切回 G2 官方顺序，S1/S2 代码和 dataset.json 必须同步改。

## 29. accepted/rejected 的最终定义

| 输出 | 进入训练 | 进入消融 | 回传 G1 | 说明 |
|---|---|---|---|---|
| `accepted_for_training` | 是 | 是 | 不需要 | 可进入主 real+synth 候选 |
| `accepted_for_ablation_only` | 否 | 是 | 可选 | 质量通过，但 source 不进主训练，只用于受控实验 |
| `needs_regeneration` | 否 | 否 | 是 | G1 可根据原因重跑 |
| `rejected` | 否 | 否 | 是 | 泄漏、非法、不可追溯或严重质量错误 |

每批必须输出：

1. `synthetic_candidate_manifest_{run_id}.csv`
2. `synthetic_accepted_manifest_{run_id}.csv`
3. `synthetic_rejected_manifest_{run_id}.csv`
4. `qc_metrics_{run_id}.csv`
5. `diffusion_quality_metrics_{run_id}.csv`
6. `qc_batch_summary_{run_id}.json`
7. `G2_synthetic_data_quality_report_{run_id}.md`

## 30. G2 任务完成定义

G2 只有同时满足以下条件，才算完成本阶段任务：

1. 已能接收 G1 raw output。
2. 已能识别 legacy suffix 并生成 2026/nnU-Net 映射。
3. 已能自动补建 synthetic manifest。
4. 已能自动生成 QC 结果。
5. 已能输出 accepted/rejected。
6. 已能给 S1/S2 生成 nnU-Net raw 物化入口。
7. 已能用官方同款字段评价 real-only vs real+synth。
8. 已能解释每个检查点怎么做、为什么做、失败后怎么办。

当前仓库已经补齐对应脚本和文档；等 G1 提供真实 run 后，G2 可直接执行本文件第 26 节命令。
