# G2 Synthetic Data QC 规则策略 v2

生成日期：2026-06-08
适用对象：G2 数据报告、G1 diffusion/GliGAN-compatible synthetic output、S1/S2 real+synth nnU-Net 训练入口
主结论：G2 的 QC 目标不是证明生成图像“看起来好”，而是证明 synthetic data 在文件、空间、标签、病灶、来源、分布、可复现性和训练可用性上足够可靠，不会污染真实验证 fold，不会把生成错误伪装成数据增强。

## 1. QC 依据

### 1.1 2026 Task1 任务约束

BraTS 2026 Task1 是脑转移瘤检测与分割任务，关注治疗前和治疗后病例。任务要求模型自动检测并分割不同大小的脑转移瘤、周围水肿和切除腔。

| 项目 | 2026 要求 | 对 G2 QC 的影响 |
|---|---|---|
| 模态 | `t1n/t1c/t2w/t2f` 四模态 MRI | synthetic case 必须四模态齐全 |
| 标签 | `{0,1,2,3,4}` | 任何非法 label 都强制拒绝 |
| 标签 1 | NETC | 需要检查与 ET/SNFH 的空间关系是否合理 |
| 标签 2 | SNFH | 需要检查是否形成合理周围 FLAIR 高信号 |
| 标签 3 | ET | 需要重点检查 t1c 上是否有合理增强对应 |
| 标签 4 | RC | 只应出现在治疗后/切除腔语境，第一轮不凭空生成 |
| tiny lesion | `<27 mm3` 有临床相关性 | 不能简单删除 tiny lesion，但要防止生成过量假阳性 |
| 分割重点 | `>275 mm3` lesion 用于分割评估 | large lesion 的 label 和边界必须尤其稳 |
| 数据空间 | 原生空间和 SRI24 空间混合 | 不能要求所有病例固定 shape；只能要求病例内部几何一致 |
| 真实验证 | 官方 validation 无公开 label | 不能作为 synthetic source，不能进入训练 |

### 1.2 当前本地数据事实

当前 G2 已完成真实数据扫描：

| 项目 | 数值 |
|---|---:|
| 本地带标签训练病例 | 1296 |
| final QC pass 训练病例 | 1295 |
| final QC fail 训练病例 | 1 |
| 官方 validation 病例 | 179 |
| real-only 固定 fold train/val | 1036 / 259 |
| 真实 lesion component 总数 | 9793 |
| tiny/small/large lesion 数 | 3788 / 3922 / 2083 |
| 含 RC 的真实训练病例 | 167 |

已知必须排除的真实病例：

| 病例 | 原因 |
|---|---|
| `BraTS-MET-01094-002` | corrected overlay 后仍含非法 label value `6` |

已知 corrected label 处理：

| 病例 | 处理 |
|---|---|
| `BraTS-MET-01094-003` | 使用官方 corrected label |
| `BraTS-MET-01184-002` | 使用官方 corrected label 修正原始非法 label `8` |

### 1.3 往年方法给今年的 QC 启发

| 来源 | 可复用经验 | 今年调整 |
|---|---|---|
| 2023 GliGAN/rGANs | synthetic data 可以显著扩充训练，但必须保留 source、seed、生成方式和 split 控制 | 今年 diffusion 也必须沿用 manifest/log 追溯 |
| 2023 split 说明 | validation set 只含真实数据，且不能用 validation source 生成 synthetic 放回训练 | 今年固定 `splits_final_fold0_realval.json`，synthetic 只进 train 侧 |
| 2023 nnU-Net | 训练前运行 `nnUNetv2_plan_and_preprocess --verify_dataset_integrity` | 今年 real+synth 数据集物化后必须跑同等检查 |
| nnU-Net 数据格式 | 每例所有通道与 label 必须同几何；通道顺序固定；seg 是整数 map | 写入 hard reject |
| BraTS 2025 工具链 | 输入需要标准命名和 sanity check；默认 shape 警告不能代替拒绝 | 今年 MET 混合空间，shape 非 `240x240x155` 只记录，不拒绝 |
| 2026 自定义 loss | lesion-wise Dice/NSD、connected component 对任务很重要 | QC 必须输出 lesion-level 统计，不只看病例级 label 体积 |

## 2. QC 总流程

每一批 synthetic data 都按以下顺序检查。前一层失败，不进入后一层训练评估。

| 层级 | 名称 | 目的 | 输出 |
|---:|---|---|---|
| L0 | 交付完整性检查 | 确认 G1 交付可追溯、可复现 | run manifest、log、metadata 完整性 |
| L1 | 文件与命名检查 | 确认每例四模态和 seg 齐全 | file check table |
| L2 | NIfTI header 与几何检查 | 确认同一病例内 shape/spacing/affine 一致 | geometry metrics |
| L3 | 数组与标签合法性检查 | 确认无 NaN/Inf、label 值域正确 | value metrics |
| L4 | source 与 split 泄漏检查 | 确认只从允许训练病例生成 | leakage report |
| L5 | lesion-level 结构检查 | 确认 lesion 数、体积、bbox、label 组合合理 | lesion metrics |
| L6 | 生成痕迹与图像质量检查 | 查 ROI 边界、z 轴断裂、强度异常 | artifact metrics + review list |
| L7 | teacher/segmenter 一致性检查 | 用分割模型粗查 label 是否离谱 | teacher metrics |
| L8 | 批次分布检查 | 确认 synthetic 分布没有偏离真实数据太多 | batch distribution report |
| L9 | nnU-Net 物化检查 | 确认 accepted 数据能进入训练 | verify_dataset_integrity 结果 |
| L10 | 消融放行检查 | 用真实验证 fold 判断是否采用 | real-only vs real+synth 对比 |

## 3. L0 交付完整性检查

### 3.1 G1 每批必须交付

```text
synthetic_raw/
  run_YYYYMMDD_HHMM_modelname/
    synthetic_generation_manifest.csv
    generation_log.jsonl
    run_metadata.json
    SYN-MET-000001/
      SYN-MET-000001-t1n.nii.gz
      SYN-MET-000001-t1c.nii.gz
      SYN-MET-000001-t2w.nii.gz
      SYN-MET-000001-t2f.nii.gz
      SYN-MET-000001-seg.nii.gz
      metadata.json
```

### 3.2 必查字段

| 字段 | 必须性 | 说明 |
|---|---|---|
| `case_id` | 必需 | synthetic ID，不得复用真实 BraTS ID |
| `source_case_id` | 必需 | 真实 source 病例 |
| `generation_run_id` | 必需 | 本批次唯一 ID |
| `generator_name` | 必需 | 生成器名称 |
| `generator_checkpoint` | 必需 | checkpoint 路径或版本 |
| `generator_git_commit` | 建议 | 生成代码版本 |
| `generation_mode` | 必需 | local_insertion、full_case、2.5D 等 |
| `seed` | 必需 | 可复现随机种子 |
| `source_manifest_version` | 必需 | 使用哪个 G2 manifest |
| `label_strategy` | 必需 | label 如何生成或修改 |
| `roi_bbox_ijk` | local insertion 必需 | 插入区域 bbox |
| `allow_empty_mask` | 条件必需 | 空 mask 必须显式允许 |

### 3.3 判定

| 情况 | 判定 |
|---|---|
| 缺 `metadata.json`，但 run manifest 能完整恢复 | 人工复查 |
| 缺 `metadata.json` 且 run manifest 也不能恢复 | 强制拒绝 |
| 缺 seed/checkpoint/run_id/source_case_id | 强制拒绝 |
| synthetic ID 与目录/文件名不一致 | 强制拒绝 |

## 4. L1 文件与命名检查

每个 synthetic case 必须有 5 个 NIfTI：

| 文件 | 含义 |
|---|---|
| `{case_id}-t1n.nii.gz` | pre-contrast T1 |
| `{case_id}-t1c.nii.gz` | post-contrast T1 |
| `{case_id}-t2w.nii.gz` | T2 |
| `{case_id}-t2f.nii.gz` | T2 FLAIR |
| `{case_id}-seg.nii.gz` | segmentation label |

检查标准：

1. 文件名必须与目录名 `case_id` 一致。
2. 后缀必须是 `.nii.gz`。
3. 不允许有重复模态。
4. 不允许缺少任一模态。
5. 不允许用真实病例 ID 作为 synthetic case ID。

判定：

| 问题 | 判定 |
|---|---|
| 缺任一模态或 seg | 强制拒绝 |
| 文件损坏无法读取 | 强制拒绝 |
| 额外文件不影响读取但命名混乱 | 人工复查 |

## 5. L2 NIfTI header 与几何检查

### 5.1 同一病例内必须一致

对 `t1n/t1c/t2w/t2f/seg` 读取：

1. `shape`
2. `spacing`
3. `affine`
4. `orientation`
5. `dtype`

强制标准：

| 项目 | 标准 |
|---|---|
| shape | 五个文件完全一致 |
| spacing | 五个文件逐轴一致，容差 `<=1e-5` |
| affine | 五个文件 `np.allclose(atol=1e-4)` |
| orientation | 五个文件一致 |
| header 可读性 | nibabel/SimpleITK 能读 header 和数组 |

### 5.2 跨病例不要求统一 shape

今年 MET 数据包含原生空间和 SRI24 空间，真实病例 shape/spacing 本来就不统一。因此：

1. shape 不是 `240x240x155` 不拒绝。
2. spacing 不是 `1,1,1` 不拒绝。
3. 但 synthetic case 必须继承 source case 的空间，或在 metadata 中明确说明目标空间和 resampling 方法。

判定：

| 问题 | 判定 |
|---|---|
| 同一病例内 shape/spacing/affine 不一致 | 强制拒绝 |
| 与 source case 空间差异但无 metadata 说明 | 强制拒绝 |
| 与 source case 空间差异且有清晰 resampling 说明 | 人工复查 |

## 6. L3 数组与标签合法性检查

### 6.1 图像数组检查

每个模态检查：

| 项目 | 标准 |
|---|---|
| NaN/Inf | 不允许 |
| 全零或常数图 | 不允许 |
| dtype | 记录，不强制统一 |
| 强度范围 | 记录 min/p1/p50/p99/max |
| 非零体素比例 | 与 source case 差异过大则人工复查 |

强制拒绝：

1. 出现 NaN/Inf。
2. 图像全零或常数。
3. 数组维度不是 3D。
4. 数组 shape 与 header 不一致。

### 6.2 label 检查

每个 `seg` 检查：

| 项目 | 标准 |
|---|---|
| 整数性 | 所有非空体素必须是整数 |
| 值域 | 只能是 `{0,1,2,3,4}` |
| 空 mask | 第一轮默认不接受，除非 metadata 显式允许 |
| label dtype | 可为 float/int，但值必须整数 |

强制拒绝：

1. label 值域包含 `5/6/8/255` 等非法值。
2. label 存在非整数浮点值，例如 `1.5`。
3. label 全空且未允许。

## 7. L4 source 与 split 泄漏检查

### 7.1 source 允许条件

synthetic source 必须满足：

1. 存在于 `work_space/G2/results/manifests/real_train_manifest.csv`。
2. `final_qc_pass=True`。
3. 不在 `splits/splits_final_fold0_realval.json` 的 `val` 列表中。
4. 不来自官方 `Validation/` 目录。
5. 不使用已排除病例 `BraTS-MET-01094-002`。

### 7.2 泄漏定义

以下都算泄漏：

1. 用固定真实验证 fold 的病例生成 synthetic，然后放入训练。
2. 用官方 validation 病例生成 synthetic。
3. synthetic ID 复用真实 ID，导致训练/验证无法区分。
4. metadata 中 source 缺失，无法排除泄漏。

判定：

| 问题 | 判定 |
|---|---|
| source 来自固定 val fold | 强制拒绝 |
| source 来自官方 validation | 强制拒绝 |
| source 不在 G2 manifest | 强制拒绝 |
| source 缺失无法追溯 | 强制拒绝 |

## 8. L5 lesion-level 结构检查

### 8.1 连通组件定义

QC 时把所有非背景标签 `{1,2,3,4}` 的并集作为 lesion mask，做 3D connected component 分析。每个 component 记录：

1. `lesion_id`
2. `component_labels`
3. `component_volume_mm3`
4. `volume_bucket`
5. `bbox_ijk`
6. `center_ijk`
7. 是否含 NETC/SNFH/ET/RC

### 8.2 体积分档

| 分档 | 标准 | 说明 |
|---|---|---|
| tiny | `<27 mm3` | 检测任务关心，不能简单丢弃 |
| small | `27-275 mm3` | 检测和分割之间的过渡 |
| large | `>275 mm3` | 分割指标重点 |

### 8.3 真实分布参考

| 指标 | 当前真实训练参考 |
|---|---:|
| lesion component 总数 | 9793 |
| tiny/small/large | 3788 / 3922 / 2083 |
| 每例 lesion 数 p50/p90/p95/p99/max | 3 / 16 / 28 / 56 / 393 |
| component volume p50/p95/p99/max | 44.9 / 3983.8 / 15212.6 / 96166.5 mm3 |

### 8.4 单例判定

| 情况 | 判定 |
|---|---|
| lesion bbox 超出图像边界 | 强制拒绝 |
| lesion 与脑外全零背景大面积重叠 | 强制拒绝 |
| lesion 数 `>28` | 人工复查 |
| lesion 数 `>56` | 高优先级人工复查 |
| tiny lesion 占比 `>35%` | 人工复查 |
| 最小 lesion `<1 mm3` | 人工复查 |
| 最大 lesion `>15213 mm3` | 人工复查 |
| label 组合极罕见 | 人工复查 |

注意：不能因为 tiny lesion 小就自动拒绝。2026 Task1 明确 tiny lesion 有临床意义；G2 要拒绝的是明显生成噪点、脑外假阳性、无法追溯的异常 tiny lesion。

## 9. L6 图像质量与生成痕迹检查

### 9.1 自动检查指标

| 指标 | 检查方式 | 判定 |
|---|---|---|
| ROI boundary jump | 比较 bbox 内外边界强度差 | 明显硬边则人工复查或拒绝 |
| z continuity | 比较相邻 slice lesion 面积和强度变化 | 严重断裂强制拒绝 |
| brain mask overlap | 用四模态非零区域近似脑区 | lesion 大量在脑外则拒绝 |
| source-synthetic intensity drift | 比较 source 与 synthetic 的 p1/p50/p99 | 极端偏离人工复查 |
| modality consistency | lesion 在 t1c/t2f 上是否有合理表现 | 不合理人工复查 |

### 9.2 人工看图清单

每批至少抽查：

1. 所有 hard reject 临界病例。
2. 所有 manual review 病例。
3. accepted 病例随机抽查不少于 `max(10, accepted_count*10%)`。
4. 所有 RC synthetic。
5. tiny lesion 比例最高的前 20 例。
6. 最大 lesion 体积前 20 例。

看图时至少查看：

1. `t1c + ET` overlay。
2. `t2f + SNFH` overlay。
3. `t1n/t1c/t2w/t2f` 四模态同切片。
4. sagittal/coronal/axial 三方向。
5. ROI 边界前后若干 slice。

## 10. L7 teacher/segmenter 一致性检查

teacher model 不是最终真理，只是辅助发现离谱样本。建议使用 real-only baseline 或 S1/S2 的初版 nnU-Net。

记录指标：

| 指标 | 含义 |
|---|---|
| `teacher_dice_label_1..4` | 每个 label 的 Dice |
| `teacher_lesion_count_diff` | teacher lesion 数与 synthetic label lesion 数差 |
| `teacher_missing_large_lesion_count` | teacher 漏掉的 large lesion 数 |
| `teacher_extra_large_lesion_count` | teacher 多出的 large lesion 数 |
| `teacher_nsd_label_1..4` | 可选 NSD |

判定建议：

1. teacher 与 label 完全不一致时人工复查。
2. large lesion 被 teacher 完全漏掉时人工复查。
3. teacher 指标低不能单独强制拒绝，因为 synthetic 可能是困难样本；但必须进入报告。

## 11. L8 批次分布检查

单例通过不代表整批可用。整批 accepted synthetic 必须和真实分布保持受控差异。

### 11.1 必报分布

1. label combination 分布。
2. tiny/small/large lesion 分布。
3. 每例 lesion 数分布。
4. RC case 比例。
5. source case 使用次数分布。
6. source institution/space 分布，如果 manifest 可推断。
7. generation_mode 分布。
8. checkpoint/seed 覆盖情况。

### 11.2 批次阈值

| 项目 | 第一轮标准 |
|---|---|
| accepted synthetic 总数 | 不超过真实训练病例 25%，即约 323 例 |
| 每个 source case | 默认最多 accepted 1 例 |
| validation leakage | 必须为 0 |
| hard reject rate | 目标 `<=5%` |
| manual review unresolved | 必须为 0 |
| tiny lesion 占比 | 默认 `<=35%` |
| RC synthetic | 只能来自真实 RC source |
| 缺 metadata accepted 数 | 必须为 0 |

## 12. L9 nnU-Net 物化检查

accepted synthetic 合并入训练前，必须建立单独 real+synth dataset，不要污染 real-only Dataset260。

检查：

1. `imagesTr/{case_id}_0000.nii.gz` 对应 t1n。
2. `imagesTr/{case_id}_0001.nii.gz` 对应 t1c。
3. `imagesTr/{case_id}_0002.nii.gz` 对应 t2w。
4. `imagesTr/{case_id}_0003.nii.gz` 对应 t2f。
5. `labelsTr/{case_id}.nii.gz` 对应 seg。
6. `dataset.json` channel_names 与 label 定义正确。
7. `numTraining` 与实际病例数一致。
8. 运行 `nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity`。

只要 nnU-Net integrity check 不通过，该批不能进入训练。

## 13. L10 消融放行检查

QC 只能证明数据“可用”，不能证明“有用”。最终采用 synthetic data 必须看真实验证 fold。

最低消融：

| 实验 | 训练数据 | 验证 |
|---|---|---|
| real-only baseline | 真实 train fold | 固定真实 val fold |
| real+synth smoke | 真实 train fold + smoke accepted synthetic | 同一 val fold |
| real+synth low ratio | 真实 train fold + 少量 accepted synthetic | 同一 val fold |
| real+synth policy ablation | 不同 generation policy | 同一 val fold |

采用标准：

1. real+synth 不能显著降低 Dice、NSD、lesion F1。
2. tiny lesion 检测指标若提升，但 large lesion 分割明显下降，需要单独讨论，不自动采用。
3. RC 指标若受损，RC synthetic 必须回滚或单独降权。
4. 最终报告必须包含 real-only 对照。

## 14. QC 结果文件建议

每批 synthetic data 建议输出：

```text
qc/
  qc_metrics_{run_id}.csv
  qc_case_review_{run_id}.csv
  qc_batch_summary_{run_id}.json
  G2_synthetic_data_quality_report_{run_id}.md
  accepted_synthetic_manifest_{run_id}.csv
  rejected_synthetic_manifest_{run_id}.csv
```

## 15. 数据报告必须回答的问题

每份数据报告必须明确回答：

1. 这批 synthetic data 是谁生成的，用哪个 checkpoint 和 seed。
2. 生成了多少例，成功读取多少例。
3. 有多少例强制拒绝，拒绝原因是什么。
4. 有多少例需要人工复查，复查结论是什么。
5. 最终 accepted 多少例。
6. 是否存在 validation leakage，数量是否为 0。
7. label 值域是否全部合法。
8. shape/spacing/affine 是否病例内一致。
9. lesion 数、体积、tiny/small/large 分布是否合理。
10. RC 是否只来自真实 RC source。
11. 是否通过 nnU-Net integrity check。
12. 是否允许进入 real+synth 消融。

## 16. 最终结论模板

报告结论只能使用以下四类：

| 结论 | 使用条件 |
|---|---|
| `accepted_for_training` | 单例硬检查通过，人工复查完成，批次分布可控，nnU-Net integrity 通过 |
| `accepted_for_ablation_only` | 基本可用，但分布、teacher 或视觉质量仍需通过受控消融确认 |
| `needs_regeneration` | metadata、生成痕迹、局部质量或可复现性有问题，但可由 G1 重跑修复 |
| `rejected` | 存在泄漏、非法 label、几何错误、严重伪影或不可追溯问题 |

## 17. 当前第一轮建议

1. G1 先交 10-20 个 smoke cases。
2. G2 对 smoke cases 跑完整 L0-L9 QC。
3. smoke 通过后再生成 100-300 个候选。
4. 第一轮 accepted synthetic 不超过 323 例。
5. 不做 full-case 从零生成进入主训练；优先 local insertion 或 source-conditioned diffusion。
6. 不凭空生成 RC。
7. 不使用 official validation，也不使用固定真实 val fold source。
8. 所有 accepted synthetic 都必须能追溯到 source、checkpoint、seed、generation_mode 和 QC 版本。
