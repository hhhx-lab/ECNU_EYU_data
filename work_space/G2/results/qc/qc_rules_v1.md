# G2 Synthetic QC Rules v2

本文件是 G2 检查 G1/diffusion synthetic data 的执行摘要。完整报告口径、检查步骤和字段说明见 `G2_synthetic_data_QC规则策略_v2.md`。

## 0. 基础依据

1. 2026 Task1 使用四模态 MRI：`t1n`、`t1c`、`t2w`、`t2f`。
2. 2026 Task1 label 值域必须是 `{0,1,2,3,4}`：`0=background`、`1=NETC`、`2=SNFH`、`3=ET`、`4=RC`。
3. 小于 `27 mm3` 的 tiny lesion 对检测有临床意义；大于 `275 mm3` 的 lesion 是分割指标重点。
4. 真实数据是混合空间：不同病例可以有不同 shape/spacing；但同一病例内四模态和 seg 必须几何一致。
5. 往年 nnU-Net 流程要求固定通道顺序、严格命名、`dataset.json` 一致，并且 synthetic 不得污染真实 validation fold。

## 1. 单例强制拒绝

只要一个 synthetic case 命中任一条，`qc_pass=false`，不能进入训练集。

1. 缺少 `t1n/t1c/t2w/t2f/seg` 任一文件。
2. 文件不是可读的 `.nii.gz`，或 nibabel/SimpleITK 无法读取 header 和数组。
3. 四模态与 `seg` 在同一病例内 shape 不一致。
4. 四模态与 `seg` 在同一病例内 spacing 不一致；容差为绝对误差 `<=1e-5`。
5. 四模态与 `seg` affine 不一致；容差为 `np.allclose(affine, atol=1e-4)`。
6. 图像数组存在 NaN、Inf，或全体素恒定为同一个值。
7. `seg` 不是整数标签，或存在非整数浮点值。
8. `seg` 值域超出 `{0,1,2,3,4}`。
9. `seg` 全空，且 metadata 没有显式 `allow_empty_mask=true`；第一轮 synthetic 默认不接受空 mask。
10. `case_id` 复用真实病例 ID，或 synthetic ID 与目录/文件名不一致。
11. `source_case_id` 缺失、无法在 `real_train_manifest.csv` 中找到，或 source case 的 `final_qc_pass` 不是 `True`。
12. `source_case_id` 属于固定验证 fold，或来自官方 validation 数据。
13. `metadata.json` 缺失，且无法从 run-level manifest 和日志恢复核心字段。
14. 缺少 `generation_run_id`、`generator_checkpoint`、`seed`、`generation_mode` 中任一关键追溯字段。
15. 生成 lesion 的 bbox 超出图像边界。
16. synthetic lesion 与脑外全零背景明显重叠，且不是原病例已有合法非零区域。
17. local insertion 结果在 ROI 边界出现明显硬边、方块断层或强度突变。
18. 2D/2.5D/slice-stitching 结果在 z 轴出现严重断裂。
19. RC label `4` 凭空出现在非 RC source case，且 metadata 未说明是术后 RC 变体。
20. 该病例无法通过 `nnUNetv2_plan_and_preprocess --verify_dataset_integrity` 对应的数据完整性检查。

## 2. 单例人工复查

命中以下任一条时，不能自动 accepted；需要 G2 人工看图或复核统计。

1. lesion 数量超过真实训练集 95 分位参考值，即 `>28` 个 lesion。
2. lesion 数量超过真实训练集 99 分位参考值，即 `>56` 个 lesion。
3. tiny lesion 数量异常高，或 tiny lesion 占本例 lesion 比例 `>0.35`。
4. 最小 lesion 体积 `<1 mm3`。
5. 最大 lesion 体积超过真实 lesion 99 分位参考值约 `15213 mm3`，或明显不符合 source case 语境。
6. `label_combination` 极罕见，例如 `SNFH` only、`NETC+ET+RC`、`ET+RC`，需要确认是否医学上合理。
7. ET/NETC 没有被 SNFH 或合理上下文包绕，表现为孤立不合理结构。
8. RC 与 ET/SNFH/NETC 的空间关系不合理，或 RC 形态不像术后 cavity。
9. teacher model 与 synthetic label 的 Dice 极低，或 lesion count 差异很大。
10. 图像强度分布与 source case 差异极大，疑似生成失败或归一化错误。
11. 视觉上可见棋盘格、条纹、层间闪烁、局部模糊、插入区域纹理与周围脑组织不连续。

## 3. 只记录不单独拒绝

以下指标必须记录，但单独一项不直接决定拒绝。

1. FID、MS-SSIM、LPIPS 或其他生成质量指标。
2. teacher model Dice、NSD、lesion-wise Dice。
3. lesion-wise count difference。
4. 每个 label 的体积、bbox、center、连通组件数。
5. 与真实分布的 KL/JS divergence 或分档比例差异。
6. 单病例生成耗时、GPU、checkpoint、seed。

## 4. lesion 分档

沿用 2026 Task1 和当前真实统计：

1. `tiny_lt_27mm3`：`volume_mm3 < 27`。
2. `small_27_to_275mm3`：`27 <= volume_mm3 <= 275`。
3. `large_gt_275mm3`：`volume_mm3 > 275`。

## 5. 批次放行标准

一个 synthetic run 不能只看单例。批次进入训练前必须同时满足：

1. `hard_reject_rate <= 5%`；smoke run 可以放宽到只要所有失败原因可解释。
2. `manual_review_required_rate <= 20%`，且所有人工复查病例有结论。
3. 通过 QC 的病例全部有完整 metadata、generation manifest 和日志。
4. accepted synthetic 不超过真实训练病例数的 25%，第一轮上限约 `323` 例。
5. validation source 泄漏数量必须为 `0`。
6. 同一个 source case 默认最多 accepted 1 个 synthetic case；例外必须在报告中列出。
7. tiny lesion 在 accepted synthetic 中的比例不超过 `35%`，除非该批明确是 tiny lesion 专项 stress test。
8. RC synthetic 只能来自真实 RC source case，第一轮不做凭空 RC。
9. accepted synthetic 加入训练后，固定真实验证 fold 上 real+synth 至少不劣于 real-only；最终采用以消融结果为准。

## 6. 报告结论口径

每批 synthetic data 的结论只能写以下四类之一：

1. `accepted_for_training`：通过硬检查，人工复查已完成，允许进入 real+synth 训练。
2. `accepted_for_ablation_only`：质量基本可用，但只能进入受控消融，不进入主训练池。
3. `needs_regeneration`：存在可修复生成问题，需要 G1 重跑或补交 metadata。
4. `rejected`：存在不可接受的数据质量或泄漏问题，整批不使用。
