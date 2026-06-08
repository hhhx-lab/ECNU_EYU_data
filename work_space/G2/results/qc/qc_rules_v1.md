# G2 Synthetic QC Rules v1

## 强制拒绝

1. 缺少任一模态或 `seg.nii.gz`。
2. 四模态与 `seg` 的 shape、spacing 或 affine 明显不一致。
3. 图像含 NaN/Inf，或 label 非整数。
4. label 值域不在 `{0,1,2,3,4}`。
5. label 全空且 manifest 未显式允许空 mask。
6. synthetic lesion 出现在脑外全零背景区。
7. 插入 ROI 边界出现明显方块断层。
8. 2D/slice-stitching 结果在 z 轴严重断裂。
9. 缺少 `synthetic_generation_manifest.csv` 或 `generation_log.jsonl`，且无法补建。


## 需要人工复查

1. tiny lesion 数量异常高。
2. ET/NETC 与 SNFH 空间关系不合理。
3. RC 在非术后语境下大量出现。
4. teacher model 与 synthetic label 差异极大。

## 只记录不拒绝

1. FID、MS-SSIM 等生成质量指标。
2. teacher model Dice。
3. lesion-wise count difference。

## lesion 分档

1. `tiny_lt_27mm3`
2. `small_27_to_275mm3`
3. `large_gt_275mm3`

最终是否采用 synthetic data，以真实验证 fold 上的分割和检测消融结果为准。
