# G2 Real Data QC Summary

生成日期：2026-05-31

## 总览

1. 训练病例 manifest 行数：1296。
2. validation 病例 manifest 行数：179。
3. corrected labels 文件数：2。
4. corrected overlay 后 final QC pass：1295。
5. corrected overlay 后 final QC fail：1。
6. affine hash warning 病例数：860。这类病例 shape/spacing 一致，但模态或 label header affine hash 不完全一致，第一轮记录为 warning，不直接排除。

## corrected labels

| case_id | raw_seg_path | corrected_seg_path | raw_unique_labels | corrected_unique_labels | raw_shape | corrected_shape | raw_spacing | corrected_spacing | raw_affine_hash | corrected_affine_hash | applied | apply_reason | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BraTS-MET-01094-003 | /Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据/MICCAI-LH-BraTS2025-MET-Challenge-Training/UCSD - Training/BraTS-MET-01094-003/BraTS-MET-01094-003-seg.nii.gz | /Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels/BraTS-MET-01094-003-seg.nii.gz | 0;3 | 0;3 | 256x256x132 | 256x256x132 | 1,1,1.4 | 1,1,1.4 | 8c99565e815dadd7 | 8c99565e815dadd7 | True | shape_match |  |
| BraTS-MET-01184-002 | /Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据/MICCAI-LH-BraTS2025-MET-Challenge-Training/UCSD - Training/BraTS-MET-01184-002/BraTS-MET-01184-002-seg.nii.gz | /Users/hwaigc/比赛+课题/ECNU-NYU2026/2026的task1以及数据/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels/BraTS-MET-01184-002-seg.nii.gz | 0;1;2;3;4;8 | 0;1;2;3;4 | 249x512x512 | 249x512x512 | 1.03,0.5,0.5 | 1.03,0.5,0.5 | 162960730ac02928 | 162960730ac02928 | True | shape_match |  |

## overlay 后非法标签病例

| case_id | illegal_label_values_after_overlay | final_qc_reason |
| --- | --- | --- |
| BraTS-MET-01094-002 | 6.0 | illegal_label_values_after_overlay:[6] |

## 说明

1. 本轮未复制 NIfTI 数据，仅记录原始路径和有效 label 路径。
2. 图像全体素 NaN/Inf 检查因本地训练与验证数据约 36GB，暂不在 Mac 上全量读取；当前已完成 NIfTI header、shape、spacing、affine hash 与 label 值域检查。
3. affine hash 不一致当前作为 warning；正式训练前由 nnU-Net integrity check 和必要的 header/方向一致性复核兜底。
4. `BraTS-MET-01184-002` 使用 corrected label 后不再保留非法值 8。
5. `BraTS-MET-01094-002` 当前仍含非法值 6，第一轮训练与生成 source 中排除。
