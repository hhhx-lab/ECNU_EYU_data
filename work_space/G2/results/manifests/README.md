# Manifests

更新日期：2026-07-12

## 固定输入

| 文件 | 用途 |
|---|---|
| `real_train_manifest_raw.csv` | 历史原始训练数据扫描证据 |
| `real_train_manifest.csv` | 应用 corrected label 后的真实病例审计表 |
| `real_validation_manifest.csv` | 官方 validation 结构记录，不作 V2 source |
| `corrected_label_overlay.csv` | corrected seg 覆盖证据 |
| `nnunet_case_mapping_master.csv` | 1295 例 master 身份，包括 completion 目标 |
| `nnunet_case_mapping_realonly.csv` | 1030 例 authentic-T2W 子集 |
| `g1_v2_source_manifest.csv` | V2 source 主表，只有 823 行 allowed |

master mapping 保留 fake/broken T2W 病例，是为了让 V3 completion 修复后恢复到原病例身份；real-only mapping 明确排除这些病例。

## 正式 run 输出

通用 intake 按 run ID 生成：

```text
synthetic_generation_manifest_<run_id>.csv
synthetic_candidate_manifest_<run_id>.csv
synthetic_pending_review_manifest_<run_id>.csv
synthetic_accepted_manifest_<run_id>.csv
synthetic_accepted_evaluation_manifest_<run_id>.csv
synthetic_rejected_manifest_<run_id>.csv
synthetic_normalized_mapping_<run_id>.csv
```

`synthetic_accepted_manifest` 只含训练角色数据；`synthetic_accepted_evaluation_manifest` 只含 V3 val/test completion。

## 模板

1. `synthetic_generation_manifest_template_g1.csv`
2. `synthetic_normalized_mapping_template.csv`

模板只定义字段，不代表已有真实 G1 批次通过。
