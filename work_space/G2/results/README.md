# G2 Results

更新日期：2026-07-12

本目录只保存轻量 CSV/JSON/Markdown 契约和模板，不保存正式 NIfTI、nnU-Net 预处理缓存、模型权重或临时 smoke 输出。

## 当前数据层

| 数据层 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| master | 1035 | 130 | 130 | 1295 |
| real-only authentic T2W | 823 | 103 | 104 | 1030 |
| V3 completion 目标 | 212 | 27 | 26 | 265 |

## 目录

| 目录 | 用途 |
|---|---|
| `manifests/` | 真实身份、V2 source、G1 run 和 release 清单 |
| `splits/` | patient-group master/real-only fixed split |
| `qc/` | QC 策略、逐例模板、官方指标模板与历史真实数据审计 |
| `stats/` | 真实标签和病灶分布参考 |
| `nnunet_raw/` | 轻量 dataset contract，不保存正式影像 |
| `reports/` | 进度、路径、QC 和消融报告 |

## 正式入口

1. `../code/g2_build_realonly_from_raw.py`：刷新 master、real-only、V2 source 和 patient-group split。
2. `../code/g2_v2_compose_augmentation.py`：把 V2 平铺输出恢复成完整可审计病例。
3. `../code/g2_v3_completion_intake.py`：接收 V3 completion。
4. `../code/g2_synthetic_raw_intake_qc.py`：生成 rejected/pending/accepted 和 QC 报告。
5. `../code/g2_materialize_nnunet_dataset.py`：发布双视图、fixed split 和 integrity report。
6. `../code/g2_official_mets_metrics_parser.py`：解析已有官方 evaluation JSON 或校验 leaderboard CSV 字段。

locked test 在物化时进入 `imagesTs/labelsTs`，train/val 进入 `imagesTr/labelsTr`；case-folder view 同样按 `train/val/test` 分目录。
