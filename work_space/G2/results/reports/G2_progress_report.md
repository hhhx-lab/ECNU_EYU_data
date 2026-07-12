# G2 当前进度与文件索引

更新日期：2026-07-12

## 1. 当前结论

1. G2 已从旧 GliGAN 单入口改为 V2 augmentation、V3 completion 双入口。
2. master split 已改为 patient-group-aware，同一患者不跨 train/val/test。
3. 缺 config、checkpoint、seed、manifest 或 log 会硬拒绝；每个病例还必须有唯一且一致的 manifest/log 成功记录。
4. V2 raw output 必须先 composer 恢复 source geometry 和非 ROI 内容。
5. V3 保留原病例 ID，只替换 T2W。
6. 技术通过但没有审批的病例保持 pending，不会自动进入训练。
7. materializer 输出 nnU-Net 与 case-folder 双视图，并物理隔离 locked test；同时按 master split 二次验证审批用途和 source 身份。
8. 当前测试只证明接口和小型 NIfTI fixture 正确；尚未完成真实 V2/V3 正式影像质量验收。

## 2. 当前数据口径

| 项目 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| master | 1035 | 130 | 130 | 1295 |
| authentic-T2W real-only | 823 | 103 | 104 | 1030 |
| fake/broken T2W completion | 212 | 27 | 26 | 265 |

V2 allowed source：823。master 与 real-only 各自的 train/val/test patient-group overlap 均为 0。

## 3. 已实现代码

| 文件 | 作用 |
|---|---|
| `code/g2_build_realonly_from_raw.py` | 扫描 raw data、优先 corrected seg、标记 fake T2W、生成 master/real-only/V2 source/split |
| `code/g2_create_train_val_test_split.py` | patient-group deterministic split，与 G1 V3 seed=42 真实 T2W 划分对齐 |
| `code/g2_v2_compose_augmentation.py` | V2 平铺输出分组、强度映射、平滑回填、geometry 恢复、seg 复制和 run 证据生成 |
| `code/g2_v3_completion_intake.py` | V3 metadata/delivery 硬校验并调用 completion intake |
| `code/g2_synthetic_raw_intake_qc.py` | 通用 run 入口 |
| `code/g2_pretraining_audit.py` | 逐例 QC、release 状态、manifest、报告和模板核心实现 |
| `code/g2_materialize_nnunet_dataset.py` | 多 manifest、completion 替换、augmentation 追加、双视图、fixed split 和 integrity |
| `code/g2_official_mets_metrics_parser.py` | 已有官方 evaluation JSON 解析和 leaderboard CSV 字段校验 |

## 4. QC 当前实现

已自动实现：

1. config 与逐病例 manifest/log、source、split、ID 和患者组规则。
2. 文件、NIfTI、shape/spacing/affine/orientation。
3. NaN/Inf、常数图和标签合法性。
4. source seg 逐体素保护。
5. V2 非 ROI 保护、边界 MAE、梯度不连续、强度漂移和 ROI SSIM。
6. V3 t1n/t1c/t2f 逐体素保护，不把 fake T2W 当参考真值。
7. 脑区覆盖、病灶数量/体积、z 连续性和多模态对齐。
8. rejected/pending/accepted-training/accepted-evaluation 分流。
9. V3 VAE/EncDec/BBDM checkpoint、`bbdm_s` 和 validation run 的独立留痕。

尚未自动完成：teacher 推理、MS-SSIM、medical FID/MMD、跨 run 近似重复检测和真实正式批次人工盲评。这些未完成项不会填伪值，也不会被当作自动通过条件。

## 5. 八个主区域索引

### 5.1 G2 根目录

| 文件 | 说明 |
|---|---|
| `README.md` | G2 总入口和当前唯一数据口径 |
| `task_assignment.md` | G2 原始职责 |

### 5.2 code

见第 3 节。`tests/` 使用临时小型 NIfTI 覆盖 split、V2 composer、V3 intake、metadata gate 和 materializer。

### 5.3 docs

| 文件 | 说明 |
|---|---|
| `2026-07-12_G2_V2_V3对接与QC整改.md` | 本轮问题、改动和验收记录 |
| `G1_G2_diffusion_output_contract.md` | 两条 G1 产线正式交付契约 |
| `G2_G1适配执行清单.md` | 操作者逐步命令和验收清单 |
| `G2_数据生成与质量控制实施方案.md` | G2 总体方案 |
| `G2_模型训练完成前可执行工作清单.md` | 正式生成批次到来前的工作 |

### 5.4 results/manifests

| 文件 | 说明 |
|---|---|
| `nnunet_case_mapping_master.csv` | 全部 1295 例身份 |
| `nnunet_case_mapping_realonly.csv` | 1030 例 authentic-T2W 子集 |
| `g1_v2_source_manifest.csv` | V2 source 与 allowed 状态 |
| `real_train_manifest*.csv` | 历史真实训练数据审计 |
| `real_validation_manifest.csv` | 官方 validation 结构记录 |
| `corrected_label_overlay.csv` | corrected seg 证据 |
| `synthetic_*_template*.csv` | run 输出字段模板 |

### 5.5 results/splits

| 文件 | 说明 |
|---|---|
| `splits_master_train_val_test.json` | master 1035/130/130 |
| `splits_master_train_val_test_membership.csv` | master 逐例 membership |
| `splits_final_train_val_test.json` | real-only 823/103/104 |
| `splits_final_train_val_test_membership.csv` | real-only 逐例 membership |

### 5.6 results/qc

主策略、官方指标策略、报告/CSV 模板及 fake T2W 官方审计证据。官方 leaderboard 模板和 fake T2W 清单必须保留。

### 5.7 results/stats

保留历史全量真实标签/病灶分布，用于制定 V2 生成目标。正式实验分母以当前 master/real-only membership 为准。

### 5.8 results/nnunet_raw 与 reports

`nnunet_raw/` 只保存 Dataset260 轻量契约；正式影像在服务器由 materializer 生成。`reports/` 保存真实数据审计、run 级质量报告、进度和成对消融结果。

## 6. 清理结果

本轮删除：

1. 旧 `g1_met_source_cases_v1.csv`。
2. 旧 `splits_final_fold0_realval.json`。
3. 旧 96-ROI source、case-level split、`s2_current` 和 ablation-only 放行口径。
4. Python cache、测试临时文件和调试产物。

保留：官方榜单字段模板、corrected label 证据、fake/broken T2W 清单和 gzip header audit。

本轮自动回归：G2 34 项、G1 V3 6 项、S2 8 项、S3 2 项、S4 2 项、S5 1 项全部通过。

## 7. 下一步

1. G1 V3 正式阶段 6 输出后运行 completion intake。
2. G1 V2 正式批量输出后先 composer，再通用 intake。
3. 完成 pending 病例人工/teacher 审批。
4. 物化 completion 与 real+synth 双视图并运行 nnU-Net integrity。
5. 在当前 master split 上重训 paired real-only baseline，随后执行成对消融。
6. 用官方 evaluation 产物计算并汇总 lesionwise DSC/NSD 和 small-instance F1。
