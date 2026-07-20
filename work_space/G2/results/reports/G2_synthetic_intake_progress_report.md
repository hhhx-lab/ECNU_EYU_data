# G2 Synthetic Intake 进度报告

- 生成日期：2026-07-18
- 项目根目录：`ECNU_EYU_data`

## synthetic smoke 验证

- smoke run_id：`run_3104668`
- 候选数：265
- accepted for training：212
- accepted for evaluation：53
- pending review：0
- needs regeneration：0
- rejected：0
- legacy suffix case：0
- native suffix case：265
- mixed suffix case：0

## 下一步

1. V3 阶段 6 完成后先运行 completion 专用 intake。
2. V2 正式批量输出必须先运行 composer，再进入通用 QC。
3. pending 病例完成 teacher/人工审批后才可物化。
4. 使用固定真实验证集完成 real-only 与 real+synth 消融。

## 本次生成的文件

- `manifests/synthetic_generation_manifest_run_3104668.csv`
- `manifests/synthetic_candidate_manifest_run_3104668.csv`
- `manifests/synthetic_accepted_manifest_run_3104668.csv`
- `manifests/synthetic_accepted_evaluation_manifest_run_3104668.csv`
- `manifests/synthetic_pending_review_manifest_run_3104668.csv`
- `manifests/synthetic_rejected_manifest_run_3104668.csv`
- `manifests/synthetic_normalized_mapping_run_3104668.csv`
- `qc/qc_metrics_run_3104668.csv`
- `qc/diffusion_quality_metrics_run_3104668.csv`
- `qc/qc_case_review_run_3104668.csv`
- `qc/qc_batch_summary_run_3104668.json`
- `reports/G2_synthetic_data_quality_report_run_3104668.md`
- `reports/G2_synthetic_intake_progress_report.md`

## Intake 索引

### synthetic_generation_manifest
- `manifests/synthetic_generation_manifest_run_3104668.csv`

### synthetic_candidate_manifest
- `manifests/synthetic_candidate_manifest_run_3104668.csv`

### synthetic_accepted_manifest
- `manifests/synthetic_accepted_manifest_run_3104668.csv`

### synthetic_accepted_evaluation_manifest
- `manifests/synthetic_accepted_evaluation_manifest_run_3104668.csv`

### synthetic_pending_review_manifest
- `manifests/synthetic_pending_review_manifest_run_3104668.csv`

### synthetic_rejected_manifest
- `manifests/synthetic_rejected_manifest_run_3104668.csv`

### synthetic_normalized_mapping
- `manifests/synthetic_normalized_mapping_run_3104668.csv`

### qc_metrics
- `qc/qc_metrics_run_3104668.csv`

### diffusion_quality_metrics
- `qc/diffusion_quality_metrics_run_3104668.csv`

### qc_case_review
- `qc/qc_case_review_run_3104668.csv`

### qc_batch_summary
- `qc/qc_batch_summary_run_3104668.json`

### quality_report
- `reports/G2_synthetic_data_quality_report_run_3104668.md`


## 根目录与入口文件

| 文件 | 说明 |
|---|---|
| `README.md` | G2 项目总入口说明，概述项目目的、目录分工和本仓库的轻量化数据策略。 |
| `task_assignment.md` | G2 团队分工总表，把成员职责和工作拆分在一个入口里。 |
| `data/.gitkeep` | data 目录占位文件，保留未来数据放置点。 |
| `results/.gitkeep` | results 根目录占位文件，保留结果区目录结构。 |
| `results/README.md` | results 总说明，概括本目录只保存轻量产物，不保存大体积 NIfTI。 |

## 八个主文件夹索引

### 1. code

| 文件 | 说明 |
|---|---|
| `code/.gitkeep` | code 目录占位文件，保证空目录被版本控制保留。 |
| `code/g2_pretraining_audit.py` | 基础审计脚本：真实数据基线扫描、模板刷新、source CSV、real-only mapping、可选 synthetic intake 与进度报告生成。 |
| `code/g2_create_train_val_test_split.py` | 患者分组 master split 脚本，真实 T2W 子集复现 G1 V3 seed=42 划分。 |
| `code/g2_synthetic_raw_intake_qc.py` | 通用 G1 run 接收脚本，输出 rejected、pending、accepted-training 和 accepted-evaluation。 |
| `code/g2_v2_compose_augmentation.py` | V2 composition 脚本：将平铺 ROI 输出回填 source 并恢复完整几何。 |
| `code/g2_v3_completion_intake.py` | V3 completion 专用入口：校验 checkpoint、seed、bbdm_s 和 source manifest。 |
| `code/g2_materialize_nnunet_dataset.py` | 双视图物化脚本：同时生成 nnU-Net 和病例目录、fixed split 与完整性报告。 |
| `code/g2_official_mets_metrics_parser.py` | 官方指标代理脚本：解析 BraTS_evaluation Panoptica JSON 或校验 CSV 是否包含 2026 Task1 leaderboard 字段。 |

### 2. docs

| 文件 | 说明 |
|---|---|
| `docs/G1_G2_diffusion_output_contract.md` | G1 raw output 与 G2 适配边界的主契约，定义 raw 命名、source CSV、manifest 字段和最低 smoke 标准。 |
| `docs/G2_G1适配执行清单.md` | 按执行顺序拆解 G2 先准备什么、G1 输出后 G2 做什么、如何形成 QC 结果与回传。 |
| `docs/G1_G2_服务器训练推理QC运行手册.md` | 待补充说明 |
| `docs/G2_数据生成与质量控制实施方案.md` | 总方案，解释 G2 为什么是 adapter/auditor/publisher，以及 raw intake 到 nnU-Net 导出的全链路。 |
| `docs/G2_模型训练完成前可执行工作清单.md` | 训练前能立即执行的工作清单，属于 G2 的下一步行动仓库。 |

### 3. results/manifests

| 文件 | 说明 |
|---|---|
| `results/manifests/README.md` | 清单区说明，解释真实清单、source CSV、synthetic intake manifest 与 accepted/rejected 输出。 |
| `results/manifests/corrected_label_overlay.csv` | 真实训练病例的 corrected label 覆盖记录，说明哪些病例在最终 manifest 中替换了原始 seg。 |
| `results/manifests/g1_v2_source_manifest.csv` | V2 source 主表；只有 master train 且真实 T2W 的病例允许生成。 |
| `results/manifests/nnunet_case_mapping_master.csv` | 全部 1295 个可追溯病例身份，包含 265 个 completion 目标。 |
| `results/manifests/nnunet_case_mapping_realonly.csv` | real-only nnU-Net 映射表，用于训练机物化 imagesTr/labelsTr。 |
| `results/manifests/real_train_manifest.csv` | 真实训练病例最终主表，已应用 corrected label overlay 并带 final_qc_pass。 |
| `results/manifests/real_train_manifest_raw.csv` | 原始训练病例扫描表，保留 raw seg 与基础 QC 证据。 |
| `results/manifests/real_validation_manifest.csv` | 官方 validation 路径与结构检查表，绝不作为 synthetic source。 |
| `results/manifests/synthetic_generation_manifest_template_g1.csv` | G1 raw output 或 G2 补建时使用的 synthetic manifest 表头模板。 |
| `results/manifests/synthetic_normalized_mapping_template.csv` | 逐模态标准化映射模板，定义 raw source、normalized target 与 nnU-Net target 的对应关系。 |

### 4. results/stats

| 文件 | 说明 |
|---|---|
| `results/stats/README.md` | 统计区说明，解释 label/lesion 分布与 synthetic 目标分布。 |
| `results/stats/real_label_distribution.csv` | 真实训练病例级 label 体素与体积分布。 |
| `results/stats/real_lesion_distribution.csv` | 真实 lesion component 级分布。 |
| `results/stats/real_lesion_distribution_summary.json` | 机器可读统计摘要。 |
| `results/stats/real_lesion_distribution_summary.md` | 人可读统计摘要。 |
| `results/stats/target_synthetic_distribution_v1.md` | 第一轮 synthetic 目标分布与生成限制。 |

### 5. results/qc

| 文件 | 说明 |
|---|---|
| `results/qc/README.md` | QC 目录总说明，定义这里是 synthetic data 质量闸门，不是训练代码。 |
| `results/qc/G2_synthetic_data_QC报告模板_v2.md` | 每批 synthetic run 的正式报告模板。 |
| `results/qc/G2_synthetic_data_QC规则策略_v2.md` | v2 QC 主标准，定义 L0-L12、硬拒绝、人工复查和放行规则。 |
| `results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md` | 官方指标对齐策略，说明 G2 QC 与官方 leaderboard 字段如何衔接。 |
| `results/qc/UCSD_T2W_内容异常检查报告_2026-06-14.md` | UCSD Training 的 t2w 人工/自动核查记录，属于真实数据健康检查参考。 |
| `results/qc/diffusion_quality_metrics_template.csv` | 扩散质量专项指标表头，覆盖 ROI、边界、z 连续性、teacher 与相似性。 |
| `results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` | 官方训练集 t2w gzip header 原始文件名含 fake 的病例清单。 |
| `results/qc/official_leaderboard_metrics_template.csv` | 官方 leaderboard 同款字段模板，用于 real-only 与 real+synth 训练后验收。 |
| `results/qc/official_non000_t2w_cases_2026-06-15.csv` | 非 000 编号病例辅助清单，只用于追踪编号分布，不作为 fake T2W 判据。 |
| `results/qc/official_t2w_gzip_header_audit_2026-06-15.csv` | 官方训练集 T2W gzip header 全量 audit，一例一行记录 fake 判定证据。 |
| `results/qc/qc_case_review_template.csv` | 人工复查记录表头，用于视觉审查与复核结论。 |
| `results/qc/qc_metrics_template_v2.csv` | 新版逐例总 QC 表头，当前 synthetic intake 的主要机器可读输出。 |

### 6. results/splits

| 文件 | 说明 |
|---|---|
| `results/splits/README.md` | 固定真实 train/val/test 划分说明。 |
| `results/splits/splits_master_train_val_test.json` | 全部病例 patient-group master split。 |
| `results/splits/splits_master_train_val_test_membership.csv` | 待补充说明 |
| `results/splits/splits_final_train_val_test.json` | 从 master split 派生的真实 T2W real-only split。 |
| `results/splits/splits_final_train_val_test_membership.csv` | 逐病例 split membership 表，便于人工核查和脚本读取。 |

### 7. results/reports

| 文件 | 说明 |
|---|---|
| `results/reports/README.md` | 报告目录总说明，承接路径检查、QC 汇总、进度报告与模板。 |
| `results/reports/G2_progress_report.md` | G2 主进度报告，汇总当前完成度、文件索引和下一步计划。 |
| `results/reports/ablation_plan_template.md` | real-only / real+synth 的消融模板。 |
| `results/reports/local_data_paths_check.md` | 本机外部数据路径检查结果。 |
| `results/reports/real_data_qc_summary.md` | 真实训练数据 QC 汇总。 |

### 8. results/nnunet_raw

| 文件 | 说明 |
|---|---|
| `results/nnunet_raw/README.md` | nnU-Net raw 根目录说明，说明这里是训练机物化入口，不在仓库保存正式大体积影像。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md` | real-only 数据集占位说明，表示当前只保存 dataset.json 与路径契约。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json` | nnU-Net dataset.json 草案，定义四模态顺序与五类标签。 |

## 结论

1. G2 已完成 patient-group master split、V2/V3 分流、严格 metadata gate 和三态 release 接口。
2. 当前测试证明接口与小型 NIfTI fixture 可运行；真实生成质量仍须等待服务器 NIfTI 批次验收。
3. 大体积影像仍留在外部数据盘或训练机器，不进入仓库。
