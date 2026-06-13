# G2 Synthetic Intake 进度报告

- 生成日期：2026-06-14
- 项目根目录：`/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2`

## 当前进度

- 真实数据基线 run_id：`g2_synthetic_smoke_run_20260614`
- 训练集病例数：3
- accepted：2
- ablation only：1
- needs regeneration：1
- rejected：1

## synthetic smoke 验证

- smoke run_id：`g2_synthetic_smoke_run_20260614`
- 候选数：3
- accepted：2
- ablation only：1
- needs regeneration：1
- rejected：1
- legacy suffix case：1
- native suffix case：1
- mixed suffix case：1

## 下一步

1. 接入真实 G1 生成目录，替换当前 smoke 例子。
2. 复核正式批次的 accepted / rejected 比例，并根据真实样本再微调 QC 阈值。
3. 将通过的 synthetic 样本物化到训练机上的 nnU-Net raw / mapping 流程。
4. 完成训练前的最终消融准备和版本冻结。

## 本次生成的文件

- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_generation_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_candidate_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_accepted_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_rejected_manifest_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_normalized_mapping_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_metrics_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/diffusion_quality_metrics_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_case_review_g2_synthetic_smoke_run_20260614.csv`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_batch_summary_g2_synthetic_smoke_run_20260614.json`
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/reports/G2_synthetic_data_quality_report_g2_synthetic_smoke_run_20260614.md`

## Intake 索引

### synthetic_generation_manifest
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_generation_manifest_g2_synthetic_smoke_run_20260614.csv`

### synthetic_candidate_manifest
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_candidate_manifest_g2_synthetic_smoke_run_20260614.csv`

### synthetic_accepted_manifest
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_accepted_manifest_g2_synthetic_smoke_run_20260614.csv`

### synthetic_rejected_manifest
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_rejected_manifest_g2_synthetic_smoke_run_20260614.csv`

### synthetic_normalized_mapping
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/manifests/synthetic_normalized_mapping_g2_synthetic_smoke_run_20260614.csv`

### qc_metrics
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_metrics_g2_synthetic_smoke_run_20260614.csv`

### diffusion_quality_metrics
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/diffusion_quality_metrics_g2_synthetic_smoke_run_20260614.csv`

### qc_case_review
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_case_review_g2_synthetic_smoke_run_20260614.csv`

### qc_batch_summary
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/qc/qc_batch_summary_g2_synthetic_smoke_run_20260614.json`

### quality_report
- `/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G2/results/reports/G2_synthetic_data_quality_report_g2_synthetic_smoke_run_20260614.md`


## 根目录与入口文件

| 文件 | 说明 |
|---|---|
| `README.md` | G2 项目总入口说明，概述项目目的、目录分工和本仓库的轻量化数据策略。 |
| `task_assignment.md` | G2 团队分工总表，把成员职责和工作拆分在一个入口里。 |
| `data/.gitkeep` | data 目录占位文件，保留未来数据放置点。 |
| `results/.gitkeep` | results 根目录占位文件，保留结果区目录结构。 |
| `results/README.md` | results 总说明，概括本目录只保存轻量产物，不保存大体积 NIfTI。 |
| `results/使用说明.md` | results 根目录总使用说明，帮助快速定位各子目录作用。 |

## 八个主文件夹索引

### 1. code

| 文件 | 说明 |
|---|---|
| `code/.gitkeep` | code 目录占位文件，保证空目录被版本控制保留。 |
| `code/g2_pretraining_audit.py` | 单入口脚本：真实数据基线扫描、synthetic raw intake、legacy suffix 归一、manifest 自动补建、QC、accepted/rejected 导出与进度报告生成。 |

### 2. docs

| 文件 | 说明 |
|---|---|
| `docs/G1_G2_diffusion_output_contract.md` | G1 raw output 与 G2 适配边界的主契约，定义 raw 命名、source CSV、manifest 字段和最低 smoke 标准。 |
| `docs/G2_G1适配执行清单.md` | 按执行顺序拆解 G2 先准备什么、G1 输出后 G2 做什么、如何形成 QC 结果与回传。 |
| `docs/G2_数据生成与质量控制实施方案.md` | 总方案，解释 G2 为什么是 adapter/auditor/publisher，以及 raw intake 到 nnU-Net 导出的全链路。 |
| `docs/G2_数据生成与质量控制实现难度评估.md` | 实现难度与风险说明，帮助确认哪些字段可以自动恢复，哪些必须由 G1 给出。 |
| `docs/G2_模型训练完成前可执行工作清单.md` | 训练前能立即执行的工作清单，属于 G2 的下一步行动仓库。 |

### 3. results/manifests

| 文件 | 说明 |
|---|---|
| `results/manifests/README.md` | 清单区说明，解释真实清单、source CSV、synthetic intake manifest 与 accepted/rejected 输出。 |
| `results/manifests/corrected_label_overlay.csv` | 真实训练病例的 corrected label 覆盖记录，说明哪些病例在最终 manifest 中替换了原始 seg。 |
| `results/manifests/g1_gligan_source_cases_v1.csv` | G2 写给 G1 的兼容 source 表，既保留 GliGAN 口径，也保留 G2 扩展列。 |
| `results/manifests/nnunet_case_mapping_realonly.csv` | real-only nnU-Net 映射表，用于训练机物化 imagesTr/labelsTr。 |
| `results/manifests/real_train_manifest.csv` | 真实训练病例最终主表，已应用 corrected label overlay 并带 final_qc_pass。 |
| `results/manifests/real_train_manifest_raw.csv` | 原始训练病例扫描表，保留 raw seg 与基础 QC 证据。 |
| `results/manifests/real_validation_manifest.csv` | 官方 validation 路径与结构检查表，绝不作为 synthetic source。 |
| `results/manifests/synthetic_accepted_manifest_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 的通过清单，包含进入训练或仅用于消融的样本。 |
| `results/manifests/synthetic_candidate_manifest_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 的候选合并清单，保留原始输入与 QC 判定对照。 |
| `results/manifests/synthetic_generation_manifest_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 自动补建的主清单，承接 G1 legacy raw output 与 G2 标准化字段。 |
| `results/manifests/synthetic_generation_manifest_template_g1.csv` | G1 raw output 或 G2 补建时使用的 synthetic manifest 表头模板。 |
| `results/manifests/synthetic_normalized_mapping_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 的逐模态标准化映射表，连接 raw legacy/native 文件、2026 标准文件和 nnU-Net 目标路径。 |
| `results/manifests/synthetic_normalized_mapping_template.csv` | 逐模态标准化映射模板，定义 raw source、normalized target 与 nnU-Net target 的对应关系。 |
| `results/manifests/synthetic_rejected_manifest_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 的拒绝清单，记录未通过的候选和拒绝原因。 |
| `results/manifests/使用说明.md` | 清单区的手工说明，解释每张 CSV 在 G1/G2/S1/S2 流程里的作用。 |

### 4. results/stats

| 文件 | 说明 |
|---|---|
| `results/stats/README.md` | 统计区说明，解释 label/lesion 分布与 synthetic 目标分布。 |
| `results/stats/real_label_distribution.csv` | 真实训练病例级 label 体素与体积分布。 |
| `results/stats/real_lesion_distribution.csv` | 真实 lesion component 级分布。 |
| `results/stats/real_lesion_distribution_summary.json` | 机器可读统计摘要。 |
| `results/stats/real_lesion_distribution_summary.md` | 人可读统计摘要。 |
| `results/stats/target_synthetic_distribution_v1.md` | 第一轮 synthetic 目标分布与生成限制。 |
| `results/stats/使用说明.md` | 统计区使用说明。 |

### 5. results/qc

| 文件 | 说明 |
|---|---|
| `results/qc/README.md` | QC 目录总说明，定义这里是 synthetic data 质量闸门，不是训练代码。 |
| `results/qc/G2_synthetic_data_QC报告模板_v2.md` | 每批 synthetic run 的正式报告模板。 |
| `results/qc/G2_synthetic_data_QC规则策略_v2.md` | v2 QC 主标准，定义 L0-L12、硬拒绝、人工复查和放行规则。 |
| `results/qc/UCSD_T2W_内容异常检查报告_2026-06-14.md` | UCSD Training 的 t2w 人工/自动核查记录，属于真实数据健康检查参考。 |
| `results/qc/diffusion_quality_metrics_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 的扩散质量专项表，记录 ROI、边界、z 连续性等专项指标。 |
| `results/qc/diffusion_quality_metrics_template.csv` | 扩散质量专项指标表头，覆盖 ROI、边界、z 连续性、teacher 与相似性。 |
| `results/qc/qc_batch_summary_g2_synthetic_smoke_run_20260614.json` | 本次 smoke run 的批次汇总 JSON，提供机器可读统计结果。 |
| `results/qc/qc_case_review_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 的人工复核表，记录需要视觉复查的病例。 |
| `results/qc/qc_case_review_template.csv` | 人工复查记录表头，用于视觉审查与复核结论。 |
| `results/qc/qc_metrics_g2_synthetic_smoke_run_20260614.csv` | 本次 smoke run 的逐例 QC 主表，记录每个样本的 pass/review/reject 判定。 |
| `results/qc/qc_metrics_template.csv` | 旧版 QC 表头，保留兼容。 |
| `results/qc/qc_metrics_template_v2.csv` | 新版逐例总 QC 表头，当前 synthetic intake 的主要机器可读输出。 |
| `results/qc/qc_rules_v1.md` | 旧版规则文档，保留兼容与历史对照。 |
| `results/qc/使用说明.md` | QC 目录使用说明，解释模板、规则和报告怎么串起来。 |

### 6. results/splits

| 文件 | 说明 |
|---|---|
| `results/splits/README.md` | 固定真实验证 fold 的说明。 |
| `results/splits/splits_final_fold0_realval.json` | 当前固定 fold0 的 train/val 划分。 |
| `results/splits/使用说明.md` | split 文件的使用说明。 |

### 7. results/reports

| 文件 | 说明 |
|---|---|
| `results/reports/README.md` | 报告目录总说明，承接路径检查、QC 汇总、进度报告与模板。 |
| `results/reports/G2_progress_report.md` | G2 主进度报告，汇总当前完成度、文件索引和下一步计划。 |
| `results/reports/G2_synthetic_data_quality_report_g2_synthetic_smoke_run_20260614.md` | 本次 smoke run 的质量报告正文，汇总生成、接收和 QC 结论。 |
| `results/reports/G2_synthetic_data_quality_report_template.md` | synthetic 批次质量报告模板。 |
| `results/reports/G2_synthetic_intake_progress_report.md` | synthetic intake 运行索引，专门记录一次 intake 的 manifest / QC / report 产物。 |
| `results/reports/G2_训练前数据准备与G1方案对接总结.docx` | 当前总结的 Word 版，可直接发队友或导师。 |
| `results/reports/G2_训练前数据准备与G1方案对接总结.md` | 当前总结的 Markdown 源稿。 |
| `results/reports/ablation_plan_template.md` | real-only / real+synth 的消融模板。 |
| `results/reports/g2_pretraining_execution_summary.md` | 训练前数据准备的执行摘要。 |
| `results/reports/local_data_paths_check.md` | 本机外部数据路径检查结果。 |
| `results/reports/real_data_qc_summary.md` | 真实训练数据 QC 汇总。 |
| `results/reports/使用说明.md` | 报告目录使用说明，解释不同报告的定位。 |

### 8. results/nnunet_raw

| 文件 | 说明 |
|---|---|
| `results/nnunet_raw/README.md` | nnU-Net raw 根目录说明，说明这里是训练机物化入口，不在仓库保存正式大体积影像。 |
| `results/nnunet_raw/使用说明.md` | nnU-Net raw 区总说明，强调这里只放轻量占位与契约，不放正式影像。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md` | real-only 数据集占位说明，表示当前只保存 dataset.json 与路径契约。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json` | nnU-Net dataset.json 草案，定义四模态顺序与五类标签。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/使用说明.md` | Dataset260 real-only 占位目录说明，指导 S1/S2 根据 mapping 表在训练机生成 imagesTr 和 labelsTr。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/imagesTr/使用说明.md` | imagesTr 目录说明，解释训练机上如何物化四模态图像。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/labelsTr/使用说明.md` | labelsTr 目录说明，解释训练机上如何放置 seg。 |

## 结论

1. G2 已完成真实数据侧的基线准备，并用 smoke run 证明了 synthetic raw intake、legacy suffix 归一、manifest 自动补建、QC 和 accepted/rejected 闭环。
2. 当前工作区已形成清晰的 G2 文件结构索引，后续只需用真实 G1 目录替换 smoke 例子即可。
3. 大体积影像仍留在外部数据盘或训练机器，不进入仓库。
