# G2 当前进度报告

生成日期：2026-06-15

## 1. 本轮结论

1. 已核对官方 Task1 当前 Results leaderboard 字段：官方主口径是 `lesionwise_*` DSC/NSD 与 `small_instance_*` TP/FN/FP/F1。
2. 已明确 G2 当前 QC 是“训练前数据质量闸门”，不是官方 leaderboard 指标替代品。
3. 已按 2026 Task1 重新整理官方检测策略：Panoptica config、ET/RC/TC/WT 区域、large lesion lesionwise DSC/NSD、小病灶 TP/FN/FP/F1、排名与统计检验。
4. 已补齐 G2 三个工程入口：G1 raw intake/QC、accepted 数据 nnU-Net 物化、官方指标 parser/validator。
5. 已清理 2026-06-14 smoke run 演示产物、旧 v1 QC 规则和旧 QC 模板。
6. 已保留真实数据 manifest、统计结果、T2W gzip header audit、v2 QC 模板和 nnU-Net 轻量契约。

## 2. 当前可用数据状态

| 项目 | 状态 |
|---|---:|
| 本地带标签训练病例 | 1296 |
| corrected overlay 后 final QC pass | 1295 |
| corrected overlay 后 final QC fail | 1 |
| 官方 validation 病例 | 179 |
| fixed fold0 train / val | 1036 / 259 |
| 真实 lesion component 总数 | 9793 |
| tiny / small / large lesion 数 | 3788 / 3922 / 2083 |
| 含 RC 真实训练病例 | 167 |
| G1 96 ROI source 候选 | 472 |
| gzip header 原始文件名含 fake 的 T2W | 265 |

## 3. 官方指标口径

后续判断 synthetic data 是否真正有用，必须填 `results/qc/official_leaderboard_metrics_template.csv` 同款字段，并比较：

1. real-only fold0。
2. real+synth fold0。

主字段：

1. ET/RC/TC/WT 的 `lesionwise_dsc_mean_*`。
2. ET/RC/TC/WT 的 `lesionwise_nsd_mean_*`。
3. ET/TC/WT/RC 的 `small_instance_tp/fn/fp/f1_*`。

`HD95` 和 `AUC over F1` 只作为内部辅助指标，不作为当前官方主报告字段。

## 4. 八个主文件夹索引

### 4.1 code

| 文件 | 作用 |
|---|---|
| `code/.gitkeep` | 保留 code 目录。 |
| `code/g2_pretraining_audit.py` | G2 基础审计脚本：真实数据扫描、manifest 生成、模板刷新、source CSV、real-only mapping 和可选 synthetic intake。 |
| `code/g2_synthetic_raw_intake_qc.py` | G1 raw run 正式接收脚本：自动生成 candidate/accepted/rejected manifest、QC CSV、diffusion quality、batch summary 和质量报告。 |
| `code/g2_materialize_nnunet_dataset.py` | 转换脚本：把 real mapping 与 accepted synthetic manifest 转成 nnU-Net raw dataset 入口，支持 `manifest-only/symlink/copy` 和 `g2_official/s2_current` 通道顺序。 |
| `code/g2_official_mets_metrics_parser.py` | 检验脚本：解析 BraTS_evaluation Panoptica JSON 或校验 CSV 是否包含 2026 Task1 leaderboard 字段。 |

### 4.2 docs

| 文件 | 作用 |
|---|---|
| `docs/G1_G2_diffusion_output_contract.md` | G1 raw output 与 G2 intake 的接口契约。 |
| `docs/G2_G1适配执行清单.md` | G2 如何适配 G1 输出、如何回传 QC 结果的执行清单。 |
| `docs/G2_数据生成与质量控制实施方案.md` | G2 数据生成、QC、nnU-Net 导出和消融总方案。 |
| `docs/G2_模型训练完成前可执行工作清单.md` | 模型训练前 G2 可先完成的任务清单。 |

### 4.3 results/manifests

| 文件 | 作用 |
|---|---|
| `results/manifests/README.md` | manifest 目录总说明。 |
| `results/manifests/使用说明.md` | 每张 manifest CSV 的字段和用法说明。 |
| `results/manifests/real_train_manifest_raw.csv` | 原始训练病例扫描结果，保留 raw seg 与基础 QC 证据。 |
| `results/manifests/real_train_manifest.csv` | 最终训练 manifest，已应用 corrected label overlay。 |
| `results/manifests/real_validation_manifest.csv` | 官方 validation 结构检查表，不能作为 synthetic source。 |
| `results/manifests/corrected_label_overlay.csv` | corrected labels 覆盖记录。 |
| `results/manifests/g1_gligan_source_cases_v1.csv` | 给 G1 diffusion/GliGAN-compatible 生成流程的 source CSV。 |
| `results/manifests/nnunet_case_mapping_realonly.csv` | S1/S2 物化 real-only nnU-Net 数据集的映射表。 |
| `results/manifests/synthetic_generation_manifest_template_g1.csv` | G1 正式批次或 G2 补建 synthetic manifest 时使用的表头模板。 |
| `results/manifests/synthetic_normalized_mapping_template.csv` | raw/native/legacy 文件到 2026 suffix 和 nnU-Net 目标文件的逐模态映射模板。 |

### 4.4 results/stats

| 文件 | 作用 |
|---|---|
| `results/stats/README.md` | stats 目录总说明。 |
| `results/stats/使用说明.md` | 统计结果使用说明。 |
| `results/stats/real_label_distribution.csv` | 真实训练病例级 label 体素与体积分布。 |
| `results/stats/real_lesion_distribution.csv` | 真实 lesion component 级分布。 |
| `results/stats/real_lesion_distribution_summary.json` | 机器可读统计摘要。 |
| `results/stats/real_lesion_distribution_summary.md` | 人可读统计摘要。 |
| `results/stats/target_synthetic_distribution_v1.md` | 第一轮 synthetic 目标分布和生成限制。 |

### 4.5 results/qc

| 文件 | 作用 |
|---|---|
| `results/qc/README.md` | QC 目录总说明。 |
| `results/qc/使用说明.md` | QC 文件、模板和官方指标验收说明。 |
| `results/qc/G2_synthetic_data_QC规则策略_v2.md` | v2 主 QC 规则，定义 L0-L12、硬拒绝、人工复查、脚本责任、S1/S2 通道对齐和训练后验收。 |
| `results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md` | 官方检测策略完整总结：Synapse 2026 Task1、BraTS_evaluation config/parser、leaderboard 字段和 G2 QC 对齐。 |
| `results/qc/G2_synthetic_data_QC报告模板_v2.md` | 每批 synthetic data 的 QC 报告模板。 |
| `results/qc/qc_metrics_template_v2.csv` | 正式逐例自动 QC 指标模板。 |
| `results/qc/diffusion_quality_metrics_template.csv` | 扩散生成质量专项指标模板。 |
| `results/qc/qc_case_review_template.csv` | 人工复查记录模板。 |
| `results/qc/official_leaderboard_metrics_template.csv` | 官方 leaderboard 同款字段模板。 |
| `results/qc/UCSD_T2W_内容异常检查报告_2026-06-14.md` | T2W 内容异常与 fake T2W 补充检查报告。 |
| `results/qc/official_t2w_gzip_header_audit_2026-06-15.csv` | 1296 例训练病例 T2W gzip header 全量 audit。 |
| `results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` | gzip header 原始文件名含 fake 的 265 例清单。 |
| `results/qc/official_non000_t2w_cases_2026-06-15.csv` | 非 000 后缀病例辅助清单，不能作为 fake 判据。 |

### 4.6 results/splits

| 文件 | 作用 |
|---|---|
| `results/splits/README.md` | split 目录总说明。 |
| `results/splits/使用说明.md` | fixed fold 使用说明。 |
| `results/splits/splits_final_fold0_realval.json` | real-only 与 real+synth 统一复用的固定 fold0。 |

### 4.7 results/reports

| 文件 | 作用 |
|---|---|
| `results/reports/README.md` | reports 目录总说明。 |
| `results/reports/使用说明.md` | 报告文件定位说明。 |
| `results/reports/G2_progress_report.md` | 当前进度报告和文件索引。 |
| `results/reports/g2_pretraining_execution_summary.md` | 训练前数据准备执行摘要。 |
| `results/reports/local_data_paths_check.md` | 本机外部数据路径检查结果。 |
| `results/reports/real_data_qc_summary.md` | 真实训练数据 QC 汇总。 |
| `results/reports/ablation_plan_template.md` | real-only vs real+synth 消融模板，已对齐官方指标。 |

### 4.8 results/nnunet_raw

| 文件 | 作用 |
|---|---|
| `results/nnunet_raw/README.md` | nnU-Net raw 根目录说明。 |
| `results/nnunet_raw/使用说明.md` | 训练机物化 nnU-Net 数据的总说明。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md` | real-only 数据集占位说明。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/使用说明.md` | Dataset260 real-only 使用说明。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json` | nnU-Net dataset.json 草案。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/imagesTr/使用说明.md` | imagesTr 物化说明。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/labelsTr/使用说明.md` | labelsTr 物化说明。 |

## 5. 已清理内容

1. 2026-06-14 `g2_synthetic_smoke_run_20260614` 的演示 manifest、QC CSV、batch summary 和质量报告。
2. 旧版 `qc_rules_v1.md`。
3. 旧版 `qc_metrics_template.csv`。
4. results 目录下的 `.DS_Store` 本地缓存文件。

## 6. 下一步

1. 等 G1 交付真实 synthetic run 后，按 v2 模板重新生成 `{run_id}` 产物。
2. 用 `g2_synthetic_raw_intake_qc.py` 产出 accepted/rejected，不能手工跳过 QC。
3. 用 `g2_materialize_nnunet_dataset.py` 在训练机物化 real+synth 数据，并在 S1/S2 之间固定通道顺序。
4. S1/S2 训练出 real-only 与 real+synth 预测后，用 `g2_official_mets_metrics_parser.py` 或官方 scorer 输出同款字段。
5. 只有官方同款指标不下降，且 small-instance 指标稳定或提升，才能把 synthetic batch 写成主方案。
