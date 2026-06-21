# G2 当前进度报告

更新日期：2026-06-19

## 1. 本轮结论

1. G2 已从“按旧 GliGAN 口径接 synthetic 数据”调整为“优先承接 G1 新 Brownian Bridge 缺失 T2W 补全模型输出”。
2. G1 推理输出口径已对齐为 `data/output/<case_id>/...`；G2 intake/QC 允许 `label_kind=completion`，不再强制要求旧 GliGAN source 关系。
3. G2 已补齐 synthetic raw intake、legacy suffix 转换、synthetic manifest 自动填充、QC 自动生成、accepted/rejected 输出、nnU-Net 物化映射。
4. QC 策略已升级为两层口径：训练前数据质量闸门负责筛 synthetic 数据；训练后用官方 Task1 leaderboard 字段验证 real-only 与 real+synth 的实际收益。
5. 官方/模板/榜单字段表已保留，尤其是 `official_leaderboard_metrics_template.csv`，这是从官方 leaderboard 指标整理出的标签表，不能作为旧产物清理。
6. `results` 目录已收敛为 README + 模板 + 真实数据审计结果 + 固定 split + 轻量 nnU-Net 契约；正式 synthetic 影像、大体积预处理缓存和临时 smoke run 不进入仓库。
7. 固定划分已升级为 train/val/test 三分法：历史 259 例 fixed val 锁定为 internal test，再从历史 1036 train 中切出 207 例作为 dev/val。

## 2. 当前可用数据状态

| 项目 | 状态 |
|---|---:|
| 本地带标签训练病例 | 1296 |
| corrected overlay 后 final QC pass | 1295 |
| corrected overlay 后 final QC fail | 1 |
| 官方 validation 病例 | 179 |
| 当前固定 train / val / test | 829 / 207 / 259 |
| 历史兼容 fold0 train / val | 1036 / 259 |
| G1 完整真实 T2W 投影 train / val / test | 660 / 160 / 210 |
| 真实 lesion component 总数 | 9793 |
| tiny / small / large lesion 数 | 3788 / 3922 / 2083 |
| 含 RC 真实训练病例 | 167 |
| G1 MET 96 ROI source 候选 | 472 |
| gzip header 原始文件名含 fake 的 T2W | 265 |

## 3. 官方指标口径

后续判断 synthetic data 是否真正有用，不能只看 G2 训练前 QC pass 数量，必须比较同一 internal test 下的官方同款字段：

1. real-only train+val 调参后，在 internal test 上评估。
2. real+synth train+val 调参后，在同一 internal test 上评估。

核心字段：

1. ET/RC/TC/WT 的 `lesionwise_dsc_mean_*`。
2. ET/RC/TC/WT 的 `lesionwise_nsd_mean_*`。
3. ET/TC/WT/RC 的 `small_instance_tp/fn/fp/f1_*`。

`HD95`、体素级 Dice、分布相似度、强度统计和扩散质量指标只作为内部辅助指标。最终是否采用某一批 synthetic 数据，要以官方同款指标不下降、关键小病灶指标稳定或提升为准。

## 4. 不要清理的模板和审计资产

这些文件是 G2 后续复现、对齐官方评价或给 S1/S2 做消融比较必须保留的资产：

| 文件 | 保留原因 |
|---|---|
| `results/qc/official_leaderboard_metrics_template.csv` | 官方 leaderboard 同款字段模板，是后续记录 real-only / real+synth 官方结果的标签表。 |
| `results/qc/qc_metrics_template_v2.csv` | G2 自动 QC 的逐例字段模板。 |
| `results/qc/diffusion_quality_metrics_template.csv` | 评估扩散补全质量的专项字段模板。 |
| `results/qc/qc_case_review_template.csv` | 人工复查和 borderline case 记录模板。 |
| `results/qc/official_t2w_gzip_header_audit_2026-06-15.csv` | 官方训练数据 T2W gzip header 全量审计证据。 |
| `results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` | 原始 header 中含 fake 的 T2W 病例清单。 |
| `results/qc/official_non000_t2w_cases_2026-06-15.csv` | 非 000 后缀病例辅助审计清单。 |
| `results/manifests/*.csv` | 真实数据、source 候选、synthetic intake 和 nnU-Net 映射的核心索引。 |
| `results/splits/splits_final_train_val_test.json` | S1/S2/G2/G1 必须共用的正式 train/val/test split。 |
| `results/splits/splits_final_train_val_test_membership.csv` | 逐病例 split membership 表，便于人工核查和脚本读取。 |
| `results/splits/splits_final_fold0_realval.json` | 历史兼容 two-way fold；旧 val 已锁定为 internal test。 |

## 5. 八个主区域索引

### 5.1 code

| 文件 | 作用 |
|---|---|
| `code/.gitkeep` | 保留 code 目录。 |
| `code/g2_pretraining_audit.py` | G2 基础审计脚本：扫描真实训练/验证数据，生成 manifest、真实数据统计、G1 source CSV、real-only nnU-Net mapping 和可选 synthetic intake。 |
| `code/g2_create_train_val_test_split.py` | 固定划分脚本：把历史 fixed val 锁成 internal test，并从训练池稳定切出 dev/val。 |
| `code/g2_synthetic_raw_intake_qc.py` | G1 raw output 正式接收脚本：生成 candidate/accepted/rejected manifest、normalized mapping、逐例 QC、扩散质量指标、batch summary 和 run 级报告。 |
| `code/g2_materialize_nnunet_dataset.py` | 把 real mapping 与 accepted synthetic manifest 转成 nnU-Net raw dataset 入口；completion 默认替换 fake/broken T2W，不追加重复病例；`accepted_for_ablation_only` 需显式开启。 |
| `code/g2_official_mets_metrics_parser.py` | 解析 BraTS_evaluation / Panoptica 输出或校验 CSV 是否包含 2026 Task1 leaderboard 字段。 |

### 5.2 docs

| 文件 | 作用 |
|---|---|
| `docs/G1_G2_diffusion_output_contract.md` | G1 输出与 G2 intake/QC 的正式接口契约。 |
| `docs/G1_G2_服务器训练推理QC运行手册.md` | 服务器上从 G1 数据放置、训练/推理到 G2 QC 和 nnU-Net 导出的操作手册。 |
| `docs/G2_G1适配执行清单.md` | G2 如何完全适配 G1 新模型输出、如何回传 QC 结果的逐项执行清单。 |
| `docs/G2_数据生成与质量控制实施方案.md` | G2 数据生成、QC、nnU-Net 导出和消融验证总方案。 |
| `docs/G2_模型训练完成前可执行工作清单.md` | G1 模型训练完成前，G2 可以独立推进的准备工作。 |

### 5.3 results/manifests

| 文件 | 作用 |
|---|---|
| `results/manifests/README.md` | manifest 目录总说明。 |
| `results/manifests/real_train_manifest_raw.csv` | 原始训练病例扫描结果，保留 raw seg 与基础 QC 证据。 |
| `results/manifests/real_train_manifest.csv` | 最终训练 manifest，已应用 corrected label overlay。 |
| `results/manifests/real_validation_manifest.csv` | 官方 validation 结构检查表，不能作为 synthetic source。 |
| `results/manifests/corrected_label_overlay.csv` | corrected labels 覆盖记录。 |
| `results/manifests/g1_met_source_cases_v1.csv` | 给 G1/旧 MET-compatible 流程复用的 source 候选 CSV。 |
| `results/manifests/nnunet_case_mapping_realonly.csv` | S1/S2 物化 real-only nnU-Net 数据集的映射表。 |
| `results/manifests/synthetic_generation_manifest_template_g1.csv` | G1 正式批次或 G2 补建 synthetic manifest 时使用的表头模板。 |
| `results/manifests/synthetic_normalized_mapping_template.csv` | raw/native/legacy 文件到 2026 suffix 与 nnU-Net 目标文件的逐模态映射模板。 |

### 5.4 results/stats

| 文件 | 作用 |
|---|---|
| `results/stats/README.md` | stats 目录总说明。 |
| `results/stats/real_label_distribution.csv` | 真实训练病例级 label 体素与体积分布。 |
| `results/stats/real_lesion_distribution.csv` | 真实 lesion component 级分布。 |
| `results/stats/real_lesion_distribution_summary.json` | 机器可读统计摘要。 |
| `results/stats/real_lesion_distribution_summary.md` | 人可读统计摘要。 |
| `results/stats/target_synthetic_distribution_v1.md` | 第一轮 synthetic 目标分布和生成限制。 |

### 5.5 results/qc

| 文件 | 作用 |
|---|---|
| `results/qc/README.md` | QC 目录总说明。 |
| `results/qc/G2_synthetic_data_QC规则策略_v2.md` | v2 主 QC 规则，定义 L0-L12、硬拒绝、人工复查、脚本责任、S1/S2 通道对齐和训练后验收。 |
| `results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md` | 官方检测策略完整总结：Synapse 2026 Task1、BraTS_evaluation config/parser、leaderboard 字段和 G2 QC 对齐。 |
| `results/qc/G2_synthetic_data_QC报告模板_v2.md` | 每批 synthetic data 的 QC 报告模板。 |
| `results/qc/qc_metrics_template_v2.csv` | 正式逐例自动 QC 指标模板。 |
| `results/qc/diffusion_quality_metrics_template.csv` | 扩散生成质量专项指标模板。 |
| `results/qc/qc_case_review_template.csv` | 人工复查记录模板。 |
| `results/qc/official_leaderboard_metrics_template.csv` | 官方 leaderboard 同款字段模板，必须保留。 |
| `results/qc/UCSD_T2W_内容异常检查报告_2026-06-14.md` | T2W 内容异常与 fake T2W 补充检查报告。 |
| `results/qc/official_t2w_gzip_header_audit_2026-06-15.csv` | 1296 例训练病例 T2W gzip header 全量 audit。 |
| `results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv` | gzip header 原始文件名含 fake 的 265 例清单。 |
| `results/qc/official_non000_t2w_cases_2026-06-15.csv` | 非 000 后缀病例辅助清单，不能单独作为 fake 判据。 |

### 5.6 results/splits

| 文件 | 作用 |
|---|---|
| `results/splits/README.md` | split 目录总说明。 |
| `results/splits/splits_final_train_val_test.json` | 当前正式 train/val/test 划分：829/207/259。 |
| `results/splits/splits_final_train_val_test_membership.csv` | 逐病例 membership 表。 |
| `results/splits/splits_final_fold0_realval.json` | 历史兼容 two-way fold：1036/259。 |

### 5.7 results/reports

| 文件 | 作用 |
|---|---|
| `results/reports/README.md` | reports 目录总说明。 |
| `results/reports/G2_progress_report.md` | 当前进度报告、保留资产说明和结果区文件索引。 |
| `results/reports/local_data_paths_check.md` | 本机外部数据路径检查结果。 |
| `results/reports/real_data_qc_summary.md` | 真实训练数据 QC 汇总。 |
| `results/reports/ablation_plan_template.md` | real-only vs real+synth 消融模板，已对齐官方指标。 |

### 5.8 results/nnunet_raw

| 文件 | 作用 |
|---|---|
| `results/nnunet_raw/README.md` | nnU-Net raw 根目录说明。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/README.md` | real-only 数据集占位说明。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/dataset.json` | nnU-Net dataset.json 草案。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/使用说明.md` | Dataset260 real-only 使用说明，说明该目录为什么只放轻量契约。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/imagesTr/使用说明.md` | imagesTr 物化说明；正式影像不进仓库。 |
| `results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly/labelsTr/使用说明.md` | labelsTr 物化说明；正式标签不进仓库。 |

## 6. 已清理内容

1. 2026-06-14 `g2_synthetic_smoke_run_20260614` 的演示 manifest、QC CSV、batch summary 和质量报告。
2. 旧版 `qc_rules_v1.md`。
3. 旧版 `qc_metrics_template.csv`。
4. results 目录下的 `.DS_Store` 本地缓存文件。
5. 与 README 重复、且容易让索引混乱的顶层 `使用说明.md`；Dataset260 子目录下仍保留必要的物化说明。
6. 与 `G2_progress_report.md` 内容重叠的旧 `g2_pretraining_execution_summary.md` 摘要文件。

## 7. 下一步

1. 等 G1 交付真实 `data/output/<case_id>` run 后，用 `g2_synthetic_raw_intake_qc.py` 生成 candidate/accepted/rejected、normalized mapping 和 QC 报告。
2. 对 borderline 病例填写 `qc_case_review_template.csv`，不能手工绕过 QC。
3. 用 `g2_materialize_nnunet_dataset.py` 在训练机物化 real+synth 数据，默认只纳入 `accepted_for_training=True`，并用 accepted completion 替换 fake/broken T2W。
4. S1/S2 训练出 real-only 与 real+synth 预测后，用官方 BraTS_evaluation 或 `g2_official_mets_metrics_parser.py` 汇总官方同款字段。
5. 只有官方同款指标不下降，且 small-instance 指标稳定或提升，才能把某一批 synthetic 数据写成主方案。
