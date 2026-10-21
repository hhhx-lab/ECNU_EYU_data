# 研究路线、关键版本与历史身份

整理批次：`20260918T153706Z`。本文依据本地文件结构、计划、模型选择和运行合同梳理；不把目录中的 `pass`、`final` 或 `COMPLETE` 单独解释成实验正式通过。

## G1：两条生成路线分开

| 路线 | 代码或结果位置（相对work_space） | 归档解释 |
|---|---|---|
| VAE/encoder-decoder及BBDM缺失模态路线 | `G1/10_code/brats2025-latent-ensemble-synthesis-main-v3` | 完整保留训练、配置、权重、测试与Slurm布局；不拆分内部运行合同 |
| VAE微调前后对照 | `G1/30_results/vae_finetuned/run_3034060` | baseline、finetuned、delta三份指标表；run号不是日期，不虚构运行日期 |
| V3配对验证 | `G1/30_results/v3_validation/run_3056246` | evaluation_run、指标与synthesized输出 |
| 缺失T2W补全 | `G1/30_results/missing_t2w_completion/run_3104668` | 病例级合成结果、generation/inference记录与G2接收manifest |
| 扩散合成增强工程 | `G1/10_code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN` | 与缺失T2W补全不是同一用途；上游完整工程和许可证保留 |
| Diffusion V3汇总包 | `G1/30_results/g1_diffusion_v3_final_20260720` | canonical、sources、deployment、evaluation、metadata分区原样保留；同名副本不合并 |
| 单模型与r4 ensemble图像评估 | `G1/30_results/experimental_unvalidated_single_model_eval_20260812_r4` | bbdm_only、encdec_only、ensemble_r4与paired比较；旧失败launch记录仍在 |

`G1/20_data/latents` 等数据目录与 `diffusion_cache`、raw/input/output、attention_masks分别登记。不能因为名为cache就当作可删除垃圾。部分目录本地只有占位文件，不推断远端也没有数据。

G1图像重建质量、offline缺失模态补全后分割、online Diffusion增强后分割是三个不同问题；本文不把第一层排名写成后两层收益。

## G2：数据口径、QC和发布分开

`G2/10_code` 保留接收、QC、split、manifest、nnUNet materialization等脚本，`50_slurm`、`60_cloud`、`70_tests`分别保留运行资料和测试。

| 集合 | 位置 | 解释 |
|---|---|---|
| 病例及数据清单 | `G2/30_results/manifests` | 原始、补全、合成、标签和病例身份的衔接 |
| 固定数据划分 | `G2/30_results/splits` | master划分与real-T2W子集分开；不能只凭103例数量判断属于同一cohort |
| 分布统计 | `G2/30_results/stats` | 真实病灶统计和目标合成分布，不等同最终训练集 |
| V3配对影像QC | `G2/30_results/qc/v3_paired_validation` | 配对验证、spatial smoke、FINAL_GATE及人工override记录原样并存 |
| 补全人工审查 | `G2/30_results/qc/v3_completion_review/run_3104668` | completion审查，不混成无条件合成样本质量 |
| S2 teacher评估 | `G2/30_results/qc/v3_s2_teacher/run_3056246` | 下游teacher证据，不能替代全部分割对照 |
| Diffusion检查点QC | `G2/30_results/qc/diffusion_checkpoint_*` | preflight、smoke20、full94 recovery是不同阶段，分别保留 |
| 历史数据集视图 | `G2/30_results/nnunet_raw/Dataset260_BraTS2026_MET_RealOnly` | 保留其历史Dataset编号，不自动改成Dataset264 |
| 报告与模板 | `G2/30_results/reports`、`G2/40_documents` | 报告、未来计划和空白模板不能互相冒充 |

来源：[G2原README](../G2/README.md)、[split文件夹](../G2/30_results/splits)、[QC文件夹](../G2/30_results/qc)。历史说明中的1035/130/130与真实T2W子集823/103/104应按各自split和manifest追踪，不能互换。

## S1 与其他分割分支

S1主工程为 `S1/11_project_brats2026_multitask_s1_v2/repository`。`S1/10_code` 等模板目录不代表第二套已训练成果。当前本地只有57个原文件，未发现独立训练结果集合，不编造远端完成情况。

S3保留自己的模型分支与文档。S4按SAM2-UNet代码及运行手册归档。S5按SegMamba整理，其 `monai`、`mamba`、`causal-conv1d` 等第三方源码和作者、许可证均保留，不重新安装依赖，也不清理看似重复的上游文件。

## S2：基座、增强、提交和恢复分开

主工程为 `S2/11_project_brats2026_s2_rc_v1_0/repository`。`S2/30_results` 是实体结果目录，旧 `S2/results` 是兼容入口。

### 原E比赛基座的身份

用户所说“最终用于比赛的原E”对应模型选择证据中的 E：Dataset264 completion加Focal路线，epoch200，checkpoint SHA256：

`4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`

依据：[MODEL_SELECTION.json](../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/MODEL_SELECTION.json) 与 [OFFICIAL_INFERENCE_CONTRACT.json](../S2/30_results/s2_official_validation_inference_20260727_r1/OFFICIAL_INFERENCE_CONTRACT.json)。原E不是后来B-F路线中的E-SNFH，也不能把“未加online增强”误写成“没有用过offline补全或Focal改动”。本文只确认身份，不重算谁更好。

### 时间线与证据边界

| 阶段 | 原实验目录名或系列 | 整理边界 |
|---|---|---|
| Real-only与completion | `realonly_fold0`、`s2_completion_dataset264_t2w_20260720` | Dataset263 real-only与Dataset264补全后实验分开 |
| 小病灶及Focal消融 | `s2_small_lesion_ablation_20260721`、`20260724_official_eval_retry_numpy126` | 训练包与官方评估重试包均留存 |
| E继续训练与保守回退 | `s2_e_continue_fallback_20260726_r1` | 不以继续训练版本替换原E身份 |
| Route A及FixV2 | `s2_met_aug_route_a_20260726_r4/r5`、`s2_met_aug_fix_v2_20260728_r1` | 失败、人工review、配置和可视化全保留 |
| 官方推理、转移、打包 | `s2_official_validation_inference_20260727_r1`、`transfer_20260727_r1`、`submission_20260728_r1` | 模型选择、推理输出、传输证据和提交包分别索引，不新建ZIP或上传 |
| 27例缺失T2W原E评估 | 历史名 `27。result` | 计划时间UTC 2026-08-13 18:44:35，对应北京时间8月14日；原名保留，内容是27例合成T2W验证，不是任意“27号结果” |
| FixV3 full200与attempt11 | `experimental_unvalidated_s2_met_aug_fix_v3_full200_20260812_r1_route_rebuild`、`s2_met_aug_fix_v3_full200_attempt11_20260823_r1` | rebuild与attempt11分别保留，不合并成一个最新成功版本 |
| G1-r4 endpoint | `s2_g1r4_endpoint_validation_20260824_r1/r2` | r1失败与r2的val27、fixed103、locked26证据分别留存 |
| E/FixV3回顾性比较 | `s2_e_fixv3_fixed103_retrospective_20260827_r1` | 回顾性分析，不能提升为预注册正式门控 |
| Table7 | `s2_table7_disjoint_microf1_20260828_*` | r1、r2_v100、r3/r4_evaluation_only及失败证据保持独立 |
| 五路Screen125 | `s2_met_aug_btof_screen125_20260831_r1` | equal-budget五路训练；不混入175epoch的额外预算 |
| 五路selected-checkpoint官方MET | `s2_met_aug_btof_selected103_official_met_20260903_r1`、`20260907_r2/r3/r4` | 原五路选择与官方MET单独归档，历次恢复不覆盖旧失败 |
| B/E-SNFH/F Extend175 | `s2_met_aug_bef_extend175_20260903_r1/r2/r3`、`20260907_r4` | 独立收敛探索，不替代Screen125横评 |
| epoch175官方MET及CPU恢复 | `s2_met_aug_bef_epoch175_official_met_20260910_r1`、`20260911_r2` | r1原GPU包装失败与r2 CPU恢复是两个身份；新恢复不能改写旧job结局 |

上述多数实验位于 `S2/30_results/experimental_unvalidated`，少量更早历史包直接位于 `30_results`。没有为追求统一层级而改变冻结包深度；[S2分类入口](modules/S2/README.md)已按主题把它们集中导航。

`27。result`计划依据：[E_MISSING_T2W_VAL27_EXECUTION_PLAN.json](../S2/30_results/27%E3%80%82result/E_MISSING_T2W_VAL27_EXECUTION_PLAN.json)。该计划声明27例内部验证、26例locked test排除、无需训练，且使用原E checkpoint。不能把27例合成T2W结果当作存在真实T2W的配对反事实。

### 阅读结果时的约束

同一fixed103用于checkpoint选择和结果报告的Screen125/Extend175比较是exploratory。阅读报告须看原始均值、共同定义病例数、paired差异、CI、p及FP/FN方向，而不是只看单个均值或文件名中的best。本次整理未重新执行任何官方评估器或统计流程。

目录中有 `status=pass` 与旧 `FAILED` 同时存在时，应按阶段、来源和attempt读证据，不靠全局搜索一次性宣布成功。`COMPLETE.ok`数量只反映本地文件存在。

## 原件和轻量拉回包

远端快照、`remote_exact`、`remote_lightweight`、`sources`、`canonical`和本地验收文档全部保留。即使SHA相同，也可能承担来源证明、传输校验或冻结副本的职责，不能直接去重。

本次建立的是可浏览的物理分类、逐文件索引与内容版本基线，不是凭空补齐远端全量数据，也不是Git历史重建。
