# S2 分类入口
整理批次：20260918T153706Z。本地原文件 10029 份，9,743,778,789 字节。
本页是整理导航，不是实验验收。实验目录内部、失败证据和历史文档不改写。
## 目录用途
- [10_code](resources/10_code)：原入口 `S2/code` 保留为兼容链接。
- [20_data](resources/20_data)：原入口 `S2/data` 保留为兼容链接。
- [30_results](resources/30_results)：原入口 `S2/results` 保留为兼容链接。
- [40_documents](resources/40_documents)：原入口 `S2/docs` 保留为兼容链接。
- [50_slurm](resources/50_slurm)：原入口 `S2/slurm` 保留为兼容链接。
- [60_cloud](resources/60_cloud)：原入口 `S2/cloud` 保留为兼容链接。
- [80_submission_container](resources/80_submission_container)：原入口 `S2/container` 保留为兼容链接。
- [11_project_brats2026_s2_rc_v1_0](resources/11_project_brats2026_s2_rc_v1_0)：原入口 `S2/BraTS2026_S2_RC_v1.0` 保留为兼容链接。

## 本地文件分类
| 分类 | 文件数 |
|---|---:|
| 01_source_and_execution_code | 346 |
| 02_structured_configs_and_metadata | 314 |
| 03_documents_and_licenses | 131 |
| 04_tables_and_manifests | 158 |
| 05_arrays_images_and_cached_data | 7947 |
| 06_model_checkpoints | 12 |
| 07_figures_and_visual_evidence | 435 |
| 08_logs_and_event_streams | 185 |
| 09_audits_hashes_and_gate_evidence | 301 |
| 10_archives_and_submission_packages | 2 |
| 11_cache_and_system_metadata | 178 |
| 12_historical_placeholders | 4 |
| 14_other_preserved_artifacts | 16 |

## 实验与结果集合时间线
日期来自文件名或父目录，不等于核实后的运行日期；undated 表示未知。标记数量不是成功判定。

| 日期 | 主题及原实验 | 文件数 | 失败标记 | COMPLETE标记 |
|---|---|---:|---:|---:|
| 2026-07-20 | 基线、原E与比赛推理提交 / [s2_completion_dataset264_t2w_20260720](experiments_and_collections/01_baseline_and_submission/2026-07-20__s2_completion_dataset264_t2w_20260720__05554625) | 133 | 0 | 3 |
| 2026-07-21 | 小病灶消融、Route A与FixV2 / [s2_small_lesion_ablation_20260721](experiments_and_collections/02_augmentation_development/2026-07-21__s2_small_lesion_ablation_20260721__833d03ff) | 462 | 0 | 4 |
| 2026-07-24 | 小病灶消融、Route A与FixV2 / [s2_small_lesion_ablation_20260724_official_eval_retry_numpy126](experiments_and_collections/02_augmentation_development/2026-07-24__s2_small_lesion_ablation_20260724_official_eval_retry_numpy126__53a9075d) | 656 | 0 | 3 |
| 2026-07-26 | 基线、原E与比赛推理提交 / [s2_e_continue_fallback_20260726_r1](experiments_and_collections/01_baseline_and_submission/2026-07-26__s2_e_continue_fallback_20260726_r1__6de07c3e) | 110 | 0 | 1 |
| 2026-07-26 | 小病灶消融、Route A与FixV2 / [s2_met_aug_route_a_20260726_r4](experiments_and_collections/02_augmentation_development/2026-07-26__s2_met_aug_route_a_20260726_r4__a5c5086c) | 62 | 1 | 0 |
| 2026-07-26 | 小病灶消融、Route A与FixV2 / [s2_met_aug_route_a_20260726_r5](experiments_and_collections/02_augmentation_development/2026-07-26__s2_met_aug_route_a_20260726_r5__ce212742) | 62 | 0 | 0 |
| 2026-07-27 | 基线、原E与比赛推理提交 / [s2_official_validation_inference_20260727_r1](experiments_and_collections/01_baseline_and_submission/2026-07-27__s2_official_validation_inference_20260727_r1__d44dde52) | 882 | 0 | 11 |
| 2026-07-27 | 基线、原E与比赛推理提交 / [s2_official_validation_transfer_20260727_r1](experiments_and_collections/01_baseline_and_submission/2026-07-27__s2_official_validation_transfer_20260727_r1__81caa6bd) | 2 | 0 | 0 |
| 2026-07-28 | 小病灶消融、Route A与FixV2 / [s2_met_aug_fix_v2_20260728_r1](experiments_and_collections/02_augmentation_development/2026-07-28__s2_met_aug_fix_v2_20260728_r1__8e01327d) | 904 | 4 | 0 |
| 2026-07-28 | 基线、原E与比赛推理提交 / [s2_official_validation_submission_20260728_r1](experiments_and_collections/01_baseline_and_submission/2026-07-28__s2_official_validation_submission_20260728_r1__601014a5) | 4 | 0 | 0 |
| 2026-08-06 | 基线、原E与比赛推理提交 / [s2_validation_leaderboard_20260806_r1](experiments_and_collections/01_baseline_and_submission/2026-08-06__s2_validation_leaderboard_20260806_r1__3abeb77c) | 54 | 0 | 0 |
| 2026-08-12 | FixV3与回顾性比较 / [experimental_unvalidated_s2_met_aug_fix_v3_full200_20260812_r1_route_rebuild](experiments_and_collections/03_fix_v3/2026-08-12__experimental_unvalidated_s2_met_aug_fix_v3_full200_20260812_r1_route_rebuild__dfdd55c6) | 5061 | 0 | 1 |
| 2026-08-14 | S2共享代码、数据与运行资料 / [experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1](experiments_and_collections/10_s2_shared/2026-08-14__experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1__f650173f) | 7 | 0 | 0 |
| 2026-08-23 | FixV3与回顾性比较 / [s2_met_aug_fix_v3_full200_attempt11_20260823_r1](experiments_and_collections/03_fix_v3/2026-08-23__s2_met_aug_fix_v3_full200_attempt11_20260823_r1__6ca9ef0f) | 130 | 0 | 2 |
| 2026-08-24 | G1-r4补全与分割endpoint / [s2_g1r4_endpoint_validation_20260824_r1](experiments_and_collections/08_g1r4_endpoint/2026-08-24__s2_g1r4_endpoint_validation_20260824_r1__234a5c22) | 24 | 1 | 0 |
| 2026-08-24 | G1-r4补全与分割endpoint / [s2_g1r4_endpoint_validation_20260824_r2](experiments_and_collections/08_g1r4_endpoint/2026-08-24__s2_g1r4_endpoint_validation_20260824_r2__b21a4548) | 194 | 0 | 14 |
| 2026-08-27 | FixV3与回顾性比较 / [s2_e_fixv3_fixed103_retrospective_20260827_r1](experiments_and_collections/03_fix_v3/2026-08-27__s2_e_fixv3_fixed103_retrospective_20260827_r1__118d40ad) | 6 | 0 | 1 |
| 2026-08-28 | Table7与micro-F1探索 / [s2_table7_disjoint_microf1_20260828_r1](experiments_and_collections/09_table7/2026-08-28__s2_table7_disjoint_microf1_20260828_r1__a3fb70fe) | 10 | 0 | 0 |
| 2026-08-28 | Table7与micro-F1探索 / [s2_table7_disjoint_microf1_20260828_r2_v100](experiments_and_collections/09_table7/2026-08-28__s2_table7_disjoint_microf1_20260828_r2_v100__cd4f35c6) | 32 | 1 | 0 |
| 2026-08-28 | Table7与micro-F1探索 / [s2_table7_disjoint_microf1_20260828_r3_evaluation_only](experiments_and_collections/09_table7/2026-08-28__s2_table7_disjoint_microf1_20260828_r3_evaluation_only__78d14039) | 11 | 1 | 0 |
| 2026-08-28 | Table7与micro-F1探索 / [s2_table7_disjoint_microf1_20260828_r4_evaluation_only](experiments_and_collections/09_table7/2026-08-28__s2_table7_disjoint_microf1_20260828_r4_evaluation_only__a82de410) | 33 | 1 | 0 |
| 2026-08-31 | B至F等预算Screen125训练 / [s2_met_aug_btof_screen125_20260831_r1](experiments_and_collections/04_btof_screen125/2026-08-31__s2_met_aug_btof_screen125_20260831_r1__918c5150) | 74 | 0 | 5 |
| 2026-09-03 | B、E-SNFH、F独立Extend175 / [s2_met_aug_bef_extend175_20260903_r1](experiments_and_collections/06_bef_extend175/2026-09-03__s2_met_aug_bef_extend175_20260903_r1__ece6306e) | 8 | 1 | 0 |
| 2026-09-03 | B、E-SNFH、F独立Extend175 / [s2_met_aug_bef_extend175_20260903_r2](experiments_and_collections/06_bef_extend175/2026-09-03__s2_met_aug_bef_extend175_20260903_r2__833ae485) | 29 | 3 | 0 |
| 2026-09-03 | B、E-SNFH、F独立Extend175 / [s2_met_aug_bef_extend175_20260903_r3](experiments_and_collections/06_bef_extend175/2026-09-03__s2_met_aug_bef_extend175_20260903_r3__fd173a51) | 7 | 0 | 0 |
| 2026-09-03 | 五路selected-checkpoint官方MET / [s2_met_aug_btof_selected103_official_met_20260903_r1](experiments_and_collections/05_screen125_official_met/2026-09-03__s2_met_aug_btof_selected103_official_met_20260903_r1__cdc708d7) | 39 | 5 | 0 |
| 2026-09-07 | B、E-SNFH、F独立Extend175 / [s2_met_aug_bef_extend175_20260907_r4](experiments_and_collections/06_bef_extend175/2026-09-07__s2_met_aug_bef_extend175_20260907_r4__a3ad7f62) | 51 | 0 | 3 |
| 2026-09-07 | 五路selected-checkpoint官方MET / [s2_met_aug_btof_selected103_official_met_20260907_r2](experiments_and_collections/05_screen125_official_met/2026-09-07__s2_met_aug_btof_selected103_official_met_20260907_r2__dab546ab) | 89 | 2 | 6 |
| 2026-09-07 | 五路selected-checkpoint官方MET / [s2_met_aug_btof_selected103_official_met_20260907_r3](experiments_and_collections/05_screen125_official_met/2026-09-07__s2_met_aug_btof_selected103_official_met_20260907_r3__966071ed) | 13 | 0 | 0 |
| 2026-09-07 | 五路selected-checkpoint官方MET / [s2_met_aug_btof_selected103_official_met_20260907_r4](experiments_and_collections/05_screen125_official_met/2026-09-07__s2_met_aug_btof_selected103_official_met_20260907_r4__7c16ca61) | 81 | 0 | 6 |
| 2026-09-10 | epoch175官方MET与CPU恢复 / [s2_met_aug_bef_epoch175_official_met_20260910_r1](experiments_and_collections/07_epoch175_official_met/2026-09-10__s2_met_aug_bef_epoch175_official_met_20260910_r1__15aab5b6) | 29 | 1 | 0 |
| 2026-09-11 | epoch175官方MET与CPU恢复 / [s2_met_aug_bef_epoch175_official_met_20260911_r2](experiments_and_collections/07_epoch175_official_met/2026-09-11__s2_met_aug_bef_epoch175_official_met_20260911_r2__69e7390e) | 106 | 5 | 8 |
| undated | 基线、原E与比赛推理提交 / [27。result](experiments_and_collections/01_baseline_and_submission/undated__27_result__34be2b15) | 76 | 4 | 1 |
| undated | 基线、原E与比赛推理提交 / [realonly_fold0](experiments_and_collections/01_baseline_and_submission/undated__realonly_fold0__3601f10c) | 7 | 0 | 0 |
| undated | 基线、原E与比赛推理提交 / [s2_eval_results](experiments_and_collections/01_baseline_and_submission/undated__s2_eval_results__5047a08b) | 211 | 0 | 0 |

## 文档逐项分类
历史计划仍是计划；文件标题来自首个标题或文件名，不据此认定已经实施。完整记录见总目录 DOCUMENT_REGISTER.csv。

| 分类 | 日期 | 文档 | 主题 |
|---|---|---|---|
| 01_guides_and_readmes | 2026-07-20 | [S2 Dataset264 Completion-Only nnU-Net Result Archive](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/README.md) | 基线、原E与比赛推理提交 |
| 01_guides_and_readmes | 2026-07-20 | [S2 候选 B 固定 103 例官方兼容评估](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/official_style_eval/README.md) | 基线、原E与比赛推理提交 |
| 01_guides_and_readmes | 2026-07-21 | [S2 small-lesion ablation 20260721](../../../S2/30_results/s2_small_lesion_ablation_20260721/README.md) | 小病灶消融、Route A与FixV2 |
| 01_guides_and_readmes | 2026-07-30 | [S2 Original E Docker Reserve](../../../S2/80_submission_container/s2_original_e_reserve_20260730_r1/README.md) | 基线、原E与比赛推理提交 |
| 01_guides_and_readmes | 2026-08-14 | [S2 fixed-103 three-model aggregate](../../../S2/30_results/experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1/README.md) | S2共享代码、数据与运行资料 |
| 01_guides_and_readmes | 2026-08-23 | [S2 Fix-v3 Full-200 Attempt-11](../../../S2/30_results/experimental_unvalidated/s2_met_aug_fix_v3_full200_attempt11_20260823_r1/README.md) | FixV3与回顾性比较 |
| 01_guides_and_readmes | 2026-08-27 | [E vs FixV3 fixed-103 retrospective paired comparison](../../../S2/30_results/experimental_unvalidated/s2_e_fixv3_fixed103_retrospective_20260827_r1/README.md) | FixV3与回顾性比较 |
| 01_guides_and_readmes | 2026-08-28 | [Table 7 mutually exclusive size-stratified micro-F1](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r1/README.md) | Table7与micro-F1探索 |
| 01_guides_and_readmes | 2026-08-28 | [Table 7 mutually exclusive size-stratified micro-F1 (V100 migration)](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/README.md) | Table7与micro-F1探索 |
| 01_guides_and_readmes | 2026-08-28 | [Table 7 disjoint micro-F1: evaluation-only r3](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r3_evaluation_only/README.md) | Table7与micro-F1探索 |
| 01_guides_and_readmes | 2026-08-28 | [Table 7 disjoint micro-F1: evaluation-only r4](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/README.md) | Table7与micro-F1探索 |
| 01_guides_and_readmes | 2026-08-31 | [S2 MET-AUG B-F 125-epoch screening (2026-08-31)](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1/README.md) | B至F等预算Screen125训练 |
| 01_guides_and_readmes | 2026-09-03 | [S2 MET-AUG B/E/F Extend175 (2026-09-03 R1)](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r1/README.md) | B、E-SNFH、F独立Extend175 |
| 01_guides_and_readmes | 2026-09-03 | [S2 MET-AUG B/E/F Extend175 (2026-09-03 R2)](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r2/README.md) | B、E-SNFH、F独立Extend175 |
| 01_guides_and_readmes | 2026-09-03 | [S2 MET-AUG B/E/F Extend175 (2026-09-03 R3)](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r3/README.md) | B、E-SNFH、F独立Extend175 |
| 01_guides_and_readmes | 2026-09-03 | [S2 MET-AUG B-F 125-epoch screening (2026-08-31)](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1/remote_snapshot_20260903T105735%2B0800/README.md) | B至F等预算Screen125训练 |
| 01_guides_and_readmes | 2026-09-03 | [S2 Screen125 selected-checkpoint official MET comparison](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1/README.md) | 五路selected-checkpoint官方MET |
| 01_guides_and_readmes | 2026-09-07 | [S2 MET-AUG B/E/F Extend175 (2026-09-07 R4)](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260907_r4/README.md) | B、E-SNFH、F独立Extend175 |
| 01_guides_and_readmes | 2026-09-07 | [S2 Screen125 selected-checkpoint official MET comparison](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/README.md) | 五路selected-checkpoint官方MET |
| 01_guides_and_readmes | 2026-09-07 | [S2 Screen125 selected103 official MET recovery r3](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r3/README.md) | 五路selected-checkpoint官方MET |
| 01_guides_and_readmes | 2026-09-07 | [S2 Screen125 selected103 official MET recovery r4](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/README.md) | 五路selected-checkpoint官方MET |
| 01_guides_and_readmes | 2026-09-10 | [B/E-SNFH/F Epoch175 官方 MET 验证：2026-09-10 r1](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/README.md) | epoch175官方MET与CPU恢复 |
| 01_guides_and_readmes | 2026-09-11 | [Epoch175 CPU-only official MET recovery](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/README.md) | epoch175官方MET与CPU恢复 |
| 01_guides_and_readmes | 2026-09-14 | [Epoch175 CPU-only official MET recovery](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/README.md) | epoch175官方MET与CPU恢复 |
| 01_guides_and_readmes | undated | [BraTS2026 S2 RC Segmentation](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/README.md) | S2共享代码、数据与运行资料 |
| 01_guides_and_readmes | undated | [S2 服务器运行手册](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/docs/S2_%E6%9C%8D%E5%8A%A1%E5%99%A8%E8%BF%90%E8%A1%8C%E6%89%8B%E5%86%8C.md) | S2共享代码、数据与运行资料 |
| 01_guides_and_readmes | undated | [README.md](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/inference/README.md) | S2共享代码、数据与运行资料 |
| 01_guides_and_readmes | undated | [README.md](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/models/README.md) | S2共享代码、数据与运行资料 |
| 01_guides_and_readmes | undated | [S2 工作区](../../../S2/README.md) | S2共享代码、数据与运行资料 |
| 01_guides_and_readmes | undated | [BraTS2026 S2 RC Segmentation](../../../S2/40_documents/README.md) | S2共享代码、数据与运行资料 |
| 01_guides_and_readmes | undated | [S2 Real-Only Fixed-Split Baseline — Dataset260, 3d_fullres](../../../S2/30_results/realonly_fold0/README.md) | 基线、原E与比赛推理提交 |
| 01_guides_and_readmes | undated | [S2 内部验证评估结果](../../../S2/30_results/s2_eval_results/README.md) | 基线、原E与比赛推理提交 |
| 01_guides_and_readmes | undated | [Dataset263 real-only NYU jobs](../../../S2/50_slurm/legacy_realonly/README.md) | 基线、原E与比赛推理提交 |
| 02_plans_and_assignments | 2026-08-23 | [training_log_2026_8_20_11_01_50.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_fix_v3_full200_attempt11_20260823_r1/artifacts/nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/training_log_2026_8_20_11_01_50.txt) | FixV3与回顾性比较 |
| 02_plans_and_assignments | 2026-09-07 | [training_log_2026_9_7_19_55_33.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260907_r4/results/B/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_7_19_55_33.txt) | B、E-SNFH、F独立Extend175 |
| 02_plans_and_assignments | 2026-09-07 | [training_log_2026_9_7_19_55_33.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260907_r4/results/E_SNFH/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_7_19_55_33.txt) | B、E-SNFH、F独立Extend175 |
| 02_plans_and_assignments | 2026-09-07 | [training_log_2026_9_7_19_55_33.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260907_r4/results/F/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugScreen175ExtensionFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/training_log_2026_9_7_19_55_33.txt) | B、E-SNFH、F独立Extend175 |
| 02_plans_and_assignments | undated | [BraTS 2026 S2 MET-AUG Fix-v2 完整修复与验证方案](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/docs/MET_AUG_Fix_v1_%E5%AE%8C%E6%95%B4%E4%BF%AE%E5%A4%8D%E6%96%B9%E6%A1%88.md) | S2共享代码、数据与运行资料 |
| 02_plans_and_assignments | undated | [BraTS 2026 S2 小病灶消融对比执行计划](../../../S2/40_documents/S2_%E5%B0%8F%E7%97%85%E7%81%B6%E6%B6%88%E8%9E%8D%E5%AF%B9%E6%AF%94%E6%89%A7%E8%A1%8C%E8%AE%A1%E5%88%92.md) | 小病灶消融、Route A与FixV2 |
| 02_plans_and_assignments | undated | [S2 任务分工](../../../S2/task_assignment.md) | S2共享代码、数据与运行资料 |
| 03_methods_and_contracts | 2026-07-20 | [evaluation_contract.txt](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/official_style_eval/evaluation_contract.txt) | 基线、原E与比赛推理提交 |
| 03_methods_and_contracts | 2026-07-24 | [evaluation_contract.txt](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1/evaluation_contract.txt) | 小病灶消融、Route A与FixV2 |
| 03_methods_and_contracts | 2026-07-24 | [evaluation_contract.txt](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1_focal/evaluation_contract.txt) | 小病灶消融、Route A与FixV2 |
| 03_methods_and_contracts | 2026-07-24 | [evaluation_contract.txt](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/focal/evaluation_contract.txt) | 小病灶消融、Route A与FixV2 |
| 03_methods_and_contracts | 2026-07-30 | [nnUNet 不带样本增强版本的改进策略](../../../S2/40_documents/2026-07-30_nnUNet%E4%B8%8D%E5%B8%A6%E6%A0%B7%E6%9C%AC%E5%A2%9E%E5%BC%BA%E7%89%88%E6%9C%AC%E7%9A%84%E6%94%B9%E8%BF%9B%E7%AD%96%E7%95%A5.md) | S2共享代码、数据与运行资料 |
| 03_methods_and_contracts | 2026-07-30 | [最终样本增强策略](../../../S2/40_documents/2026-07-30_%E6%9C%80%E7%BB%88%E6%A0%B7%E6%9C%AC%E5%A2%9E%E5%BC%BA%E7%AD%96%E7%95%A5.md) | S2共享代码、数据与运行资料 |
| 03_methods_and_contracts | 2026-09-07 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/B/evaluation/evaluation_contract.txt) | 五路selected-checkpoint官方MET |
| 03_methods_and_contracts | 2026-09-07 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/E_SNFH/evaluation/evaluation_contract.txt) | 五路selected-checkpoint官方MET |
| 03_methods_and_contracts | 2026-09-07 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/F/evaluation/evaluation_contract.txt) | 五路selected-checkpoint官方MET |
| 03_methods_and_contracts | 2026-09-07 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/C40/evaluation/evaluation_contract.txt) | 五路selected-checkpoint官方MET |
| 03_methods_and_contracts | 2026-09-07 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/D/evaluation/evaluation_contract.txt) | 五路selected-checkpoint官方MET |
| 04_reports_and_evaluations | 2026-07-23 | [S2 三候选阶段性对比（2026-07-23）](../../../S2/30_results/s2_small_lesion_ablation_20260721/stage_comparison_20260723.md) | 小病灶消融、Route A与FixV2 |
| 04_reports_and_evaluations | 2026-07-24 | [S2 小病灶消融正式比较与归档](../../../S2/30_results/s2_small_lesion_ablation_20260721/final_comparison_20260724.md) | 小病灶消融、Route A与FixV2 |
| 04_reports_and_evaluations | 2026-07-27 | [gpu0_case_ids.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/gpu0_case_ids.txt) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-07-27 | [gpu1_case_ids.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/gpu1_case_ids.txt) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-07-27 | [evaluation_contract.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/B/evaluation_contract.txt) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-07-27 | [evaluation_contract.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E/evaluation_contract.txt) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-07-27 | [evaluation_contract.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E_continue_final/evaluation_contract.txt) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-07-27 | [S2 fixed-103 official-metric model selection](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/MODEL_SELECTION.md) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-08-06 | [BraTS 2026 Task 1 Validation Leaderboard 指标归档](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/README.md) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-08-06 | [字段与排序方法](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/%E5%AD%97%E6%AE%B5%E4%B8%8E%E6%8E%92%E5%BA%8F%E6%96%B9%E6%B3%95.md) | 基线、原E与比赛推理提交 |
| 04_reports_and_evaluations | 2026-08-24 | [S2 G1-r4 Endpoint Validation (2026-08-24 r1)](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/README.md) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [S2 G1-r4 Endpoint Validation (2026-08-24 r2)](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/README.md) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/real/evaluation/evaluation_contract.txt) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/synthetic/evaluation/evaluation_contract.txt) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/test26/evaluation/evaluation_contract.txt) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/B/evaluation/evaluation_contract.txt) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/E/evaluation/evaluation_contract.txt) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/FixV3/evaluation/evaluation_contract.txt) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-24 | [evaluation_contract.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/R/evaluation/evaluation_contract.txt) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-08-26 | [S2 G1-r4 Endpoint Validation Final Acceptance Report](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/FINAL_ACCEPTANCE_REPORT_20260826.md) | G1-r4补全与分割endpoint |
| 04_reports_and_evaluations | 2026-09-03 | [S2 MET-AUG B-F Screen125 Retrieval Acceptance](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1/remote_snapshot_20260903T105735%2B0800/SCREEN125_RETRIEVAL_ACCEPTANCE_20260903.md) | B至F等预算Screen125训练 |
| 04_reports_and_evaluations | 2026-09-08 | [ECNU S2 Screen125 与 Extend175 最终验收](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/acceptance/20260908T102740Z_r2/FINAL_ACCEPTANCE_20260908.md) | B、E-SNFH、F独立Extend175 |
| 04_reports_and_evaluations | 2026-09-08 | [Screen125 全部配对结果](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/acceptance/20260908T102740Z_r2/SCREEN125_PAIRED_PRIMARY_AND_FP.md) | 五路selected-checkpoint官方MET |
| 04_reports_and_evaluations | 2026-09-11 | [B/E-SNFH/F：125与175官方MET现有指标](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/operator/metrics_summary_20260911/EXISTING_OFFICIAL_MET_COMPARISON_20260911.md) | epoch175官方MET与CPU恢复 |
| 04_reports_and_evaluations | 2026-09-14 | [CPU Recovery Final Acceptance](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/FINAL_ACCEPTANCE.md) | epoch175官方MET与CPU恢复 |
| 04_reports_and_evaluations | 2026-09-14 | [B/E-SNFH/F: 125 vs 175 Official MET](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/COMPARISON_REPORT.md) | epoch175官方MET与CPU恢复 |
| 05_notes_and_other_documents | 2026-07-20 | [training_log_2026_7_19_08_12_17.txt](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/fold_0/training_log_2026_7_19_08_12_17.txt) | 基线、原E与比赛推理提交 |
| 05_notes_and_other_documents | 2026-07-20 | [training_log_2026_7_19_09_02_20.txt](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/fold_0/training_log_2026_7_19_09_02_20.txt) | 基线、原E与比赛推理提交 |
| 05_notes_and_other_documents | 2026-07-23 | [training_log_2026_7_22_10_20_06.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_timeout_20260723T1435/a1/fold_0/training_log_2026_7_22_10_20_06.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-23 | [training_log_2026_7_22_13_56_09.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_timeout_20260723T1435/a1_focal/fold_0/training_log_2026_7_22_13_56_09.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-23 | [training_log_2026_7_22_13_56_09.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_timeout_20260723T1435/focal/fold_0/training_log_2026_7_22_13_56_09.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-24 | [training_log_2026_7_22_10_20_06.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/a1/fold_0/training_log_2026_7_22_10_20_06.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-24 | [training_log_2026_7_22_13_56_09.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/a1_focal/fold_0/training_log_2026_7_22_13_56_09.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-24 | [training_log_2026_7_24_00_20_01.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/a1_focal/fold_0/training_log_2026_7_24_00_20_01.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-24 | [training_log_2026_7_22_13_56_09.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/training_log_2026_7_22_13_56_09.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-24 | [training_log_2026_7_23_21_32_28.txt](../../../S2/30_results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/training_log_2026_7_23_21_32_28.txt) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-24 | [S2 逐病例风险复核](../../../S2/30_results/s2_small_lesion_ablation_20260721/risk_review_20260724.md) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-07-27 | [training_log_2026_7_26_15_10_35.txt](../../../S2/30_results/s2_e_continue_fallback_20260726_r1/remote_snapshot_complete_20260727T0923/fold_0/training_log_2026_7_26_15_10_35.txt) | 基线、原E与比赛推理提交 |
| 05_notes_and_other_documents | 2026-07-28 | [index.html](../../../S2/30_results/s2_met_aug_fix_v2_20260728_r1/development_blind_review/index.html) | 小病灶消融、Route A与FixV2 |
| 05_notes_and_other_documents | 2026-09-11 | [125 与 175 epoch 官方 MET 原始值对比](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/existing_official_values_20260911/OFFICIAL_MET_125_175_SIDE_BY_SIDE.md) | epoch175官方MET与CPU恢复 |
| 05_notes_and_other_documents | undated | [S2 Fixed Splits](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/configs/split_info.md) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [test_locked.txt](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/data/splits/completion_warmstart/test_locked.txt) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [train_fixed.txt](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/data/splits/completion_warmstart/train_fixed.txt) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [val_fixed.txt](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/data/splits/completion_warmstart/val_fixed.txt) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [S2 Dataset](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/docs/DATASET.md) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [S2 Inference Protocol](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/docs/INFERENCE.md) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [S2_experiment.txt](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/docs/S2_experiment.txt) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [S2 Training Protocol](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/docs/TRAINING.md) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [Hyperparameter Tuning Notes](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/docs/TUNING.md) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [On-the-Fly 肿瘤增强策略](../../../S2/40_documents/ON_THE_FLY_AUGMENTATION.md) | S2共享代码、数据与运行资料 |
| 05_notes_and_other_documents | undated | [mean_metrics.txt](../../../S2/30_results/27%E3%80%82result/mean_metrics.txt) | 基线、原E与比赛推理提交 |
| 05_notes_and_other_documents | undated | [LAUNCH_ATTEMPT_01_DUAL_GPU_PENDING_SNAPSHOT.txt](../../../S2/30_results/27%E3%80%82result/runtime_attempt_evidence/LAUNCH_ATTEMPT_01_DUAL_GPU_PENDING_SNAPSHOT.txt) | 基线、原E与比赛推理提交 |
| 05_notes_and_other_documents | undated | [LAUNCH_ATTEMPT_03_GPU_ALLOCATION_SNAPSHOT.txt](../../../S2/30_results/27%E3%80%82result/runtime_attempt_evidence/LAUNCH_ATTEMPT_03_GPU_ALLOCATION_SNAPSHOT.txt) | 基线、原E与比赛推理提交 |
| 05_notes_and_other_documents | undated | [training_log_2026_7_10_05_32_06.txt](../../../S2/30_results/realonly_fold0/training_log_2026_7_10_05_32_06.txt) | 基线、原E与比赛推理提交 |
| 06_failures_and_recovery | undated | [LAUNCH_ATTEMPT_02_FAILED_SACCT.txt](../../../S2/30_results/27%E3%80%82result/runtime_attempt_evidence/LAUNCH_ATTEMPT_02_FAILED_SACCT.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-20 | [SHA256SUMS.txt](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/checksums/SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-20 | [evaluation_environment.txt](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/official_style_eval/evaluation_environment.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-24 | [evaluation_environment.txt](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1/evaluation_environment.txt) | 小病灶消融、Route A与FixV2 |
| 07_environment_hashes | 2026-07-24 | [evaluation_environment.txt](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1_focal/evaluation_environment.txt) | 小病灶消融、Route A与FixV2 |
| 07_environment_hashes | 2026-07-24 | [evaluation_environment.txt](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/focal/evaluation_environment.txt) | 小病灶消融、Route A与FixV2 |
| 07_environment_hashes | 2026-07-27 | [RUNTIME_SNAPSHOT_SHA256SUMS.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/RUNTIME_SNAPSHOT_SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-27 | [evaluation_environment.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/B/evaluation_environment.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-27 | [evaluation_environment.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E/evaluation_environment.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-27 | [evaluation_environment.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E_continue_final/evaluation_environment.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-27 | [MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/MODEL_SELECTION_ARTIFACT_SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-27 | [LOCAL_SHA256SUMS.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/source_transfer/LOCAL_SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-27 | [REMOTE_SHA256SUMS.txt](../../../S2/30_results/s2_official_validation_inference_20260727_r1/source_transfer/REMOTE_SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-27 | [LOCAL_SHA256SUMS.txt](../../../S2/30_results/s2_official_validation_transfer_20260727_r1/LOCAL_SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-07-30 | [Validated S2 userspace stack (Python 3.10). Install the cluster-compatible](../../../S2/80_submission_container/s2_original_e_reserve_20260730_r1/requirements-inference.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-08-06 | [SHA256SUMS.txt](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | 2026-08-14 | [OUTPUT_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1/OUTPUT_SHA256SUMS.txt) | S2共享代码、数据与运行资料 |
| 07_environment_hashes | 2026-08-23 | [LOCAL_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_fix_v3_full200_attempt11_20260823_r1/LOCAL_SHA256SUMS.txt) | FixV3与回顾性比较 |
| 07_environment_hashes | 2026-08-23 | [REMOTE_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_fix_v3_full200_attempt11_20260823_r1/provenance/REMOTE_SHA256SUMS.txt) | FixV3与回顾性比较 |
| 07_environment_hashes | 2026-08-24 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/STATIC_SHA256SUMS.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [LOCAL_LIGHTWEIGHT_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/LOCAL_LIGHTWEIGHT_SHA256SUMS.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/STATIC_SHA256SUMS.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/real/evaluation/evaluation_environment.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/synthetic/evaluation/evaluation_environment.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/test26/evaluation/evaluation_environment.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/B/evaluation/evaluation_environment.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/E/evaluation/evaluation_environment.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/FixV3/evaluation/evaluation_environment.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-24 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/R/evaluation/evaluation_environment.txt) | G1-r4补全与分割endpoint |
| 07_environment_hashes | 2026-08-28 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r1/STATIC_SHA256SUMS.txt) | Table7与micro-F1探索 |
| 07_environment_hashes | 2026-08-28 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/STATIC_SHA256SUMS.txt) | Table7与micro-F1探索 |
| 07_environment_hashes | 2026-08-28 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/official_reproduction/evaluation_environment.txt) | Table7与micro-F1探索 |
| 07_environment_hashes | 2026-08-28 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r3_evaluation_only/STATIC_SHA256SUMS.txt) | Table7与micro-F1探索 |
| 07_environment_hashes | 2026-08-28 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/STATIC_SHA256SUMS.txt) | Table7与micro-F1探索 |
| 07_environment_hashes | 2026-08-28 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/official_reproduction/evaluation_environment.txt) | Table7与micro-F1探索 |
| 07_environment_hashes | 2026-09-03 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r1/STATIC_SHA256SUMS.txt) | B、E-SNFH、F独立Extend175 |
| 07_environment_hashes | 2026-09-03 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r2/STATIC_SHA256SUMS.txt) | B、E-SNFH、F独立Extend175 |
| 07_environment_hashes | 2026-09-03 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260903_r3/STATIC_SHA256SUMS.txt) | B、E-SNFH、F独立Extend175 |
| 07_environment_hashes | 2026-09-03 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1/STATIC_SHA256SUMS.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_extend175_20260907_r4/STATIC_SHA256SUMS.txt) | B、E-SNFH、F独立Extend175 |
| 07_environment_hashes | 2026-09-07 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/STATIC_SHA256SUMS.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/B/evaluation/evaluation_environment.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/E_SNFH/evaluation/evaluation_environment.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/F/evaluation/evaluation_environment.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r3/STATIC_SHA256SUMS.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/STATIC_SHA256SUMS.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/C40/evaluation/evaluation_environment.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-07 | [evaluation_environment.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/D/evaluation/evaluation_environment.txt) | 五路selected-checkpoint官方MET |
| 07_environment_hashes | 2026-09-10 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/STATIC_SHA256SUMS.txt) | epoch175官方MET与CPU恢复 |
| 07_environment_hashes | 2026-09-11 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/STATIC_SHA256SUMS.txt) | epoch175官方MET与CPU恢复 |
| 07_environment_hashes | 2026-09-14 | [STATIC_SHA256SUMS.txt](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/STATIC_SHA256SUMS.txt) | epoch175官方MET与CPU恢复 |
| 07_environment_hashes | undated | [Validated S2 userspace stack (Python 3.10). Install the cluster-compatible](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/requirements.txt) | S2共享代码、数据与运行资料 |
| 07_environment_hashes | undated | [Official BraTS 2026 segmentation evaluation package.](../../../S2/11_project_brats2026_s2_rc_v1_0/repository/requirements_eval.txt) | S2共享代码、数据与运行资料 |
| 07_environment_hashes | undated | [SHA256SUMS.txt](../../../S2/30_results/27%E3%80%82result/SHA256SUMS.txt) | 基线、原E与比赛推理提交 |
| 07_environment_hashes | undated | [H20 云主机独立环境。PyTorch 2.7.1/cu128 由 setup 脚本先单独安装，](../../../S2/50_slurm/requirements-h20.txt) | S2共享代码、数据与运行资料 |
| 08_tables | 2026-07-20 | [leaderboard_metrics.csv](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/official_style_eval/leaderboard_metrics.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-20 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_completion_dataset264_t2w_20260720/official_style_eval/nnunet_to_source_id.tsv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-24 | [leaderboard_metrics.csv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1/leaderboard_metrics.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-24 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1/nnunet_to_source_id.tsv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-24 | [leaderboard_metrics.csv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1_focal/leaderboard_metrics.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-24 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/a1_focal/nnunet_to_source_id.tsv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-24 | [leaderboard_metrics.csv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/focal/leaderboard_metrics.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-24 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/focal/nnunet_to_source_id.tsv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-24 | [mapping_fixed.tsv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/tools/mapping_fixed.tsv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-24 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126/tools/nnunet_to_source_id.tsv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-26 | [manual_review_decisions.csv](../../../S2/30_results/s2_met_aug_route_a_20260726_r4/gate2_run/manual_review_decisions.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-26 | [manual_review_template.csv](../../../S2/30_results/s2_met_aug_route_a_20260726_r4/gate2_run/manual_review_template.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-26 | [manual_review_decisions.csv](../../../S2/30_results/s2_met_aug_route_a_20260726_r5/gate2_run/manual_review_decisions.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-26 | [manual_review_template.csv](../../../S2/30_results/s2_met_aug_route_a_20260726_r5/gate2_run/manual_review_template.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-27 | [gpu0_output_manifest.tsv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/gpu0_output_manifest.tsv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [gpu1_output_manifest.tsv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/gpu1_output_manifest.tsv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [official_prediction_manifest.csv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/official_prediction_manifest.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [official_validation_input_manifest.csv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/official_validation_input_manifest.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [leaderboard_metrics.csv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/B/leaderboard_metrics.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/B/nnunet_to_source_id.tsv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [leaderboard_metrics.csv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E/leaderboard_metrics.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E/nnunet_to_source_id.tsv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [leaderboard_metrics.csv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E_continue_final/leaderboard_metrics.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/E_continue_final/nnunet_to_source_id.tsv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [OFFICIAL_METRICS_AGGREGATE.csv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/OFFICIAL_METRICS_AGGREGATE.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-27 | [mapping_fixed.tsv](../../../S2/30_results/s2_official_validation_inference_20260727_r1/selection_evidence/tools/mapping_fixed.tsv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-07-28 | [DEVELOPMENT_BLINDED_REVIEW_DECISIONS_ATTEMPT_01.csv](../../../S2/30_results/s2_met_aug_fix_v2_20260728_r1/calibration/DEVELOPMENT_BLINDED_REVIEW_DECISIONS_ATTEMPT_01.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-28 | [DEVELOPMENT_BLINDED_REVIEW_DECISIONS_ATTEMPT_02.csv](../../../S2/30_results/s2_met_aug_fix_v2_20260728_r1/calibration/DEVELOPMENT_BLINDED_REVIEW_DECISIONS_ATTEMPT_02.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-28 | [QC_HOLDOUT_BLINDED_REVIEW_DECISIONS_ATTEMPT_01.csv](../../../S2/30_results/s2_met_aug_fix_v2_20260728_r1/calibration/QC_HOLDOUT_BLINDED_REVIEW_DECISIONS_ATTEMPT_01.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-28 | [blinded_manual_review_template.csv](../../../S2/30_results/s2_met_aug_fix_v2_20260728_r1/development_run_attempt_01/blinded_manual_review_template.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-28 | [blinded_manual_review_template.csv](../../../S2/30_results/s2_met_aug_fix_v2_20260728_r1/qc_blind_review_attempt_02/blinded_manual_review_template.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-28 | [blinded_manual_review_template.csv](../../../S2/30_results/s2_met_aug_fix_v2_20260728_r1/qc_holdout_run_attempt_02/blinded_manual_review_template.csv) | 小病灶消融、Route A与FixV2 |
| 08_tables | 2026-07-28 | [S2_Task1_validation_179_original_E.manifest.csv](../../../S2/30_results/s2_official_validation_submission_20260728_r1/S2_Task1_validation_179_original_E.manifest.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [BraTS2026_Task1_Valid_Leaderboard_Top75_By_Metric_20260806.xlsx](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/BraTS2026_Task1_Valid_Leaderboard_Top75_By_Metric_20260806.xlsx) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_dsc_mean_et.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_dsc_mean_et.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_dsc_mean_rc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_dsc_mean_rc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_dsc_mean_tc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_dsc_mean_tc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_dsc_mean_wt.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_dsc_mean_wt.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_nsd_mean_et.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_nsd_mean_et.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_nsd_mean_rc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_nsd_mean_rc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_nsd_mean_tc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_nsd_mean_tc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Lesionwise_nsd_mean_wt.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Lesionwise_nsd_mean_wt.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_f1_et.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_f1_et.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_f1_rc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_f1_rc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_f1_tc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_f1_tc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_f1_wt.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_f1_wt.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fn_et.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fn_et.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fn_rc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fn_rc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fn_tc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fn_tc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fn_wt.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fn_wt.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fp_et.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fp_et.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fp_rc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fp_rc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fp_tc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fp_tc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_fp_wt.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_fp_wt.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_tp_et.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_tp_et.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_tp_rc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_tp_rc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_tp_tc.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_tp_tc.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-06 | [Small_instance_tp_wt.csv](../../../S2/30_results/s2_validation_leaderboard_20260806_r1/csv/Small_instance_tp_wt.csv) | 基线、原E与比赛推理提交 |
| 08_tables | 2026-08-14 | [aggregate_statistics_three_models.csv](../../../S2/30_results/experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1/aggregate_statistics_three_models.csv) | S2共享代码、数据与运行资料 |
| 08_tables | 2026-08-14 | [core_metrics_three_models.csv](../../../S2/30_results/experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1/core_metrics_three_models.csv) | S2共享代码、数据与运行资料 |
| 08_tables | 2026-08-14 | [per_case_three_models_long.csv](../../../S2/30_results/experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1/per_case_three_models_long.csv) | S2共享代码、数据与运行资料 |
| 08_tables | 2026-08-14 | [three_model_fixed103_aggregate.xlsx](../../../S2/30_results/experimental_unvalidated_three_model_fixed103_aggregate_20260814_r1/three_model_fixed103_aggregate.xlsx) | S2共享代码、数据与运行资料 |
| 08_tables | 2026-08-24 | [FIXED103_REAL_SYNTHETIC_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/manifests/FIXED103_REAL_SYNTHETIC_CASES.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [TEST26_LOCKED_MISSING_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/manifests/TEST26_LOCKED_MISSING_T2W_CASES.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [VAL27_MISSING_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r1/manifests/VAL27_MISSING_T2W_CASES.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/paired_comparison/METRIC_RANKINGS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/paired_comparison/PAIRED_CASE_METRICS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/paired_comparison/PAIRWISE_BOOTSTRAP.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/real/evaluation/leaderboard_metrics.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [input_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/real/input_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/synthetic/evaluation/leaderboard_metrics.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [input_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/fixed103/synthetic/input_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/test26/evaluation/leaderboard_metrics.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [input_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/test26/input_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [geometry_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/test26/synthesis_r4/geometry_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [spatial_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/test26/synthesis_r4/spatial_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [input_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/input_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/B/evaluation/leaderboard_metrics.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/E/evaluation/leaderboard_metrics.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/FixV3/evaluation/leaderboard_metrics.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/models/R/evaluation/leaderboard_metrics.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/paired_comparison/METRIC_RANKINGS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/paired_comparison/PAIRED_CASE_METRICS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/segmentation/paired_comparison/PAIRWISE_BOOTSTRAP.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [geometry_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/synthesis_r4/geometry_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [spatial_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/cohorts/val27/synthesis_r4/spatial_audit.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [FIXED103_REAL_SYNTHETIC_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/manifests/FIXED103_REAL_SYNTHETIC_CASES.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [TEST26_LOCKED_MISSING_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/manifests/TEST26_LOCKED_MISSING_T2W_CASES.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [VAL27_MISSING_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/manifests/VAL27_MISSING_T2W_CASES.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [FIXED103_PRIMARY_PAIRED_STATISTICS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/summary/FIXED103_PRIMARY_PAIRED_STATISTICS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [TEST26_PRIMARY_METRIC_MEANS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/summary/TEST26_PRIMARY_METRIC_MEANS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [VAL27_PRIMARY_MEAN_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/summary/VAL27_PRIMARY_MEAN_RANKINGS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-24 | [VAL27_PRIMARY_PAIRED_STATISTICS.csv](../../../S2/30_results/experimental_unvalidated/s2_g1r4_endpoint_validation_20260824_r2/summary/VAL27_PRIMARY_PAIRED_STATISTICS.csv) | G1-r4补全与分割endpoint |
| 08_tables | 2026-08-27 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_e_fixv3_fixed103_retrospective_20260827_r1/METRIC_RANKINGS.csv) | FixV3与回顾性比较 |
| 08_tables | 2026-08-27 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_e_fixv3_fixed103_retrospective_20260827_r1/PAIRED_CASE_METRICS.csv) | FixV3与回顾性比较 |
| 08_tables | 2026-08-27 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_e_fixv3_fixed103_retrospective_20260827_r1/PAIRWISE_BOOTSTRAP.csv) | FixV3与回顾性比较 |
| 08_tables | 2026-08-28 | [case_level_disjoint_counts.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/disjoint_microf1/case_level_disjoint_counts.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [component_events.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/disjoint_microf1/component_events.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [disjoint_microf1_summary.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/disjoint_microf1/disjoint_microf1_summary.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [leaderboard_metrics_threshold27.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/official_reproduction/leaderboard_metrics_threshold27.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [leaderboard_metrics_threshold50.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r2_v100/official_reproduction/leaderboard_metrics_threshold50.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [case_level_disjoint_counts.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/disjoint_microf1/case_level_disjoint_counts.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [component_events.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/disjoint_microf1/component_events.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [disjoint_microf1_summary.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/disjoint_microf1/disjoint_microf1_summary.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [CASE_ID_MAPPING_AND_SYMLINKS.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/evidence/CASE_ID_MAPPING_AND_SYMLINKS.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [leaderboard_metrics_threshold27.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/official_reproduction/leaderboard_metrics_threshold27.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-08-28 | [leaderboard_metrics_threshold50.csv](../../../S2/30_results/experimental_unvalidated/s2_table7_disjoint_microf1_20260828_r4_evaluation_only/official_reproduction/leaderboard_metrics_threshold50.csv) | Table7与micro-F1探索 |
| 08_tables | 2026-09-03 | [SCREEN125_EPOCH125_COMPARISON.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_screen125_20260831_r1/remote_snapshot_20260903T105735%2B0800/SCREEN125_EPOCH125_COMPARISON.csv) | B至F等预算Screen125训练 |
| 08_tables | 2026-09-03 | [FIXED103_REAL_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260903_r1/manifests/FIXED103_REAL_T2W_CASES.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [input_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/input_audit.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/B/evaluation/leaderboard_metrics.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/E_SNFH/evaluation/leaderboard_metrics.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/cohort/fixed103/models/F/evaluation/leaderboard_metrics.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [FIXED103_REAL_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r2/manifests/FIXED103_REAL_T2W_CASES.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [FIXED103_REAL_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r3/manifests/FIXED103_REAL_T2W_CASES.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/C40/evaluation/leaderboard_metrics.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/models/D/evaluation/leaderboard_metrics.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/paired_comparison/METRIC_RANKINGS.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/paired_comparison/PAIRED_CASE_METRICS.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/cohort/fixed103/paired_comparison/PAIRWISE_BOOTSTRAP.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-07 | [FIXED103_REAL_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/manifests/FIXED103_REAL_T2W_CASES.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-08 | [EXTEND175_CHECKPOINT_SELECTION.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/acceptance/20260908T102740Z_r2/EXTEND175_CHECKPOINT_SELECTION.csv) | B、E-SNFH、F独立Extend175 |
| 08_tables | 2026-09-08 | [SCREEN125_METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/acceptance/20260908T102740Z_r2/SCREEN125_METRIC_RANKINGS.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-08 | [SCREEN125_OFFICIAL_MET_MEANS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/acceptance/20260908T102740Z_r2/SCREEN125_OFFICIAL_MET_MEANS.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-08 | [SCREEN125_PAIRED_ALL_720.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_btof_selected103_official_met_20260907_r4/acceptance/20260908T102740Z_r2/SCREEN125_PAIRED_ALL_720.csv) | 五路selected-checkpoint官方MET |
| 08_tables | 2026-09-10 | [FIXED103_REAL_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/manifests/FIXED103_REAL_T2W_CASES.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-11 | [OFFICIAL_MET_125_175_MEANS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260910_r1/operator/metrics_summary_20260911/OFFICIAL_MET_125_175_MEANS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-11 | [OFFICIAL_MET_125_175_SIDE_BY_SIDE.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/existing_official_values_20260911/OFFICIAL_MET_125_175_SIDE_BY_SIDE.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/B_175_vs125/METRIC_RANKINGS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/B_175_vs125/PAIRED_CASE_METRICS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/B_175_vs125/PAIRWISE_BOOTSTRAP.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/E_SNFH_175_vs125/METRIC_RANKINGS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/E_SNFH_175_vs125/PAIRED_CASE_METRICS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/E_SNFH_175_vs125/PAIRWISE_BOOTSTRAP.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/F_175_vs125/METRIC_RANKINGS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/F_175_vs125/PAIRED_CASE_METRICS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/F_175_vs125/PAIRWISE_BOOTSTRAP.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [OFFICIAL_MET_125_VS175_VALUES_AND_PAIRED.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/OFFICIAL_MET_125_VS175_VALUES_AND_PAIRED.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [METRIC_RANKINGS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/epoch175_cross_route/METRIC_RANKINGS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRED_CASE_METRICS.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/epoch175_cross_route/PAIRED_CASE_METRICS.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [PAIRWISE_BOOTSTRAP.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/results/epoch175_cross_route/PAIRWISE_BOOTSTRAP.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/B/epoch125/leaderboard_metrics.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/B/epoch175/leaderboard_metrics.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/E_SNFH/epoch125/leaderboard_metrics.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/E_SNFH/epoch175/leaderboard_metrics.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/F/epoch125/leaderboard_metrics.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [leaderboard_metrics.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/F/epoch175/leaderboard_metrics.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [FIXED103_REAL_T2W_CASES.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/FIXED103_REAL_T2W_CASES.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | 2026-09-14 | [input_audit.csv](../../../S2/30_results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/operator/snapshot_20260914T083326Z/remote_exact/sources/input_audit.csv) | epoch175官方MET与CPU恢复 |
| 08_tables | undated | [lesion_pool.csv](../../../S2/40_documents/lesion_pool.csv) | S2共享代码、数据与运行资料 |
| 08_tables | undated | [PREDICTION_MANIFEST.csv](../../../S2/30_results/27%E3%80%82result/PREDICTION_MANIFEST.csv) | 基线、原E与比赛推理提交 |
| 08_tables | undated | [VAL27_CASE_MANIFEST.csv](../../../S2/30_results/27%E3%80%82result/evidence/VAL27_CASE_MANIFEST.csv) | 基线、原E与比赛推理提交 |
| 08_tables | undated | [VAL27_PREDICTION_AUDIT.csv](../../../S2/30_results/27%E3%80%82result/evidence/VAL27_PREDICTION_AUDIT.csv) | 基线、原E与比赛推理提交 |
| 08_tables | undated | [leaderboard_metrics.csv](../../../S2/30_results/27%E3%80%82result/leaderboard_metrics.csv) | 基线、原E与比赛推理提交 |
| 08_tables | undated | [synthetic_accepted_evaluation_manifest_run_3104668.csv](../../../S2/30_results/27%E3%80%82result/runtime_attempt_evidence/synthetic_accepted_evaluation_manifest_run_3104668.csv) | 基线、原E与比赛推理提交 |
| 08_tables | undated | [official_submission_manifest.csv](../../../S2/30_results/official_submission_manifest.csv) | 基线、原E与比赛推理提交 |
| 08_tables | undated | [leaderboard_metrics.csv](../../../S2/30_results/s2_eval_results/leaderboard_metrics.csv) | 基线、原E与比赛推理提交 |
| 08_tables | undated | [nnunet_to_source_id.tsv](../../../S2/30_results/s2_eval_results/nnunet_to_source_id.tsv) | 基线、原E与比赛推理提交 |
| 09_references_and_notebooks | undated | [BraTS 2025.pdf](../../../S2/40_documents/BraTS%202025.pdf) | S2共享代码、数据与运行资料 |
