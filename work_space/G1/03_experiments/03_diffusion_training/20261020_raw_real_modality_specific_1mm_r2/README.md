# 原始真实影像模态独立训练 R2

用户于2026-10-20明确授权“修正测试后核验并提交 R2”。R1远端预检29CPU测试28pass/1error，safeTERM CPU double模拟CUDA可用后触发AdamW CUDA graph驱动调用；零sbatch/零job。R1冻结包、预约、原失败永久保留。R2仅给CPU测试增加is_current_stream_capturing=False模拟；生产训练、预处理、数据身份程序和原37上游字节不变，正式单A100检查不降低。

## 数据与训练

| 模态 | train | val | 输入 |
|---|---:|---:|---|
| T1C / T1N / T2F | 1035 | 130 | master通过原QC病例的原始真实图 |
| T2W | 823 | 103 | authentic真实T2W，缺失病例全部剔除该模态 |

原patient-group seed42 split保持，test130只登记不读图、不评分。无VAE自重建或缺失T2W补全。R1来源审核1165例5586文件、29612856946bytes SHA/header零错误；authentic926对照历史source SHA，missing239无旧字节基线但绑定新原图SHA及归档几何，两例corrected seg overlay保留。R2复用同一冻结源记录，不重扫测试图、不改原数据。

预处理为完整原FOV、T1N/T1C/T2F共同统计mask、整例zscore和原physical_spacing_v2严格1mm。统一量化丢类warning记录体素/mm3，不补点；真实来源、几何、finite、非法标签、SHA错误逐例收集FAILED不出COMPLETE，不自动exclude。每模态独立lesion CSV/marker/index/source身份，缺T2W不访问占位/补全图。训练各200000成功optimizer更新，64cube batch16 AMP eager nonDDP seed20261013 AdamW2e-4 beta1.9；5k/final/safeTERM保存network/optimizer/scaler/RNG-MONAI/permutation/batchcursor，USR1@300，无自动失败重投。

独立CPU8核90G24h及四A100各8核64G48h afterok CPU，唯一held提交、逐job身份/资源/日志/依赖/源码审计全部通过后再释放。PENDING非RUNNING，训练完成须COMPLETED0:0与COMPLETE step200000。只做当次轻量预检/回执，不下载图像或模型、不安装/Git/删旧数据、不改R9/评价或恢复PAUSED monitor。后续用户手动查询。

原图与三路样本扩展是两个变化；后续与R9比较须共同val103及新增val27分列，不称单因素改善。生成质量、最佳模型选择、人工审核与nnU-Net等价尚未运行或实测pass；本次只提交训练。

状态见[STATUS](STATUS_20261020.md)，恢复授权与父失败见plan。artifact_status=experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed。
