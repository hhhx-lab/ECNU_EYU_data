# 原始真实影像、模态独立病例集 Diffusion R1

用户于2026-10-20授权新增实验，并明确使用原始真实影像，不做VAE自重建、不使用缺失T2W补全结果。本批次不修改R9训练或正在进行的R9最佳模型评价。

| 模态 | train | val | 输入 |
|---|---:|---:|---|
| T1C / T1N / T2F | 1035 | 130 | master通过原QC病例的真实原图 |
| T2W | 823 | 103 | master中authentic子集，排除所有fake_or_broken |

保留原patient-group划分，seed42；locked test130仅登记身份，不读影像、不评分。原始非法label6的既定QC排除不改变，纠正标签overlay需绑定来源。新增错误不自动exclude、不重新划分。

## 执行合同

1. 核验master/realonly/raw manifest SHA和患者split，映射远端原始真实路径，检查真实目标、header、容量与资源。禁止把重建或合成目录作为原图来源；缺失T2W不访问该模态文件。
2. 预处理1165例train/val：真实T1N/T1C/T2F共同统计mask，整例zscore，完整原FOV，原physical_spacing_v2严格1mm算法。目标分辨率丢类采用uniform_original_quantization_warning_v1，记录原生/目标体素和mm3，不补点；身份/finite/geometry/非法标签/SHA错误逐例收集后FAILED，不产生COMPLETE。
3. 生成每模态独立lesions CSV和病例/marker/索引/源文件身份。CPU成功后四单A100训练各200000次成功optimizer更新，64cube/batch16/AMP/eager/nonDDP/seed20261013/AdamW2e-4。5k/final/safeTERM包含network、optimizer、scaler、RNG-MONAI、permutation及batchcursor，USR1@300。失败不自动重提、改batch或修标签。
4. 仅一次held提交与独立逐job资源/依赖/脚本审计后释放；CPU8核90G24h，四GPU各8核64G48h且afterok CPU。PENDING不称RUNNING，完成须COMPLETED0:0及COMPLETE step200000。用户选择手动查询，不恢复heartbeat。

原37份上游代码保持字节一致；包装层另立原图数据身份。新批次独立全量工程验收在运行入口执行；人工质量和nnU-Net等价未验证，不沿用R9 waiver充当新pass。最佳模型选择/生成质量评价不包含于本次训练提交。

## 比较边界

R9仅用823/103完整病例的VAE自重建图，不是用补全缺失T2W的全集。新批次同时改变图像来源和三模态样本规模，不是单因素对照。后续与R9比较必须报告共同val103及三模态新增val27的独立结果，不将val130与val103混为相同分母。

当前操作与证据见[STATUS](STATUS_20261020.md)。全部产物为experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed；工程pass不是生成质量/科学优越性pass。
