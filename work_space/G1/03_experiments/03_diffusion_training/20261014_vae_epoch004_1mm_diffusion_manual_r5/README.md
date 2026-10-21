# G1 epoch004 1mm Diffusion 独立 R5

最新终态优先：[预处理mask覆盖失败](PREPARE_FAILED_MASK_COVERAGE_20261015.md)。4086739 FAILED1:0；四GPU4069916-19均CANCELLED零运行，无训练checkpoint。heartbeat按失败规则已PAUSED，旧失败/部分输出保留，不原位重跑或自动新attempt。下文提交成功仅是历史状态。

当前作业状态：[2026-10-15预处理替换成功](PREPARE_REPLACEMENT_STATUS_20261015.md)。新CPU4086739已在cpu_192G释放，四原GPU4069916-19均afterok:4086739，旧CPU4069915主动取消；最新回执为PENDING，不称训练已运行。后续不得再次运行原submit或替换操作器。

用户授权修复并独立新批次提交；R3预检120秒超时、R4操作器函数遮蔽及容量保守膨胀失败均保留，无此前job ID。线程受控CPU预检已21.3秒七测试通过；R4预检再次通过后按全视野输出再加10%开销得到231590329172 bytes，实际余224164577280 bytes而停止，未sbatch。

R5保留同样的全视野未压缩数据上界，改用默认zlib压缩最坏开销的可计算上界，不假设有利压缩率；另留1GiB header/metadata、12GiB全四路checkpoint、四worker双份最大临时输出及4GiB额外余量。无清理/减少病例/降低精度或训练参数。数据实际写入仍须空间成立。

验收豁免不改写为pass。train823/val103共926独立严格1mm，真实T1N/T1C/T2F共同mask，test104锁定。一个CPU24h预处理和四afterok单A10048h训练，各200000成功更新、seed20261013、64³/batch16 AMP/eager/nonDDP、AdamWlr2e-4beta1=.9。消费SHA、finite/geometry/legal label、完整checkpoint和安全TERM保持，活动37源码和旧失败不改。

仅experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。P0/P1人工/P2独立smoke等待waived_by_user，不证明质量优势。
