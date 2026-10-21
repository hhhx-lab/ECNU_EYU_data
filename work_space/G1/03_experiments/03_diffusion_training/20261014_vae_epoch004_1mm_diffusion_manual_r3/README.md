# G1 epoch004 1mm Diffusion 独立 R3

用户于2026-10-14明确要求不再等待验收，人工授权直接提交训练。该授权记录为验收豁免，不伪造自动、逐例人工或科学通过。R2 P0作业4058680超时，746/1030病例及3730/5150文件部分记录与原审计永久保留。

本批次采用独立补充合同：跳过单独P0全量复验、P1人工验收、P2单独smoke及人工门；预处理实际消费数据时仍拒绝缺文件、非finite、非法标签、错位、SHA漂移和空间不足。先提交一个CPU预处理，再提交四个依赖其成功退出的独立单A100训练。依赖入队不等于训练已运行。

源为epoch004原生空间重建r2；共同脑mask来自同病例真实T1N/T1C/T2F robust非零union与holefill，不从正值重建背景推定。仅train823/val103生成严格1mm，test104不读取影像、不评分。四模态分别随机初始化，seed20261013、64³、batch16、AMP、非DDP/eager、固定AdamW lr2e-4、beta1=0.9，各200000成功更新；不套用S2的175epoch。

原37份活动源码不改；独立运行包装器补充数据消费身份、每5000更新完整checkpoint、RNG/MONAI随机状态/采样顺序和batch位置、安全TERM。后续固定val103选点仍按原方案执行，本次未生成评价结果。VAE重建、1mm和新版优化器是组合变化，只能exploratory。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
