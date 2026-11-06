# 2025 METS全量103预测本地回传

本目录保存已完成作业4310575的103份分割预测，合计4260855 bytes，位于`prediction/`。用户2026-11-06明确授权此次NIfTI结果回传；没有原始MRI、checkpoint、镜像或概率图。

来源：远端外部模型批次`attempts/full103_2025_r1/predictions/EXT2025_MET_GLI`。allocation/batch COMPLETED0:0，18分27秒；完成manifest SHA `c0ffe07f452aa911d7e1b17128add3db25c443ddfb64fe45e7dac2bb84a34717`。传输前远端与完成清单、传输后本地每份文件大小/SHA均匹配，见[验收](DOWNLOAD_ACCEPTANCE.json)。本目录不覆盖旧结果。

用途：与K1 selected175、原比赛E epoch200同fixed103验证集统一MET评分。不是锁定测试集；推理和传输工程通过不代表新评分已产生或模型质量优越。原退出清理Errno39告警保留在服务器日志/状态审计。研究状态保持experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed。
