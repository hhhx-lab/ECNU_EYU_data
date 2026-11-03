# 两套外部METS模型同fixed103服务器横评计划

制定日期：2026-11-04，Asia/Shanghai。

用户已选定2025 METS第一名镜像与2026 Bio-Intel，并要求先拟定服务器执行计划。本目录是计划，不是已经实施的推理批次。当前未连接ECNU、下载资产、安装环境、创建目录或提交Slurm作业；没有JobID或外部模型新分数。

- [完整执行计划](PLAN.md)：比较对象、同103合同、资产/容量/容器门、分阶段作业、统一评分及验收。
- [机器可读计划](PLAN.json)：冻结的候选身份、数据/基线SHA、待核项、资源建议和依赖图；不是可直接执行的提交器。
- [计划核验](VERIFICATION.json)：只验本地证据、链接和计划一致性，不代表远端资源或模型smoke通过。
- [前置可行性调查](../../07_references/20261104_external_models_fixed103_feasibility/REPORT.md)：公开源审计与训练暴露聚合计数。

当前阶段：`plan_ready_server_execution_not_started`。资产/环境准备及提交须按用户后续执行确认推进；新增依赖安装必须先列清单取得授权。计划不恢复监控，不读locked test，不替换比赛模型。

研究状态：`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。
