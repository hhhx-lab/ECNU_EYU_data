# 两套外部METS模型同fixed103服务器横评计划

制定日期：2026-11-04，Asia/Shanghai。

最新实施：修正路径后的独立R2已获用户确认，但本次只到VPN启动，归属未确认而停止，未进入SSH/Conda。未调用stop；一次只读status确认aTrust logout/tunnelUnavailable，SSO密码表单未完成，不推断密码/账号故障。用户需完成学校正常浏览器登录并确认接续同一尚未远端执行的R2；无新安装/作业/分数或SIF重试，见[接入失败分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/ENVIRONMENT_R4_OFFICIAL_R2_ACCESS_FAILURE.json)。以下R2待首次授权为历史。

最新实施：用户“继续”确认官方完整URL预检，但操作器误要求R1旧失败marker路径，在调用Conda之前停止；不是依赖/模型/GPU失败，未创建R4或七作业。自建VPN已关闭，未取得本次远端UTC/最新存储/完整身份门。独立R2仅修历史标记与审计路径，60本地测试通过，新的远端尝试待确认；SIF超时继续保留未重试。详见[官方预检分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/ENVIRONMENT_R4_OFFICIAL_FAILURE_ANALYSIS.json)。下述首次官方源待确认是历史。

最新实施：R4已获用户授权但预检未通过，实际Conda别名指向清华镜像，缺connected-components-3d4.1.1；旧R1-R3与失败核验未变，未创建/安装R4。官方API访问失败，不等于官方同版本缺失；已请求一次官方完整URL独立预检确认，未执行。51本地测试pass非实际兼容环境/模型通过。模型中转与Mac专属清理成功、SIF展开超时保留，七作业未创建。下列“R4待确认”属之前历史，此次待确认仅指新的官方源独立预检；详见[执行STATUS](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/STATUS_20261104.md)。原PLAN/VERIFICATION不改写。

用户已选定2025 METS第一名镜像与2026 Bio-Intel，并要求先拟定服务器执行计划。本目录的PLAN/VERIFICATION保留制定时快照，不是实际提交回执；其“未执行”描述仅适用于制定当时。用户随后授权执行、独立依赖安装、指定资产Mac临时中转、独立环境R2/R3修正及一次受控断点接续；实际已接入ECNU并创建本批根/环境。R3兼容pandas/NumPy成功，但其余三个间接依赖源码构建失败，独立R4尚待确认。15资产7,281,531,462bytes已完成两轮远端SHA，本次Mac临时目录已清除、自建VPN已关闭；七权重及metadata/OCI diff_id核验完成，但原生SIF的sandbox展开超时失败，仍没有JobID或外部模型新分数。实时状态见[独立执行批次STATUS](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/STATUS_20261104.md)。

- [完整执行计划](PLAN.md)：比较对象、同103合同、资产/容量/容器门、分阶段作业、统一评分及验收。
- [机器可读计划](PLAN.json)：冻结的候选身份、数据/基线SHA、待核项、资源建议和依赖图；不是可直接执行的提交器。
- [计划核验](VERIFICATION.json)：只验本地证据、链接和计划一致性，不代表远端资源或模型smoke通过。
- [前置可行性调查](../../07_references/20261104_external_models_fixed103_feasibility/REPORT.md)：公开源审计与训练暴露聚合计数。

当前实施阶段：`S0_TRANSFER_ACCEPTED_MAC_CLEANED_ENV_R3_FAILED_SIF_TIMEOUT`。R1/R2/R3与下载失败保留，不自动重试；S0环境/隔离门尚未通过，七作业未创建。现有CE4.1.1在sandbox展开触发原1800秒时限，未到压缩/SIF封装，无转换COMPLETE；原资产/归档/失败保留。转换后独立结束诊断命令编码失败，未取得新存储/进程终态；仅修正本地诊断代码及38测试pass，没有再SSH。最后有效余999.479492GiB为转换中快照，不当转换后最新余量或账号quota。R4依赖及转换新attempt分别核定，不自动延时/重建；不恢复监控、不读locked test、不替换比赛模型。指定Mac中转例外见[TRANSFER_AMENDMENT_20261104.md](TRANSFER_AMENDMENT_20261104.md)，实际限定清理见执行批次`operator/transfer_assets_20261103T180257Z/MAC_CLEANUP_COMPLETE.json`。

研究状态：`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。
