# 两套外部METS模型同fixed103服务器横评计划

北京时间2026-11-05 12:51复核：用户授权独立R3a后，单个CPU准备4304783已held核验并同ID释放，唯一即时快照RUNNING/node201；8CPU/64GiB/4h/无GPU，非全量推理或CPU产物完成。STARTED/FAILED/COMPLETE均无、日志0bytes，201本机tests/70AST及真实冻结脚本bootstrap通过非wheel/SIF/A100通过。原R3引用路径错误在sbatch前失败无JobID，其四文件SHA与旧4304740/环境全部保留。复用R2环境和版本，不重装/solver/恢复监控；用户VPN保留，835.12890625GiB为文件系统余量非数值quota/全量峰值验收。完整环境、两真实smoke和J01-J07仍未完成，见[执行记录](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_CPU_R3_R3A_EXECUTION_20261105.md)。原PLAN不改，下方为历史。

北京时间2026-11-05 12:18：按用户要求补实际代码检查，修正初始化候选的真实远端 bootstrap/import/C/C++语法/容器CLI通过，162本机测试/65AST通过；非真实CPU构建或GPU通过。原R2/4304740失败保留，无新R3/作业/安装/solver/转换，容量未刷新，用户VPN保留。见[启动复核](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_CPU_BOOTSTRAP_REVIEW_20261105.md)。原PLAN不改，下方状态为历史。

北京时间2026-11-05 12:02最新：用户客户端重连后，独立R2首次CPU准备作业4304740已held核验并同ID释放，随后21秒FAILED 1:0。冻结Slurm的nounset早于系统profile，`COLORTERM`未定义导致Python入口之前退出；原R2/全部失败保留，不自动重提。一次只读快照无身份漂移，余835.58984375GiB非数值quota/全量峰值验收，用户VPN保留；138本机测试非模型/GPU通过。独立R3仅调整初始化顺序已询问待确认，wheel/SIF、完整推理环境、真实smoke及J01-J07仍未完成，见[CPU使用接续](S0_CPU_USAGE_EXECUTION_20261105.md)。原PLAN不改写，以下未SSH/无JobID等为历史。

北京时间2026-11-05 11:15最新：用户已批准复合字段解析修正后的独立R2，136本机测试通过；本机调用因aTrust断开在VPN归属门停止，尚未SSH/执行远端R2/sbatch，无JobID。只读status返回disconnect、界面未登录，旧登录完成卡片不作在线证据；未logout/改VPN或安装/solver，没有新远端UTC/存储观察。已请求恢复学校登录，随后接续同一未远端执行的R2，不新建提交attempt。wheel/SIF、完整推理环境、两真实smoke和全量未完成，见[CPU使用接续](S0_CPU_USAGE_EXECUTION_20261105.md)；原PLAN不改写，下方“当前/最新/待确认”为历史。

当前实施：已批准同版独立构建/额外CPU准备/单次7200秒转换及R2局部配置。编译器R2因操作器误比较Channel JSON而停止，solver未执行；独立R3修正已本机87测试通过、首次远端执行待确认。没有新构建环境/R4/SIF/JobID或推理结果；具体脚本完成度和门控见[S0接续记录](S0_COMPILER_EXECUTION_UPDATE_20261104.md)。下方“当前/最新/待确认”等均是历史，原PLAN不改写。

制定日期：2026-11-04，Asia/Shanghai。

2026-11-04用户再次要求按本方案执行提交；当前只读复核确认S0仍有同版依赖与2025 SIF两处阻断。PyPI三个原版本元数据均200，但所列Linux CPython3.10 wheel要求glibc2.24/2.27，不能直接装到2.17。已提出[独立同版构建及单次CPU转换补充方案](S0_COMPATIBILITY_AMENDMENT_20261104.md)，两项新方法确认尚待用户答复；未新增SSH、安装/转换或作业。原7作业DAG不在门未通过时提前提交，旧失败/模型/同103合同保留。

当前实施终态：`S0_R4_OFFICIAL_R2_PACKAGES_NOT_FOUND_NO_INSTALL_NO_JOBS`。用户完成校园登录并确认接续，修正版已首次实际远端执行；官方完整channel一次dry-run报告缺`connected-components-3d=4.1.1`，host/virtual glibc2.17，无联合兼容方案或所选包来源门通过。旧环境和四失败marker未变；R4未创建/安装、七作业未创建、无新分数，SIF仍无验收产物且未重试。远端失败06:06:50.526279/本地收尾06:18:16.729937分别保存，文件系统余909.103515625GiB不是账号quota或全量预算，手动VPN未注销。见[官方R2终态分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/ENVIRONMENT_R4_OFFICIAL_R2_FAILURE_ANALYSIS.json)。以下“最新/进行中/待确认”是先前历史，不覆盖当前；原PLAN/VERIFICATION继续保留制定时合同，不自动改变依赖方法/版本或重建SIF。

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
