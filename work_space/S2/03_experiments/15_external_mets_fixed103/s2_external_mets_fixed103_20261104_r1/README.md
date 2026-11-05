# 两外部METS模型同fixed103执行批次R1

**2026-11-06最新：Bio独立R2已获授权并执行一次held提交，Slurm明确因账号zqchen可用CPU机时不足拒绝，无JobID。** 原始stderr完整保留，唯一后续queue/accounting无Bio；2025的4310575仍PENDING/Priority、运行零、0预测，不重复提交/释放。R5环境独立验收及完整七折/真实fixed103/资源不改，原R1原因仍未知不倒推。292本机发现/290pass/2版本skip、100生产3.10 AST和15轻量回执SHA通过，非GPU/质量pass。余693.491GiB为文件系统余量，非数值quota或机时余额；需管理员恢复CPU机时后另行授权，不自动R3/重装solver/smoke或监控。详见[R2执行](BIO_FULL103_R2_EXECUTION_20261106.md)、[失败复核](BIO_FULL103_R2_FAILURE_REVIEW_20261106.json)和[最新状态](STATUS_20261106.md)。下方均为历史，跳过门保持`waived_by_user_not_pass`及原科学边界。

**最新北京时间2026-11-05 23:46：CPU4309579已COMPLETED 0:0，耗时1小时16分29秒，最终2025 SIF转换及大小/SHA验收通过。**SIF 4.1253 GiB，三复用wheel身份一致，COMPLETE存在、无FAILED或漂移；这是转换工程通过，不是GPU推理通过。完整Bio-Intel环境、两路真实GPU smoke及全量仍未完成或提交。最新文件系统余744.78 GiB，账号数值quota/全量峰值门未验证；本次仅一次只读快照，不安装或新增作业，用户VPN保留，原失败不删。见[完成复核](operator/snapshot_sif_recovery_r2_20261105T154434Z/LOCAL_SNAPSHOT_REVIEW.json)与[STATUS](STATUS_20261104.md)，下方压缩中为历史。

**最新北京时间2026-11-05 22:40：CPU4309579 RUNNING/node225约1小时13分钟，原生展开成功，正在SquashFS压缩。**暂无FAILED/最终COMPLETE或漂移，完整SIF/Bio环境/两GPU smoke及全量仍未完成/提交。七轻量报告和源码本机复核通过；文件系统余763.01GiB非数值quota/全量预算，一次只读查询不改作业、用户VPN保留。见[快照复核](operator/snapshot_sif_recovery_r2_20261105T143947Z/LOCAL_SNAPSHOT_REVIEW.json)与[STATUS](STATUS_20261104.md)，下方为历史。

**最新北京时间2026-11-05 21:50：CPU4309579已RUNNING/node225约23分钟，11层规范化及逐成员校验完成，正在原生镜像展开。**暂无FAILED/最终COMPLETE或漂移；完整SIF/Bio环境、两路真实GPU smoke与全量未完成/提交。六轻量报告/执行源码本机复核通过，详细ledger留远端；余775.88GiB为文件系统余量，不是数值quota或全量峰值验收。一次只读查询无重提/安装，用户VPN保留，见[快照复核](operator/snapshot_sif_recovery_r2_20261105T135002Z/LOCAL_SNAPSHOT_REVIEW.json)与[STATUS](STATUS_20261104.md)，下方PENDING为历史。

**最新北京时间2026-11-05 21:26：R2服务器Python3.10/最小CE验证通过，唯一新CPU4309579已held核验并同ID释放，当前PENDING/Resources，尚未运行。**本机完整246发现/244pass/2版本限定skip，3.10核心24项无skippass；17轻量审计和三实际源码/协议输入及冻结Slurm独立复核通过。只有STAGING_COMPLETE，无Python STARTED/FAILED/COMPLETE，完整SIF/Bio环境/两真实GPU smoke仍未通过，两全量/J01-J07未提交。复用三wheel，无重编/下载/安装solver/自动retry，旧失败保留；余798.46875GiB非数值quota/全量预算，两端UTC分记、用户VPN保留。见[R2实际执行记录](S0_SIF_RECOVERY_R2_EXECUTION_20261105.md)与[STATUS](STATUS_20261104.md)，下方均为历史。

**最新17:23本机复核：4304987已FAILED1:0/8分35秒，不是等待GPU；完整SIF/Bio环境和两路真实GPU smoke尚未通过，两全量未提交。**Python3.10空tar的结束读取触发本方normalizer Errno22，原最小验证漏空层、本机3.11回归未覆盖该版本边界。已用现有Conda3.10复现，独立修正候选8项全pass；3.11全回归230发现/229pass/1skip（原错误复现项另在3.10通过）、81生产AST。冻结原程序/所有失败不改，候选只本机，独立服务器验证和新held CPU恢复已询问待确认，无自动重提/重编/重下载/安装。见[失败验收](S0_SIF_RECOVERY_R1_FAILURE_ACCEPTANCE_20261105.json)，remote09:06:44.706516/local finally09:18:11.995601分记、自建VPN关闭，余815.79296875GiB非数值quota/全量峰值门；下方RUNNING均为历史。

**最新本机复核09:06UTC：CPU4304987仍RUNNING/node214，Python STARTED和三个wheel复用验收已出现；无FAILED/COMPLETE，完整SIF待完成，两路GPU smoke和全量仍未提交。**远端快照08:50:08.374584、本机finally09:01:36.157955分记；三个轻量报告及执行源码/合同SHA复核pass，自建VPN关闭。余815.7939453125GiB非数值quota/全量峰值门；最终正式验证222tests/79生产源码AST（此前221tests/78AST，下方80AST为计数笔误，原回执不改），非模型/GPU质量通过。见[最新快照复核](operator/snapshot_sif_recovery_r1_20261105T090112Z/LOCAL_SNAPSHOT_REVIEW.json)和[STATUS](STATUS_20261104.md)。保持原失败/PLAN，不重提/重编/重下载/solver或恢复监控；下方初始状态均为历史。

**2026-11-05本次接续：同镜像硬链接兼容最小实测通过，新CPU4304987已held核验并释放，即时RUNNING/node214；两全量仍未提交。**第6层whiteout删目标但缓存链接保留，原5byte样本在同CE复现原错误，物化链接后native展开pass。全原归档/实际三wheelSHA一致，不重编/下载/安装solver。派生归档只改变链接存储与diff_ids/configSHA，不称原Docker字节摘要相同；全镜像文件比较/SIF尚待本次CPU产物验收。221tests/80AST、17审计SHA和三执行源码/协议输入核验pass，非GPU/质量pass；旧4304783与全部失败保留。余822.240234375GiB非数值quota/全量预算，自建VPNfinally关闭，完整环境/两真实GPU smoke仍是全量前置门。见[完整兼容恢复记录](S0_SIF_RECOVERY_EXECUTION_20261105.md)与[STATUS](STATUS_20261104.md)，下方为历史。

**北京时间2026-11-05 15:54：VPN/SSH已恢复，CPU4304783实际FAILED1:0/35分26秒；三wheel编译与CPU探针通过，2025镜像SIF展开失败，两全量仍未提交。**只读层元数据确认硬链接目标原本存在且排在前面，具体后端展开问题/恢复方式仍待核验；独立恢复已请求，未自动新建作业/重编/重下载。10轻量报告SHA通过，202本机tests/72AST非模型/GPU通过；余822.24GiB非数值quota或全量预算，两次自建VPN均finally关闭。用base与Chrome Bridge调用网络操作器，不包在找不到playwright-mcp的brats2023_seg测试环境；不改PATH。见[完整失败验收](S0_CPU_R3A_FAILURE_ACCEPTANCE_20261105.json)，下方为历史。

**北京时间2026-11-05 15:41：用户确认重连后，旧认证会话与换网前基线仍被保留，分流验收继续失败，未SSH或提交全量。**脚本的限定路由尝试已回滚，restore_failures为空；未改基线/脚本或注销用户。现需在aTrust退出校园账号登录并暂时保持未登录，才能先捕获当前网络基线后继续官方登录。CPU4304783当前状态/远端容量未知，不凭历史RUNNING提交；完整环境、两真实smoke和联合门仍须核验。见[接续诊断](FULL_INFERENCE_RECONNECT_HOLD_20261105.json)，下方为历史。

**北京时间2026-11-05 15:25复核：已收到“提交两个的全量推理作业”，本次因分流验收未通过停在接入门，未SSH或创建新作业。**aTrust在线/隧道可用，但原本机默认网关基线已变化，三个公网探针重新走校园隧道；不改基线或绕过门、未注销用户会话。用户选择手动重连后继续，后续一次本机只读status仍同登录会话/default changed，等待完成确认。4304783本轮状态未知，12:51 RUNNING仅最后历史快照；CPU产物/完整环境/两真实smoke尚无本轮验收，不提前提交两全量。202本机tests/71AST通过只为只读入口，不是远端通过。见[接入停止记录](FULL_INFERENCE_ACCESS_HOLD_20261105.json)，以下为历史。

**北京时间2026-11-05 12:51复核：用户授权R3a首次CPU准备4304783已held核验并同ID释放，即时快照RUNNING/node201。**资源cpu_192G/8CPU/64GiB/4h/无GPU，sbatch/release均0，十轻量审计bytes/SHA和冻结身份本机通过；STARTED/FAILED/COMPLETE均无、两日志0bytes，非CPU产物或GPU推理通过。旧R3误引用operator而非reports导致sbatch前失败无JobID，原四文件SHA/旧4304740/R2及失败永久保留。201本机tests/70AST通过，真实冻结脚本bootstrap通过；无重装/solver/监控恢复，用户VPN保留。最新文件系统余835.12890625GiB非数值quota/全量峰值门；完整推理环境、两真实smoke和J01-J07仍未完成。见[R3/R3a执行记录](S0_CPU_R3_R3A_EXECUTION_20261105.md)及[STATUS](STATUS_20261104.md)，下方状态为历史。

**北京时间2026-11-05 12:18：已补实际启动检查，本次不重提交。**修正初始化候选在远端干净 shell、七 import、C/C++ 语法及容器 CLI 通过；162 本机测试/65 AST 通过。原脚本更早触发 HISTCONTROL，与 Slurm 的 COLORTERM 同根因。原回执和 R2 冻结包保留，独立本机验收仅认 bootstrap pass，不认 wheel/SIF/GPU pass。无新 R3/JobID、安装/solver/真实转换，smoke 和全量仍未提交；用户 VPN 保留，容量未刷新。见[完整复核](S0_CPU_BOOTSTRAP_REVIEW_20261105.md)；下方状态为历史。

**北京时间2026-11-05 12:02最新：CPU准备4304740首次提交/同ID释放成功，但运行21秒后FAILED 1:0。**我的冻结Slurm在系统profile之前启用nounset，遇`COLORTERM`未定义退出，尚未进入Python准备；不是GPU或模型失败。即时只读快照无静态/旧失败身份漂移，余835.58984375GiB非数值quota/全量容量门；用户VPN保留。138本机测试通过非模型通过，独立R3初始化顺序修正已询问待授权，原包/失败不改，不自动重提。CPU准备、真实smoke及J01-J07未完成，无新分数。见[失败分析](S0_CPU_USAGE_R2_SUBMISSION_FAILURE_20261105.json)和[STATUS](STATUS_20261104.md)；以下“未SSH/无JobID/最新”等均为历史。

**北京时间2026-11-05 11:15最新：R2已授权，但aTrust断开，尚未SSH或提交作业。**136本机测试通过；本次停在VPN归属门，只读status与界面确认未在线，未logout/改适配器/重装/solver，无新远端UTC或存储观察。请按学校流程恢复登录后接续同一尚未远端执行的R2，不新建提交attempt。见[接入回执](S0_CPU_USAGE_R2_ACCESS_CONTINUATION_20261105.json)和[STATUS](STATUS_20261104.md)；以下“R2待授权”均为历史。

**北京时间2026-11-05最新：实时使用验收因我的Slurm复合字段解析错误停止，未sbatch，无JobID。**只读对照确认分区/队列未变，冻结Slurm/环境/模型/版本和两次失败保留；独立R2解析与真实输出/held回放136本机测试通过，新执行确认已询问、尚未授权或执行。余851.736328125GiB非数值quota/全量预算，两真实smoke及全量未完成。见[CPU使用接续](../../../05_documents/02_plans/20261104_external_mets_fixed103_execution/S0_CPU_USAGE_EXECUTION_20261105.md)及[STATUS](STATUS_20261104.md)，以下“最新/当前/待确认”均为历史。

**最新：R2独立构建环境已安装验收成功；CPU提交前聚合QoS使用审计停止，尚未sbatch，无JobID。**冻结Slurm/原47包锁/失败回执保留；一次只读诊断确认其他两作业PENDING，CPU请求25+8=33低于1000上限，没有实际超额证据，但验收器尚未消费已有作业资源使用。独立审计接续已询问待确认，不重装/求解或自动提作业。118本机测试/57源码AST通过非A100/模型通过；三个wheel与SIF未构建，完整推理环境、两真实smoke/全量未完成。文件系统余851.736328125GiB非数值quota/全量预算，旧失败/环境及用户VPN保留。详见[STATUS](STATUS_20261104.md)和[S0 R2接续](../../../05_documents/02_plans/20261104_external_mets_fixed103_execution/S0_CPU_R2_EXECUTION_20261104.md)；以下“最新/当前/待确认”均为历史。

**当前（UTC14:08本地复核）：R3官方47包求解/精确锁已通过，整体预检因Slurm QoS查询字段失败停止；独立调度验收待确认。**仅补做调度验收且复用同solver/lock，不重跑求解器或改旧FAILED。99本地测试/47源码AST通过；安装前源码构建要求门已准备但未执行。构建环境/R4未创建，无可验收SIF或本批JobID，smoke和全量均未提交。远端13:50/本地14:01两端UTC分记，最后文件系统余853.494140625GiB非账号quota/全量预算。见[最新分析](S0_COMPILER_R3_SCHEDULER_FAILURE_ANALYSIS_20261104.json)及[STATUS](STATUS_20261104.md)；以下“当前/最新/待确认”均为历史。

**当前（UTC12:40复核）：S0编译器R2因操作器Channel JSON比较错误停止；独立R3修正待确认。**只读诊断证实官方配置/cache/TLS正确，但solver未执行、构建环境/R4未创建，无SIF/JobID/GPU结果。CPU准备runtime/staging/held审计器仅本机草稿，正式Slurm/原七作业包未完成；87测试/44源码AST pass仅准备边界。见[最新STATUS](STATUS_20261104.md)及[S0分析](S0_COMPILER_PREFLIGHT_FAILURE_ANALYSIS_20261104.json)。下面“当前/最新/待确认”均为历史阶段。

当前执行请求为`S0_COMPATIBILITY_AMENDMENT_AWAITING_USER_CONFIRMATION`：再次执行请求后仅本地复核/公共PyPI元数据，独立同版构建与额外CPU准备/单次转换两项尚待确认，没有新远端操作或作业。见EXECUTION_CONTINUATION_REVIEW_20261104.json及计划S0_COMPATIBILITY_AMENDMENT_20261104.md；下方官方R2失败是最后远端终态。

当前为`S0_R4_OFFICIAL_R2_PACKAGES_NOT_FOUND_NO_INSTALL_NO_JOBS`：校园登录后的官方R2预检已结束，缺锁定`connected-components-3d=4.1.1`，R4与七作业均未创建。详见[终态分析](ENVIRONMENT_R4_OFFICIAL_R2_FAILURE_ANALYSIS.json)及下方当前终态；所有未登录/进行中/待确认段为历史，不覆盖本状态。

2026-11-04用户授权按已拟计划执行并提交。合同入口：[PLAN.md](../../../05_documents/02_plans/20261104_external_mets_fixed103_execution/PLAN.md)。两成品为2025 METS第一名固定镜像与2026 Bio-Intel完整七fold；K1-175主对照，同103真实四模态，公开训练暴露按原计划披露。

本目录先保存独立S0预检与资产/环境门证据。授权提交不等于预检已通过；未取得完整资产/许可、当次容量、独立依赖及smoke合同前不创建训练/推理作业，不修改原计划/旧失败，不自动retry。

当前事实以[STATUS_20261104.md](STATUS_20261104.md)和operator逐次UTC回执为准。`.env`只存路径配置，不复制登录密码；凭据只由原服务器env加载。用户已单独授权指定模型资产在Mac专属临时目录中转；全部远端bytes/SHA验收后只清理本批临时文件，模型长期存储仍在服务器。MRI/GT/预测不回传Mac。

最新：`S0_R4_OFFICIAL_R2_VPN_LOGIN_NOT_READY_NO_SSH`。用户已确认修正路径后的独立R2预检，33份源码与此前60测试身份一致；一次启动在VPN归属门停止，未SSH或调用Conda/创建R4/安装/提交。未调用stop；本机只读状态确认aTrust logout/tunnelUnavailable，脱敏诊断停在学校SSO密码表单、无验证码/MFA文字，但不判定密码/账号错误。请按学校流程完成aTrust浏览器登录后确认接续同一尚未执行的远端R2；未自动重启VPN或预检。见[接入失败分析](ENVIRONMENT_R4_OFFICIAL_R2_ACCESS_FAILURE.json)。原路径/镜像/SIF失败及资产验收保留，七作业未创建、无新分数；下述R2未获授权是历史。

## 输入、动作与产物

- 当前终态：`S0_R4_OFFICIAL_R2_PACKAGES_NOT_FOUND_NO_INSTALL_NO_JOBS`。用户完成校园登录后，同一修正版实际远端dry-run在官方完整channel报告缺锁定`connected-components-3d=4.1.1`，glibc2.17；没有联合兼容方案/所选包来源门通过，不能据此断言release全球不存在。远端失败06:06:50.526279/本地收尾06:18:16.729937分别记录；文件系统余909.103515625GiB不当quota或全量预算。旧环境及四失败marker核验未变、手动VPN未注销；R4未创建、七作业未创建，无新分数。63本地测试非远端环境/GPU通过；一次进行中只读快照及上方未登录等均为历史。见[官方R2终态分析](ENVIRONMENT_R4_OFFICIAL_R2_FAILURE_ANALYSIS.json)。不自动换依赖来源/构建方法/版本或重试SIF，已有资产验收与Mac清理保留。
- 官方预检/R2：原执行helper与失败保留，官方origin检查要求实际URL为官方HTTPS地址；修正版仅调整历史R1标记和新审计目录，生成代码编译/原记录布局回归及60测试通过，尚未远端执行。此前镜像预检按exists跳过不存在标记，R1失败marker未纳入其覆盖，不能将环境metadata/Python核验等同全部失败marker已核验。
- R4最新：用户已授权同版本二进制核验后创建，但预检实际走服务器既有清华镜像、缺connected-components-3d4.1.1而停止；R4未创建/安装，原R1-R3核验未变。官方API查询均访问失败，不称官方缺包；直接指定官方完整URL的独立预检已请求确认，未执行，见[预检失败分析](ENVIRONMENT_R4_PREFLIGHT_FAILURE_ANALYSIS.json)。上方SIF终态仍是已有资产准备的历史事实，不覆盖本次R4预检停止。
- operator/preflight_environment_r4.py及environment_binary_contract.py：一次真实glibc/virtual package/精确六版本dry-run，版本/平台/来源/重复/Torch或编译器替换门；不安装、不创建环境。operator/inspect_conda_forge_r4_metadata.py只读固定官方API元数据，不下载模型/包，不重试安装。
- 输入：固定2025镜像digest、Bio-Intel commit/HF revision、BrainIAC官方链接/研究许可、同103真实四模态及K1原输入参考515清单。不得用其他模型、补全T2W或其他split替代。
- `operator/preflight.py`及独立能力诊断：四份有界只读S0快照，核资源/容量/容器/资产出口/K1身份，不是完整输入体素验收。
- `operator/install_environment.py`：原独立环境因CUDA单索引的Pillow源码构建依赖失败，环境/源码/日志/回执保留。
- `operator/install_environment_r2.py`：用户批准独立R2，普通依赖走PyPI、Torch固定cu124、Pillow10.4二进制；只有版本核验、trainer发现及pip check通过才出准备COMPLETE，不是模型运行pass。
- R2实际通过Torch二进制安装，但pandas2.3.3回退源码/GCC4.8.5 Cython检查失败；用户又批准`operator/install_environment_r3.py`，先核Conda-forge兼容二进制与host/solver，再copy-clone新环境、同版pandas/numpy安装、完整依赖与trainer/版本核验，不安装系统编译器。R1/R2及失败只读保留，当前.env仅改为R3路径；没有回执不称环境已通过。
- `operator/download_bounded_resume.py`：用户批准的一次低并发断点接续，逐64MiB分片响应范围/长度/SHA登记，完整原SHA才生成15文件清单；旧无ledger的稀疏partial不作为完成片复用，不自动再建attempt。
- `operator/transfer_assets.py`：仅消费本次唯一成功清单，fresh容量门、独占SCP `.part`、逐文件和全15文件远端SHA、两端UTC验收；之后按专属临时清单SHA核对并清理，不清其他缓存或旧服务器数据。尚未执行时不得称模型已传输或缓存已清。
- `runtime/materialize_assets.py`：离线OCI/diff_id逐层核验、标准docker-archive及安全独占ZIP展开完成；一次现有CE4.1.1 sandbox展开超时，未到系统mksquashfs和原生SIF封装。原1800秒预算、压缩资源约束和失败保留，不改管理员配置、不自动延时。完整七fold元数据身份已核，模型加载/隔离和SIF门未通过。
- `operator/snapshot_materialization.py`：转换中一次只读日志/marker/路径大小快照成功，余999.479492GiB；转换后新增进程检查的独立诊断命令编码失败，未取得新存储/进程终态。仅修正本地生成代码并前置编译验证，未再SSH执行，不把本地修正当远端验收。
- `operator/verify_local_transport.py`：最新51项本地输入、传输回执JSON、敏感头清理、ZIP/解压界限、原生SIF描述符、生成代码、二进制合同及官方元数据错误语义回归通过，见`operator/verify_transport_20261104T031635Z/VERIFICATION.json`；source SHA/test日志可追溯，测试源码留本机不提交，非远端精确环境或A100测试。

## 尚未完成的门

资产与依赖准备完成后仍须真实容器隔离、完整模型消费、输入515逐SHA/几何、K1输出103当前身份与MET历史一致、GPU映射、完整smoke合同及正式Slurm/运行包冻结。当前没有可直接提交的完整运行包；先J01–J04四个held审计释放，成功后才创建J05–J07。失败不自动retry，不恢复后台监控。

2025镜像dataset.json登记1296条，fixed103全部103条属于其中。此为公开元数据交集，实际fold训练成员/训练日志仍未知；不能称2025是本103完全未曝光模型。Bio-Intel完整集成公开train-fold暴露与K1开发选模边界继续披露，同集成品描述性横评不是独立泛化验证。

状态：`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。
