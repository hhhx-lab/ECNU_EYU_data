# 两外部METS模型同fixed103执行批次R1

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
