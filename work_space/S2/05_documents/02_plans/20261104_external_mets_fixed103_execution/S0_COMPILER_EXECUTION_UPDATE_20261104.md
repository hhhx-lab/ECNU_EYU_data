# S0 编译工具链接续执行记录

原PLAN/PLAN.json保留制定时冻结合同。用户已明确授权同版独立Conda用户态构建、额外CPU S0准备/同镜像一次7200秒转换、R2专属空缓存和局部官方配置。总拟议8作业，仍4个单A100；最后要求环境准备→小规模运行验证→全量提交。

## 最新接续

独立调度验收通过，原R3/R1 staging失败保留。R2首次staging实际成功，独立构建环境及47包身份/标准backend要求已核；随后CPU提交前聚合QoS使用审计停止，未调用sbatch，无新JobID。只读诊断核冻脚本/环境/旧失败不变，队列另有两PENDING请求25 CPU，与本次合33低于1000限制；原验收器队列非空即hold，不是实际额度不足结论。已询问独立使用审计及复用同Slurm/环境首次提交，待确认，不再安装/求解或自动重提。118本机测试/57源码AST通过仅准备；SIF/wheel与完整推理环境仍未完成。后续步骤与精确证据见[S0 CPU R2执行记录](S0_CPU_R2_EXECUTION_20261104.md)。以下“调度待授权/R3未执行”等均为历史。

## 最新：R3求解通过，调度验收停止

用户已确认R3独立解析修正；一次实际求解产生47个官方来源包及精确锁，真实glibc2.17，solver SHA `68099499cdaee295940740b6144d1f9c0b73e9822338f310a05b8a44386d481c`、lock SHA `866d02c7106d809d4f14b87c0da101e2c3c722ab50f7a8f78e3ea89fb5fc2d77`。整体在Slurm QoS查询停止，没有预检COMPLETE；只读诊断明确`maxtrespj`字段不受支持，普通查询成功但无表头，不能猜列位置放行。

已准备带表头解析及安装前源码构建要求门，99本地测试/47源码AST与SHA通过，不当环境/GPU通过。已请求独立调度验收确认：仅复核同solver/lock、读取鲜活调度与容量、另存新验收；不再solver、不修改旧FAILED。通过后才安装构建环境、提交CPU准备并验收wheel/SIF，继而独立R4和J01–J04两完整smoke、联合门、J05–J07全量。当前均无JobID、未安装新环境、无可验收SIF或新分数。

失败远端UTC13:50:00.101397/本地14:01:26.494924，诊断远端13:51:50.030743/本地14:03:16.588496；最后文件系统余853.494140625GiB非账号quota或全量容量门。本轮接续只做本地证据复核/同步，无再次远端执行；原失败与用户VPN保留。详见[独立R3分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_COMPILER_R3_SCHEDULER_FAILURE_ANALYSIS_20261104.json)。下表及以下“待首次确认”等为先前UTC12:40历史，不能覆盖本节。

## 已发生

1. 编译器R1 solver成功，但实际LINK仍TUNA，来源门失败，无安装/环境/作业；原helper和失败保留。
2. R2已执行，实际在有效配置门停止，尚未调用solver。一次只读诊断发现Conda返回结构化Channel JSON，操作器误按URL字符串比较；官方scheme/location/name、TLS、cache正确，不是包缺失/模型/GPU错误。
3. R2远端12:23:10.174073/本地12:34:36.557981；诊断远端12:27:07.404710/本地12:38:33.632285。保留约11分钟UTC差；文件系统余854.400390625GiB非quota/全量预算。两次均复用用户VPN、未注销，旧环境及六失败marker guard保留。

## 修正与剩余门

独立R3解析修正已本机测试，支持真实Channel JSON和字符串，官方地址/名称/无凭据及实际LINK/FETCH/版本/文件类型/hash门不放宽，在assert前保存脱敏配置。既有brats2023_seg中87测试、44源码AST/SHA通过，非远端兼容/A100 pass；R3首次执行确认已询问，尚未收到，没有自动另开solver。

| 步骤 | 当前状态 |
|---|---|
| 原资产/ZIP/OCI/archive SHA | 已验收，继续服务器保留 |
| R3真实官方工具链求解 | 本机修正已测，首次执行待确认 |
| 独立构建环境及sdist构建要求 | 未安装/消费 |
| CPU S0 | runtime/staging/held审计器草稿；正式Slurm/冻结/提交器未完成，无JobID |
| 三个wheel/2025 SIF | 无可验收新产物，原SIF超时失败保留 |
| Bio-Intel推理R4/trainer/import/pip check | 未创建/完成 |
| J01–J04输入门与两完整smoke | 完整运行包和Slurm未完成，未提交 |
| J05–J07全103/MET/paired | 需真实smoke及联合容量/估时门，未提交 |

后续CPU须COMPLETED0:0及COMPLETE/wheel/SIF SHA，再消费至独立推理环境；CPU产物完成不当模型运行pass。同103真实输入、完整集成、K1-175主对照、native/MET/公开训练暴露边界不变，test不读，不恢复监控/自动重试或删旧失败。

详见[执行STATUS](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/STATUS_20261104.md)及[预检分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_COMPILER_PREFLIGHT_FAILURE_ANALYSIS_20261104.json)。三项研究状态保持experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed。
