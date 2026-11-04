# S0 编译工具链接续执行记录

原PLAN/PLAN.json保留制定时冻结合同。用户已明确授权同版独立Conda用户态构建、额外CPU S0准备/同镜像一次7200秒转换、R2专属空缓存和局部官方配置。总拟议8作业，仍4个单A100；最后要求环境准备→小规模运行验证→全量提交。

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
