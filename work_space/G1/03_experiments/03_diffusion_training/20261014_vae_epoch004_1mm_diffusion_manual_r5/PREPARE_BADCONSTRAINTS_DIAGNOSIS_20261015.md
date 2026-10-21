# 预处理BadConstraints根因只读核查

## 范围与原证据

用户授权核实预处理堵塞原因。本次新增本地只读操作器，使用约定env凭据和显式分流VPN，一次有界SSH读取作业、分区、63个CPU节点、QOS及调度配置；没有sbatch/test-only提交、取消/更新作业、改远端文件或读取大数据。结束只关闭本次VPN，冻结57文件包未修改。

- 本地UTC：2026-10-15T05:44:38.923124+00:00（北京时间13:44:38）。
- 远端UTC：2026-10-15T05:33:33.562234+00:00，hostname=login1；时间独立记录。
- [CONSTRAINT_DIAGNOSIS.json](operator/constraints_readonly_20261015T054429Z/CONSTRAINT_DIAGNOSIS.json)含完整原始scontrol/sinfo/sacct/配置响应。
- [原held审计](operator/pull_launch_evidence_20261014T075642Z/pulled/r5/PREPARE_LAUNCH_AUDIT.json)与当前实际请求一致：1节点、8CPU、90G、24h、cpu_96G，无feature/指定节点/GPU请求。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。

## 已确认的根因

**90G单节点内存请求超过cpu_96G所有节点扣除系统预留后的最大可调度内存。** 分区名称不是可分配内存承诺；只核对分区UP和提交接受并不足以证明资源可运行。

当前Slurm为select/cons_res、CR_CORE_MEMORY。scontrol节点信息如下，单位为Slurm MiB：

| 项目 | 数值 |
|---|---:|
| 作业MinMemoryNode=90G | 92160 MiB |
| 62台常规节点RealMemory | 95051 MiB |
| 每节点MemSpecLimit系统预留 | 4096 MiB |
| 常规节点可调度上限 | 90955 MiB = 88.823 GiB |
| 作业超过常规节点上限 | 1205 MiB = 1.177 GiB |
| node25 RealMemory | 87043 MiB |
| node25扣预留上限 | 82947 MiB = 81.003 GiB |
| 可满足单节点90G的分区节点数 | 0/63 |

CPU总数均28，可满足8CPU总量；Features、ReqNodeList、ExcNodeList、Licenses均无要求。分区UP、AllowAccounts包含本账号，无分区MaxTime限制；本次读取的user/partition QOS内存/时限上限字段为空，不出现QOS拒绝。最明确且充分的静态不满足条件就是上述节点内存上限，不需要把资源忙碌、磁盘空间或代码错误混入根因。

节点AllocMem也印证上限：node6/node31满配时AllocMem=90955，而不是95051；空闲后仍无法给一个作业92160 MiB。FreeMem是操作系统运行时观测，不是Slurm可授权容量，不能用FreeMem或分区名替代RealMemory-MemSpecLimit。本次未读取控制器内部调度日志，根因判定依据为完整节点静态资源不满足加当前BadConstraints，不伪称管理员日志确认。

这不是普通Priority排队，继续等不能消除静态内存不匹配；也不是OOM、CUDA、脚本启动错误、数据SHA或磁盘不足。原预检只核分区/合同/held请求，漏核MemSpecLimit，属于提交可调度性检查缺口；原STATIC_PREFLIGHT历史pass保留，但不能继续据此宣称CPU资源可运行。

## 当前影响与保护

4069915仍PENDING BadConstraints，运行0；四A100训练4069916-4069919仍PENDING Dependency，运行0。无日志、输出病例、STARTED/FAILED/COMPLETE，训练未启动。package SHA仍13f188b2609ba57544c1258aa368daeabe0201ecbdb81c3ec86700f6441b6ede、57文件drift为空，源CSV SHA不变；账号余208.77GiB，非本阻塞原因。

没有新FAILED/非0/合同漂移；未取消/改作业/自动新attempt，旧失败和审计完整保留。本次仅核实原因，不把调度阻塞写成训练失败或科学结论。

## 不降资源的建议修复路径（未执行）

分区总览显示cpu_192G为UP、节点RealMemory约190913 MiB，是保持原8CPU/90G/24h需求的候选；本次未取其逐节点MemSpecLimit、分区账号/QOS准入，不能据总览直接放行。应先核验可分配内存和准入，再制定独立资源修正/launch审计；保持926例、四worker、严格1mm、数据SHA、四路batch16/各200k等合同不变，不能直接降低内存/worker绕过。

原四GPU作业依赖旧CPU4069915。任何修复还必须明确审计依赖迁移或新的唯一提交方案，不能仅提交新CPU后让GPU继续等旧job，更不能未经授权取消旧CPU触发四路invalid-dependency终止。若资源修改需更新冻结合同或job，必须另行授权和记录，禁止覆盖原57文件/SHA/held审计或自动重复提交。本次未做任何远端变更。
