# R7预处理整类标签丢失失败：只读终态核验

## 输入与范围

用户要求检查预处理是否结束及四训练是否正常开启。仅调用现有operator/snapshot.py一次，查询五作业squeue/sacct、日志/完成标记/56文件SHA/磁盘；不重扫影像、加载checkpoint或修改远端。

## 核查事实

本次唯一只读快照本地UTC2026-10-16T07:45:10.953602/远端07:34:04.557710确认CPU4096309 FAILED1:0，用时05:01:38。dataset/FAILED.json存在、COMPLETE.ok不存在，报BraTS-MET-01110-002重采样后前景类别4整类消失；runtime/prepare.py:155主动ValueError，非本次日志中的OOM/CUDA/时间上限或SHA漂移。stdout最后确认462例，471病例目录含部分/并行输出，不等于471完成或全926通过。四GPU4096310-13 CANCELLED0:0、运行0，STARTED/RUN_CONTRACT/progress/checkpoint/COMPLETE均无，空squeue不当成功。56文件SHA无漂移，余212798013440bytes约198.18GiB，VPN已结束，无第二快照/远端写入/新attempt/重提/标签或mask修改。监控当前PAUSED，本次只核实不重复暂停操作；原部分输出、R5/R6、原两覆盖门false及waived_by_user_not_pass/探索边界保留。该病例原生标签4数量、物理体积、位置、采样具体成因及其他病例风险尚未诊断；不能据本次结果判源标签错误或靠补标签/删病例/降门推进。

[本次快照](operator/snapshot_20261016T074500Z/SNAPSHOT.json)包含轻量调度、stderr/stdout尾部与标记摘要；本地UTC07:45:10.953602（北京时间15:45:10），远端快照07:34:04.557710分别保存，两个时钟不混用。

| 阶段 | 作业 | 终态 | 用时 |
|---|---|---|---|
| 预处理 | 4096309 | FAILED 1:0 | 05:01:38 |
| T1C | 4096310 | CANCELLED 0:0 | 0 |
| T1N | 4096311 | CANCELLED 0:0 | 0 |
| T2W | 4096312 | CANCELLED 0:0 | 0 |
| T2F | 4096313 | CANCELLED 0:0 | 0 |

## 失败身份

- package SHA：41c47c0516e9818c82319da2a370507ebc3fd773ef121f4baed786159e5a4158；drift=[]。
- 远端FAILED.json SHA：ad6368a7a7a4a7a469b0a78003ddd33f914db0a9a70b02c65eba9373e7e91a5d。
- FAILED记录的远端UTC：2026-10-15T18:38:27.663728，不是本地实时时间校准。
- 消息：BraTS-MET-01110-002: whole foreground classes lost during 1mm resampling: ['4']。
- 冻结代码先计算原生/目标标签计数，再检查原生有而目标无的类别并抛错；断言位于影像写出该病例目录之前。并行任务可能留下其他部分输出，不能仅以目录数认定成功。

## 后续边界

需要用户授权定向诊断该病例的原生标签4数量/体积、spacing/affine、目标网格与重采样行为后，才能讨论合法修订和独立新批次；本次没有执行修复或重提。不自动删标签/病例、膨胀mask、放宽整类保留要求或复用失败根。

更新STATUS、G1专题、知识登记、更新记录并运行Conda base catalog build/check；只有check通过才称同步。实验仍experimental_unvalidated、operator_approved=false、formal_gate_status=not_run_not_passed。
