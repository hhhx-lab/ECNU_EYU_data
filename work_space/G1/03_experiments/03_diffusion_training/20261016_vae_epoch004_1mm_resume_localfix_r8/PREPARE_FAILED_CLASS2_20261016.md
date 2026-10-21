# R8预处理再次整类丢失，训练未启动

2026-10-16北京时间18:20终态核验：本地UTC10:20:49.564089/远端10:09:43.149052分别记录，CPU4124867 FAILED1:0，用时01:26:45。dataset/FAILED.json存在（远端失败UTC09:44:08.514890，SHAfc254bde644be02374cf0714c0029bda8fd874fdc77a34c46bfd2ae94ca7482d），COMPLETE.ok无；BraTS-MET-01132-002严格1mm重采样后类别2整类消失，runtime/prepare.py:268主动ValueError，未定向核原生体积/采样成因，不能直接套用01110-0024点例外。stdout最后确认506例，514目录含引用/并行或部分输出，不当514完成或全471引用终验；旧失败病例01110-002已在本次日志完成。四A1004124868/69/70/71 CANCELLED0:0、运行0，全部训练STARTED/RUN_CONTRACT/progress/checkpoint/COMPLETE无，未开始训练，空squeue不当成功。58文件SHA无漂移，余211776700416bytes约197.232GiB；本周期一次只读快照、VPN结束，保留R8及R7/R5/R6全部失败/部分数据，无新attempt/重提/扩大例外/降batch/源重扫/大数据拉回。已通过automation_update将heartbeat设PAUSED并读取配置确认，待该病例定向诊断与新明确合同，不能自动推进。原两覆盖门false、waived_by_user_not_pass及experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed不变。

## 已核实的证据

- [本周期唯一只读快照](operator/snapshot_20261016T102041Z/SNAPSHOT.json)，本地文件SHA `72b0a707c57e66681345254b780fd6bb4b8ea996960a2f51e74701ae4f8773f0`。快照保存sacct、日志尾部、远端FAILED摘要与SHA、58文件漂移结果和两端时间。
- FAILED消息：`BraTS-MET-01132-002: whole foreground classes lost during 1mm resampling: ['2']`。这是数据整类保留断言触发；当前证据未显示OOM、CUDA错误、磁盘耗尽、时间上限或包漂移。
- CPU终态 `FAILED 1:0`，四GPU终态 `CANCELLED 0:0` 且Elapsed均为零。依赖阻断按设计生效，没有训练或权重结果。
- 原局部修复病例 `BraTS-MET-01110-002` 日志记录第463例完成，15条rows；不等于全926或连通域细节验收通过。
- 当前stdout506例，目录514只是物理路径数量，不作完整输出可复用数量。今后接续仍须逐marker/实际目标/内容SHA及索引核验，不能把本次部分目录直接算已完成。

## 暂停与边界

自动化 `g1-epoch004-1mm-diffusion` 已通过应用工具更新为 `PAUSED`，保留原name/prompt/30分钟计划/目标线程；工具回执及本地automation.toml读取确认一致。暂停不删除任何作业证据或服务器数据。

本次不再SSH诊断、重提、修改冻结58文件或增加标签例外。下一步应定向核对01132-002标签来源SHA/affine、原生类别2体积与spacing、默认重采样与最近邻命中情况，再根据证据决定合法局部修订；本次没有这些结论。仅已授权01110-0024点例外不可扩展为通用补标签规则。不得删标签/删病例/膨胀mask制造通过。

