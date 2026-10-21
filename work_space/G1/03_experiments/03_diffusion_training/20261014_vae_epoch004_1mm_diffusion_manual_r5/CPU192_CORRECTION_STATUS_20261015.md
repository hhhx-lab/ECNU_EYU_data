# CPU192资源修正：分区已改，释放权限阻塞

此文保留旧操作历史，最新用户授权已[唯一替换CPU提交成功](PREPARE_REPLACEMENT_STATUS_20261015.md)：4086739已释放，四GPU依赖已迁移，旧4069915主动取消，不再需要管理员release旧ID。下文未释放状态不是当前状态。

## 授权与不变合同

用户明确授权核验并改用cpu_192G，保持8CPU/90G/24h及四训练依赖。本次选择只修正原来尚未运行的CPU job4069915，不新sbatch、不取消作业，保留四A100 job4069916-4069919的afterok:4069915。没有改数据、worker、batch16、四路各200k成功更新、split823/103/104或冻结57文件SHA。

保持artifact_status=experimental_unvalidated、operator_approved=false、formal_gate_status=not_run_not_passed；验收豁免仍waived_by_user，不宣称训练/科学通过。

## 核验结果

cpu_192G State=UP，AllowAccounts包含zqchen，AllowGroups/AllowQos为ALL；无MaxTime/ReqResv限制，读取的user与分区QOS内存/时间上下限为空。29节点均28CPU、RealMemory190913MiB、MemSpecLimit10240MiB，扣预留后180673MiB（176.438GiB），能满足单节点92160MiB（90G）及8CPU；当前节点ALLOCATED/MIXED意味着需要正常排队，并非资源静态不可满足。

账号可用224162480128 bytes（208.77GiB），仍覆盖冻结213513415862 bytes容量预算。原CPU零运行、无dataset/training；四GPU均PENDING Dependency，原SHA/源CSV核验不变。静态资源与实际JobDetails核对通过，但不代表已经开始运行。

## 实际发生的操作与停止点

1. 第一独立审计：原Job4069915 PENDING BadConstraints，资源核验后执行一次` scontrol hold 4069915`成功。返回JobHeldAdmin而非预期JobHeldUser，校验中断，未执行分区更新。首失败/挂起回执保留。
2. 一次只读检查确认管理员级挂起来自本次成功hold，无旧管理员挂起或分区修改回执；四GPU依赖未变。
3. 在独立completion审计中保留首失败，重新核验资源和明确的本次挂起来源，仅完成尚未执行步骤，不重复hold/sbatch。执行一次`scontrol update JobId=4069915 Partition=cpu_192G`返回0。
4. 修正后held审计确认有效Partition=cpu_192G，NumCPUs8、MinMemoryNode90G、TimeLimit1-00:00:00、输出日志路径和原Command不变；逐一确认四GPU仍afterok:4069915(unfulfilled)。57文件SHA未变，原Slurm头的cpu_96G不改，本次有效分区以已授权scheduler override及新审计为准。
5. 执行一次`scontrol release 4069915`返回1：`Access/permission denied for job 4069915`、`slurm_suspend error: Access/permission denied`。**分区已改，但CPU仍JobHeldAdmin，尚未释放/运行；本次修正没有端到端完成。**

这是操作器选择普通scontrol hold导致管理员级挂起的工程缺口，不能将其说成用户数据或训练失败；测试模拟了成功hold但未验证实际账号对管理员级hold的释放权限。初次离线5测试发现字段解析CPUs/Task遗漏（未远端操作），修复后5/5pass；实际hold原因差异后增加测试至6/6pass，但模拟测试仍不能证明远端释放权限。首挂起校验中断与后释放拒绝证据分别保留，不覆盖。

遇到明确权限拒绝后停止，不尝试Priority修改、uhold/转换挂起或其他绕过管理员限制方法，不重复release、取消/重投或创建新attempt。后续只读拉回两次操作的JSON审计，没有额外job轮询、NIfTI或checkpoint下载。每次操作显式分流VPN，finally只关闭本次VPN。

## 证据与时间

- [首次操作回执](operator/prepare_cpu192_correction_20261015T063601Z/CORRECTION_RECEIPT.json)：本地UTC06:36:11.642838，远端停止UTC06:25:07.446966；仅成功hold。
- [挂起检查](operator/inspect_partition_hold_20261015T063714Z/INSPECTION.json)：确认JobHeldAdmin、旧分区与四依赖。
- [独立完成回执](operator/complete_unexecuted_cpu192_update_20261015T063905Z/CORRECTION_RECEIPT.json)：本地UTC06:39:15.246412（北京时间14:39:15），远端停止UTC06:28:11.052393；update成功、release拒绝。
- [两次轻量远端审计拉回](operator/pull_partition_correction_audits_20261015T063950Z/AUDITS.json)：含STATIC_RESOURCE_PREFLIGHT、RESERVATION、CORRECTED_HELD_LAUNCH_AUDIT、各命令响应及失败记录。此次为JSON结构拉回，非原始字节SHA验收，不宣称逐文件原字节备份。

原提交/held/release审计、57文件packageSHA13f188b2609ba57544c1258aa368daeabe0201ecbdb81c3ec86700f6441b6ede及历史失败均未覆盖。新本地操作器和离线测试不属于冻结runtime，不变更package。

## 需要管理员完成

当前最小操作是管理员核对Job4069915已经cpu_192G/8CPU/90G/24h后执行：

```bash
scontrol release 4069915
```

释放后应只读核对JobHeldAdmin消失、有效分区和资源不变，以及四GPU仍afterok:4069915；不能只据命令返回宣称训练RUNNING。四训练不需要新提交或依赖迁移，因为生产者ID保持原4069915。未经新的明确授权，本操作器不会另建任务替代或取消原CPU。

本地现有heartbeat配置已PAUSED，本轮保留暂停状态并通过automation工具更新提示，优先读本报告和管理员hold事实，禁止自动权限修复/取消/重投或沿用旧“全部已释放”作为当前状态；未自动恢复。总纲更新并通过catalog build/check后才称文档同步。
