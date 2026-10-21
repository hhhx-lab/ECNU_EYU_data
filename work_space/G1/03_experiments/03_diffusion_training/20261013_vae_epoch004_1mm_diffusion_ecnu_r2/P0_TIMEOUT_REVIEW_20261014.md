# P0超时终态核查

核查北京时间：2026-10-14 12:09:34。本地UTC `2026-10-14T04:09:34.429769+00:00`；远端UTC `2026-10-14T03:58:31.153059+00:00`，两端分别记录。

## 结论

**P0未完成，不能判定验收通过。** 作业`4058680`为`TIMEOUT`，Elapsed `04:00:19`；batch为`CANCELLED 0:15`。主作业显示`0:0`不覆盖TIMEOUT终态。

stderr完整101 bytes说明 `DUE TO TIME LIMIT`。stdout最后为746例pass，部分manifest3730/5150条记录；剩余284例不能视为通过。STARTED存在，VALIDATION/COMPLETE/FAILED均未生成。调度器终止可令Python来不及写FAILED，不能据FAILED不存在判成功。

当前冻结包SHA无漂移，日志节选无OOM/CUDA/Traceback/nonfinite/ImportError；这不是全量数据无问题的证明，只能确认本次中止原因是4小时上限。路径可用224180305920 bytes，非当前容量门阻塞。

## 处理边界

- 保留R1、R2、原作业、部分manifest、stdout/stderr和全部冻结审计，不原位重跑、不覆盖、不删除。
- 本轮只做一次只读快照，未重提、未建新attempt、未运行P1/P2或GPU训练。本次启动VPN已关闭。
- 原授权的受控checkpoint续训针对200k训练恢复，不能自动转移到没有checkpoint的P0验收重试。当前heartbeat配置已为PAUSED；本轮不改变其配置。
- 建议另获授权后建立独立P0 attempt，按实际I/O增加walltime（例如12h，先核验分区/QOS上限），全量复验并重新生成完整证据；若设计复用部分记录，必须先证明全量文件及源SHA未漂移、完整case事务和新旧记录唯一性，不能简单跳过746例。
- 数据/病例/标签/空间/SHA验收标准不降低。P0最终pass之前不放行1mm处理、smoke和训练。

## 证据

`operator/snapshot_p0_20261014T040908Z/P0_SNAPSHOT.json`含squeue/sacct、markers、日志尾部、冻结包SHA、容量和两端时间戳。squeue报Invalid job id是终态作业不再列队的表现，sacct已返回TIMEOUT，不当作SSH或新作业失败。

研究标记保持`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。
