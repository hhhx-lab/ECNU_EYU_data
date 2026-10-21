# K1累计175轮续训提交验收

用户于2026-10-07明确要求“立刻提交续训”。本次使用此前部署的独立`s2_met_aug_k1_175_resume_20261002_r1`，源job `3827543`仍保留原`TIMEOUT 0:0`结果与全部原证据。

| 项目 | 验收结果 |
|---|---|
| 新job | `3919543` / `s2k1res175r1` |
| 资源 | A100分区、单GPU、8CPU、64GB、5天、非exclusive/非DDP |
| 提交与释放 | 唯一`sbatch --hold --parsable` rc0；LAUNCH_AUDIT记录canonical job ID；`scontrol release` rc0 |
| 提交后即时状态 | `PENDING`，squeue正确名称/时限；sacct暂为空，无已验证GPU启动 |
| 恢复输入 | 超时后最终`checkpoint_latest.pth`，`current_epoch=100`，249775050 bytes |
| latest SHA256 | `67911689e637557c5ee3b0c43c2d3bddc693a3eaf6e11fc39a9f7015b96773dd` |
| 完整状态 | 网络292张量、优化器106项、AMP/日志finite，175历史LR与SGD合同pass |
| 最终CPU真实恢复 | 网络/优化器/logger/current_epoch一致，175 horizon与save_every25通过 |
| epoch100下一轮LR | `0.00046646696566721586` |
| 候选与输出 | 源25/50/75/100候选验收并复制；latest副本0600，候选副本0400，日志/事件/结果新根独立 |
| 学习率与数据 | 保持原累计175曲线、1035/103 split、seed20260724、原K1 RC27/G1/增强配置 |
| 后续预算 | 从100继续到累计175，共75轮；源100之后未保存轮次重算 |

源终态即时核验远端UTC `2026-10-06T16:37:16.778523Z`；最终CPU预检远端UTC `16:38:34.030014Z`；held audit `16:38:34.194592Z`；释放 `16:38:34.274071Z`。本地收到提交结果UTC `16:49:31.090106Z`，Asia/Shanghai为2026-10-07 00:49:31；本次VPN关闭回执UTC `16:49:37.062915Z`。远端时钟约落后11分钟，应分别保留时间戳。

本次`squeue`对已离队源job返回空输出和精确错误`slurm_load_jobs error: Invalid job id specified`。补充提交入口仅在`sacct`明确匹配源job名、ID、TIMEOUT后接受这一终态读取，保留原始rc与stderr。原部署的11份文件SHA、计划SHA`7f1de71844ee75e8ff78831acd01f4f6281813996adb65250fe61e51c931e5f6`及Slurm SHA`66ca687ae8901bacfd289b79463ef7d85214cb9a1d9194d7815b7d7692d5a15f`均保持；原研究/运行包没有覆盖。新增本地入口`operator/submit_final_once_20261007.py`记录该受限终态读取与用户本次提交授权。

轻量证据目录：[本次提交审计](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/k1_resume_final_submit_20261007_20261006T164507Z/OPERATION_RESULT.json)。已拉回8份JSON，逐文件SHA/JSON校验通过，包括最终源验收、最终CPU预检、reservation、held launch、release和提交验收；没有拉回checkpoint/NIfTI/events JSONL。原TIMEOUT证据副本存新远端根，SHA绑定保留。

截至本次提交后即时快照，新job仍PENDING，不能写成实际训练循环已恢复；CUDA AMP/网络/优化器/logger恢复与`K1_175_RESUME_STATE_RESTORE_PASS current_epoch=100`须调度启动后验收。最终训练须补齐125/150/175及final、TRAINING_VALIDATION和COMPLETE，再按原门控选点/官方MET/paired。

所有新产物`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`；`F_STRICT_PREREG_GATE=false`，同fixed103选择与报告仅exploratory，K5科学门未满足。
