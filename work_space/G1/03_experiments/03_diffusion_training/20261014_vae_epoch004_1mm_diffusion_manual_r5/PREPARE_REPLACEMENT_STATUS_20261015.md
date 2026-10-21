# R5预处理替换成功：不再等待管理员释放

后续最新运行状态：[共同mask覆盖门失败](PREPARE_FAILED_MASK_COVERAGE_20261015.md)。4086739已FAILED1:0、00:05:46；四训练均CANCELLED零运行，未训练。此文保留替换/释放成功的原提交事实，不能再用下文PENDING判断当前状态；heartbeat已PAUSED，不自动重提。

用户要求立刻进入训练提交，授权只替换无法由账号释放的预处理作业，保留四个已提交A100训练。此次唯一sbatch成功，依赖全部迁移后取消旧CPU，再释放新CPU；没有重新提交GPU、复制源数据、改冻结runtime或尝试转换管理员hold。

- 本地完成UTC：2026-10-15T07:06:10.380506+00:00（北京时间15:06:10）。
- 远端完成UTC：2026-10-15T06:55:06.107940+00:00；hostname=login1。两端时间分别记录。
- [完整轻量回执](operator/replace_prepare_cpu192_20261015T070559Z/REPLACEMENT_RECEIPT.json)含静态门、held canonical审计、四依赖更新/复核、旧取消证明及释放响应；失败历史全部保留。

| 阶段 | job | 资源/依赖 | 释放后状态 |
|---|---|---|---|
| 新预处理 | 4086739 | cpu_192G，8CPU、90G、24h | PENDING，Reason=None，运行0 |
| T1C | 4069916 | 单A100，afterok:4086739 | PENDING，运行0 |
| T1N | 4069917 | 单A100，afterok:4086739 | PENDING，运行0 |
| T2W | 4069918 | 单A100，afterok:4086739 | PENDING，运行0 |
| T2F | 4069919 | 单A100，afterok:4086739 | PENDING，运行0 |
| 旧预处理 | 4069915 | 已取消，零运行 | CANCELLED证明已保存 |

释放返回0，新CPU不再JobHeldUser/JobHeldAdmin/BadConstraints；四GPU逐项确认原资源和新afterok，迁移时新CPU始终held，旧CPU取消后才释放。GPU队列Reason=None为同一即时观测，不代表依赖已满足；scontrol审计仍明确afterok新CPU。**已提交并释放，不等于已开始训练**，不能保证排队开始时间。预处理成功退出后四训练自动调度，无需管理员release旧ID。

新脚本只改分区、job name和日志前缀，独立保存于远端evidence/replace_prepare_cpu192_20261015T070559Z/prepare_cpu192.slurm；SHA=3c6333d443da326cfaeb3938e71d61f0bc3ec2644ef4c5c932fc5b26f422cb12。冻结57文件packageSHA=13f188b2609ba57544c1258aa368daeabe0201ecbdb81c3ec86700f6441b6ede，逐文件/源CSV/原脚本与派生脚本/bash-n/bash-u通过。cpu_192G扣预留容量满足；账号余224162480128bytes（208.77GiB），仍覆盖冻结198.85GiB需求，不引用共享盘余量替代。

本机6个离线测试方法通过，含5种提交前停止条件、一次QOS拒绝、部分依赖迁移停止、取消拒绝和释放拒绝；模拟不代替真实权限，本次实际sbatch/update/scancel/release返回0才确认操作完成。操作VPN正常结束；仅本地轻量JSON，无NIfTI/checkpoint下载或Drive上传。新增根级排他reservation/active mapping；后续不得再次运行替换操作器或原submit。snapshot.py读取active mapping，区分旧CPU主动取消与新失败，继续一次即时快照。

用户独立P0/P1人工/P2等待豁免仍waived_by_user，不伪造验收pass。共同源mask/strict1mm/finite/合法标签/消耗SHA和train823/val103/test104锁定不改；四路原单A100/64³/batch16/AMP/eager/seed20261013/AdamWlr2e-4beta1=.9/各200k成功更新及完整checkpoint合同不改。科学结果仍experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed，仅exploratory。

恢复原30分钟heartbeat，以4086739及四原GPU为活动目标；无变化安静，新失败保留证据停止通知，不自动重投。受控TIMEOUT恢复须按原完整checkpoint/合同及独立审计门。总纲同步以catalog build/check通过为准。
