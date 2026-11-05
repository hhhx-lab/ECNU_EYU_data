# SIF 恢复 R2：实际验证和 CPU 提交

## 2026-11-05 22:40 最新查询

CPU4309579 allocation/batch RUNNING/node225，sacct01:12:32、scontrol01:13:25是顺序观察，0:0不是完成终态。原生CE展开成功，日志native_rootfs Build complete，阶段1于remote UTC13:59:50.868280进入mksquashfs压缩；暂无FAILED/最终COMPLETE或身份漂移。七轻量报告bytes/SHA、执行源码再生、STARTED合同及三wheel复用报告通过，详细规范化ledger留远端。remote snapshot14:28:32.072525/local finally14:39:59.486201分记，用户VPN保留；余819276546048bytes=763.0107421875GiB非数值quota/全量峰值门。只读查询不改/重提作业或安装、不读MRI-GT-test；完整SIF/Bio环境/两真实GPU smoke及全量未完成/提交。见[快照复核](operator/snapshot_sif_recovery_r2_20261105T143947Z/LOCAL_SNAPSHOT_REVIEW.json)。以下为此前阶段历史。

## 2026-11-05 21:50 最新查询

CPU4309579 allocation/batch均RUNNING/node225，sacct00:22:44、scontrol00:23:37是顺序观察，运行中0:0不是成功终态。11层/18998硬链接规范化已完成，逐成员bytes/metadata比较通过；8850720-byte详细ledger仅留SHA在轻量快照，原件留远端。CE阶段0在remote UTC13:28:48.146751启动，日志目前为Extracting OCI image；暂无FAILED/最终COMPLETE或身份漂移，完整SIF尚未完成。

六轻量报告bytes/SHA与实际执行源码重新生成一致，STARTED绑定本JobID/原合同，三wheel fresh SHA复用报告通过。远端snapshot13:38:41.953739、本机finally13:50:09.433781UTC分记；用户VPN保留，文件系统余833090486272bytes=775.8759765625GiB，非数值quota/全量峰值门。只查询一次，无远端写入/重提/安装solver/重编下载/读MRI-GT-test或恢复监控。完整Bio环境/两真实GPU smoke/J01-J07/两全量未完成或提交。证据：[快照复核](operator/snapshot_sif_recovery_r2_20261105T135002Z/LOCAL_SNAPSHOT_REVIEW.json)。以下为13:26提交阶段历史，不覆盖本次RUNNING观察。

本地复核 UTC：2026-11-05T13:26:37Z。`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 当前状态

用户已确认“允许独立R2验证后继续”，并完成校园登录。服务器同 Python 验证通过后，唯一新 CPU 作业 **4309579** 已 `sbatch --hold`、独立 held 核验、同 ID `release` 成功。即时快照为 **PENDING / Resources**，运行时间零，未分配节点；不是 RUNNING，不是等待 GPU。

本次完成了 R2 验证和 CPU 提交。完整 2025 SIF、完整 Bio-Intel 推理环境、两路真实 GPU smoke 及联合门仍未通过，原 J01–J07 与两全量推理未提交。原 4304987/4304783 和全部旧失败不改。

## 输入与本机检查

复用原镜像归档、三个已成功编译的 wheel、原构建环境、同版模型/依赖及资源。R2 仅接续已验证的空层迭代器候选，原冻结 normalizer 不修改。

本机使用既有 Conda 环境，没有安装新依赖：

| 检查 | 实际结果 |
|---|---|
| brats2023_seg 完整回归 | 246 项发现，244 pass，2 个 Python3.10 限定 skip；84 生产源码 AST |
| g1_t2w_bbdm Python3.10.20 空层检查 | 8 项 pass，无 skip |
| 同3.10生成的远端验证入口控制流 | 2 项 pass，无 skip；native 调用为模拟 |
| 同3.10完整 R2 提交流程 | 14 项 pass，无 skip；冻结/held/释放/重复/失败/漂移门为模拟 |

两个完整回归 skip 均已在上述 3.10 检查单独通过，不把 skip 算实测 pass。完整 Slurm 的 bash -n、profile/module 后启用 nounset、绝对路径、含 `%j` 日志、资源和同 ID 释放流程已检查。

源码及测试/日志身份见 [完整回归](operator/verify_transport_20261105T132018Z/VERIFICATION.json)与[空层3.10回归](operator/verify_sif_empty_layer_20261105T132038Z/VERIFICATION.json)。本机测试不等于服务器完整模型或 GPU pass。

## 服务器验证

一次 `verify_sif_empty_layer_remote_r2.py` 在原构建环境 Python3.10 执行。原空层 Errno22 复现，候选完整八项、11层空第四层/硬链接/whiteout、额外/缺失/内容篡改/deadline拒绝均通过；现有 CE4.1.1 最小展开成功，文件内容/whiteout及 runscript 核验通过。没有尝试完整镜像转换或加载模型。

远端验证 COMPLETE 与本机 RESULT 的规范化 SHA 均为 `b61741848dc75fc97813e72c8ee7984ac266ed2d0e721017bbb581f57631473b`。实际执行源码 SHA 和重新生成内容一致，用户手动 VPN 复用且未注销。见[有限验收](S0_SIF_RECOVERY_R2_PYTHON310_ACCEPTANCE_20261105.json)。

## 冻结与提交

远端 attempt：

`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_external_mets_fixed103_20261104_r1/attempts/s0_sif_recovery_r2`

冻结 Slurm：`slurm/prepare.slurm`；日志：`logs/s2exts0sifr2_4309579.out` 和 `.err`。资源 cpu_192G / 8 CPU / 64 GiB / 4 h / 0 GPU，无依赖、禁止 requeue。

- 合同 SHA：`2ae261ab3ccaa72b84e90fc105d3275e1b5bbd94d8c7300af98d987ce3938c20`。
- Slurm SHA：`2399c321e002bd1241c42b6dadeca8fe5642e92f1bcc82baf18ecc246b60c56d`。
- 候选 normalizer SHA：`b1535f17f33b073f3a9d9f94701045e98faf5db4372e742e64f05da3acde4748`。
- held、独立本机核验和同 ID release 均通过，release 返回0；17项轻量审计 bytes/SHA、三段实际执行源码 SHA/定义/协议输入及冻结脚本独立再核通过。

一次即时只读快照：squeue/sacct/scontrol 均 PENDING，原因 Resources，无 Python STARTED/FAILED/COMPLETE，只有 STAGING_COMPLETE；静态及父失败身份无漂移。运行中/排队时的0:0不构成完成证据。详见[提交目录](operator/submit_sif_recovery_r2_20261105T132420Z/LOCAL_HELD_VERIFICATION.json)、[即时快照](operator/submit_sif_recovery_r2_20261105T132420Z/SNAPSHOT.json)和[独立复核](S0_SIF_RECOVERY_R2_SUBMISSION_REVIEW_20261105.json)。

## 时间、容量与边界

| 事件 | 观测 UTC |
|---|---|
| 服务器最小验证结束 | 2026-11-05T13:11:54.955355+00:00 |
| 本地验证操作结束 | 2026-11-05T13:23:22.290036+00:00 |
| 服务器 held 回执 | 2026-11-05T13:13:35.711975+00:00 |
| 本地 held 验收 | 2026-11-05T13:25:04.524219+00:00 |
| 服务器 release 返回0 | 2026-11-05T13:13:44.164453+00:00 |
| 服务器即时快照 | 2026-11-05T13:13:50.833218+00:00 |
| 本地提交操作结束 | 2026-11-05T13:25:18.339937+00:00 |

两端观测时间分记，约11分钟差，不称时钟同步。两次操作均复用用户 VPN，未 stop/logout。快照文件系统余 `857349292032 bytes = 798.46875 GiB`；quota 命令返回0但没有数值，只通过 CPU 的150 GiB增量门，不代表账号数值额度或全量并发峰值已通过。实时保守 CPU 上界140运行请求+118排队请求+8本次=266，低于1000；不是实际 GPU 使用量。

独立本机复核首个临时命令错误地将 print/json.dumps 包装节点当 execute 参数，AST literal_eval 失败；只修正本机 AST 导航后全部复核通过，未再次 SSH、sbatch 或 release，未改变远端源码/原始回执。

不重编、不重下载、不重装、不重求解、不自动 retry、不恢复监控，不读 MRI/GT/test。CPU 完成仍须 allocation/batch COMPLETED0:0、COMPLETE、完整 SIF/内容身份及三 wheel验收。随后仍须完整 Bio 环境、两路真实 GPU smoke 和联合容量/耗时门，才可提交原全量。
