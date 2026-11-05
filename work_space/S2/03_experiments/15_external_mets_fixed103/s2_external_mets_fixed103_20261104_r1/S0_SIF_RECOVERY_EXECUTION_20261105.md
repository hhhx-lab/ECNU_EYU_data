# 2025固定镜像硬链接兼容恢复

## 当前阶段

最新后续单次只读快照：远端UTC08:50:08.374584，allocation及batch RUNNING/node214；STARTED绑定4304987/合同，远端UTC08:42:54.643241，WHEELS_REUSED的三个实际wheel fresh bytes/SHA通过。无FAILED/COMPLETE、NORMALIZATION_COMPLETE/SUMMARY或native阶段报告；两日志0bytes，无漂移，仍不是完整SIF或环境/GPU通过。accounting00:08:13与scontrol00:09:06为顺序非原子观察，运行中0:0不能当完成。

[最新本机独立复核](operator/snapshot_sif_recovery_r1_20261105T090112Z/LOCAL_SNAPSHOT_REVIEW.json)：三个轻量报告bytes/SHA、执行源码SHA及重新生成一致通过；local review09:06:37.827166/finally09:01:36.157955UTC分记，自建VPN关闭，本机复核未再次SSH。余875952078848bytes=815.7939453125GiB，quota0空非数值账号额度/全量峰值通过。最终正式[验证回执](operator/verify_transport_20261105T090043Z/VERIFICATION.json)为222tests/79生产源码AST，全部源码/测试与日志SHA重新核验；085118原回执实际221tests/78AST，旧文档80AST/交接摘要81AST为计数笔误，原证据不改。GPU smoke/J01-J07和两全量均未提交，CPU/SIF、完整Bio环境及两真实GPU smoke/联合门保持。以下即时无STARTED/822GiB只为初始历史。

用户已明确批准独立恢复，见[S0_SIF_RECOVERY_AUTHORIZATION_20261105.json](S0_SIF_RECOVERY_AUTHORIZATION_20261105.json)。本次已完成实际文件身份和最小兼容验证，唯一新CPU **4304987** 已held核验并释放；即时快照allocation/batch RUNNING/node214，无Python STARTED/FAILED/COMPLETE，产物待验收。没有提前创建全量推理或GPU smoke作业。

[提交与本机独立复核](operator/submit_sif_recovery_r1_20261105T085221Z/LOCAL_RECEIPT_REVIEW.json)：sbatch0、held身份pass、本机核验pass、同IDrelease0。释放PENDING/None/零运行，后续squeue/sacct RUNNING、sacct3秒与scontrol56秒为顺序非原子观测。17项轻量审计bytes/SHA与三段实际执行源码SHA/协议AST/输入payload身份通过；初次用排序后JSON文本再生repr的比较失败单列解释，原源码不改、无第二次提交。合同SHA `d957f9e1240186436b7e8ecb607a0f47ef52e7be2b1303cd384959bffe428784`，Slurm SHA `4368622f008a106baf159dde2842723f4b387d24bcd208b1b4ad5285779df60b`。

快照远端UTC08:41:57.944124，本机held验收08:53:17.132965/finally08:53:25.183130分别记录，不称同步；本次自建VPN已finally关闭。余882873729024bytes=822.240234375GiB，quota0空；实时CPU使用保守上界104运行+164排队+8本次=276<1000，只为CPU准备资源验收，非GPU使用实测/账号数值quota或全量峰值通过。

## 原因与实际验证

- 原CPU `4304783` 仍为 `FAILED 1:0`，FAILED SHA `8a5d33b7f6900d27c8f9c282c82b08cd373330ca3dcb2f321a1520a53c4c4432`，全部旧冻结包/环境/失败保留。
- 本次重新读取实际三份wheel文件，逐bytes/SHA与原manifest一致；原归档8,409,948,160 bytes重新SHA为 `731b17a23514bf2460083d4585a42d5ece4137643960c354469790857af52735`。见[身份核验](operator/inspect_sif_recovery_capabilities_20261105T083656Z/RESULT.json)。未重编或重下载。
- 第3层普通INSTALLER在硬链接之前，第6层含 `.wh.pip-25.1.1.dist-info` 删除其目标目录，缓存硬链接仍存。Singularity CE4.1.1源码使用go-containerregistry0.19.0反序flatten，再交给umoci展开，目标删除会使保留链接无法解析。此机制已用原镜像5byte普通文件、同硬链接与whiteout在**同服务器同CE4.1.1**最小复现：原结构return255同错误，转换后return0，缓存内容正确且目标删除生效。不是NFS缺陷或原始层缺目标的实证。
- 全11层100,566个实际header、18,998个硬链接检查通过：目标均可在本层已出现的普通文件/硬链接链中解析，无不支持特殊文件、重复路径、循环或前向/跨层依赖。该扫描和最小展开不等于全镜像转换/GPU通过。见[真实兼容结果](operator/verify_sif_hardlink_compatibility_20261105T084241Z/RESULT.json)。

源码依据：[Singularity4.1.1 unpack.go](https://github.com/sylabs/singularity/blob/v4.1.1/internal/pkg/ociimage/unpack.go)、[锁定依赖go.mod](https://github.com/sylabs/singularity/blob/v4.1.1/go.mod)、[go-containerregistry0.19.0 Extract](https://github.com/google/go-containerregistry/blob/v0.19.0/pkg/v1/mutate/mutate.go)。公开源码只读取，不安装工具或保存第三方副本；一次另一源码请求超时，未据其作结论。

## 恢复合同

`attempts/s0_sif_recovery_r1` 只把原tar硬链接物化为其原始普通文件内容，交给**原CE4.1.1 OCI合并器**处理层序/whiteout/启动配置；本程序不自行展开根文件系统。使用Python标准tarfile结构API，不安装Docker、umoci或其他后端，不改变namespace/setuid/系统保护。

每层原diff_id先核验，派生层逐header顺序/元数据/全部普通文件及硬链接目标bytes与原层比较，新的layer diff_id和config SHA真实计算。原config只更新rootfs.diff_ids，Entrypoint/Cmd/Env/用户/平台与其余字段不变。**派生Docker归档与SIF不是原Docker字节摘要；inode共享不保留，模型与程序文件内容保持相同。**原manifest/归档/所有失败永久保留并作为来源身份，不伪称派生摘要等于原镜像摘要；全文件比较仍须CPU实际执行后验收。

资源保持 `cpu_192G / 8CPU / 64GiB / 04:00:00 / 0GPU / no-requeue / export=NONE`。复用 `bio_build_glibc217_r2` 和三成功wheel，不求解/安装/重编/重下载。系统/Conda/module初始化期间保留-e/pipefail，之后才-u；严格退出与绝对路径、%j日志保留。

兼容物化及原生展开/压缩/SIF封装共同计入7200秒单次deadline。每层归档累计上限16GiB，提交前实际header推算上限，最多6份同时保留副本加20GiB保守预算116GiB，仍要求原150GiB增量余量门；实时资源/配额重新核验，不用历史余822GiB给全量放行。

全量链接清单留远端 `reports/NORMALIZATION_COMPLETE.json`；本机只收 `NORMALIZATION_SUMMARY.json`、大小/SHA、轻量marker和日志尾。不下载MRI/GT/预测/模型。本次CPU完成须allocation及batch `COMPLETED 0:0` + COMPLETE + SIF/三wheel SHA，之后仍需完整Bio运行环境、两路真实GPU smoke与联合门，才可提交两全量。

## 本机检查与边界

[提交前本机验证](operator/verify_transport_20261105T085118Z/VERIFICATION.json)：221tests/78生产源码AST pass，含完整恢复入口、原失败守卫、一次held、本机资源/脚本/传输SHA核验、同IDrelease、重复/不明JobID/漂移/失败拒绝及完成后大ledger只传身份。此前两次221测试失败为新增stress夹具未解除自身只读权限、夹具JobID不一致；均在SSH/提交前发现并只修测试，日志085007/085053保留。最初新最小probe测试缺模拟operator父目录也只在本机修夹具，不改远端程序。此pass不是完整SIF/模型/A100或科学质量pass。

身份快照远端UTC08:25:49.165439至08:26:15.629429；最小验证结束远端UTC08:31:40.607754，本机接收/会话finally见各LOCAL_OPERATION_RECEIPT。两端时间差原样保留，不称同步。两次自建VPN均按finally只关闭本次会话；凭据只由原指定env加载，网络操作器在Conda base/Chrome Bridge运行。

`experimental_unvalidated / operator_approved=false / formal_gate_status=not_run_not_passed`，不恢复监控、不自动retry。
