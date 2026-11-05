# Bio-Intel 独立 R2 全量提交

## 授权和输入

用户明确回答“允许独立提交 R2”。仅一个新Bio-Intel full103 held作业、独立核验、同JobID首次释放。2025的4310575不重提/不再释放；原Bio R1及全部失败永久保留。不重装、求解、GPU smoke、自动retry或修改模型/版本/数据/资源。

复用R1原13文件payload，SHA256 `c1c909050edd74ce49859b0d0332b23339680fd168bb7643d9d3cba97f534dc6`；七份运行源码和两份固定清单完全不变。只更新attempt为`full103_bio_r2`、合同/授权与Slurm路径身份，并重算冻结包SHA。合同绑定源Git `4b1916ccdf123d0824d2e13f677c2609ebbc01c2`。已独立验收R5 metadata SHA `812e15d2620646414ac335f57ced1c73106f824c189d19697780aa527313e679`；环境、完整A0–4/B1,3/BrainIAC、原融合/后处理/通道不改。

固定103真实四模态、69患者组、515输入参考身份；模型仅消费412影像，GT目录不挂载，不读test104/official179/val27。原公开训练暴露继续披露。单A100/8CPU/64GiB/48h、no-requeue、export NONE。

## 执行流程

1. 本机校验原payload/metadata及每个成员SHA，重生Slurm与原脚本仅路径/合同SHA差异等价，`bash -n`通过。
2. 远端在新attempt创建前逐SHA核R1包/原失败，确认无成功/held/release证据；实时queue和Bio accounting均无遗留作业。核2025保护包和4310575映射，只读不改。
3. 冻结新包，复核完整环境/资产身份、既有控制器与容器CPU imports、实时分区/QoS/自身资源及剩余存储；预留150GiB非实测峰值。
4. 唯一一次`sbatch --hold`，记录原始stdout/stderr/returncode。成功才独立本机核验held的JobID、资源、路径、合同/包和审计SHA；再fresh身份复核并同ID首次release。
5. 失败或返回含糊则留证停止，不重试、不生成R3。完成提交后仅一次轻量状态检查，不恢复后台监控。

## 本机验证与当前执行

[完整验证](operator/verify_transport_20261105T174627Z/VERIFICATION.json)：292发现、290pass、2原版本限定skip，100生产源码Python3.10 AST。新增父失败/保护2025漂移、queue/accounting已有Bio、运行源码/清单不变、路径/合同及包SHA回归。测试不当GPU或生成/分割质量通过。

**本次R2唯一sbatch返回1，没有JobID，也未进入held核验或release。** 原始stderr为：

```text
sbatch: error: Job refused: account zqchen have not enough available CPU time. Please contact the administrator!
sbatch: error: Batch job submission failed: Unspecified error
```

见[本次失败回执](operator/submit_bio_full103_r2_20261105T174640Z/HOLD_RESULT.json)和[独立失败复核](BIO_FULL103_R2_FAILURE_REVIEW_20261106.json)。这是调度器本次明确拒绝账号可用CPU机时，不是本机语法错误或R5环境验收失败；具体剩余机时数值与后台计费/配额规则未能验证。原R1 stderr漏记，其原因仍未知，不以本次错误改写原R1。

R2 13文件包SHA `365120b27ddf2c03ca183119c453206fae53e8a9ddaa0b3acadb67aac13f1a73`，合同SHA `95df0f5461676ec99f8f967505dd10740e34bb0a1eb6d7bec98da274212c9361`，payload SHA `a67e167e2f155cdc1f602449fbad46461808e57fec54709e9d33032699d37e69`；实际执行提交源码SHA `63e79f3d1a56126e2ec3e893ebe4316666aef3370840d7414f4900ed1da3476a`。原始非零命令回执SHA `9c1f30e3b76d0aa2aeeba53d8d90bd0cd038c74c546cdd2ce19c066fac19f3cc`，FAILED SHA `406626ad25fa72fb5d159e5364758dce1365f7fafb8797debd972e5d33f7a663`。原失败/保护2025身份及提交前资产/环境门通过，未据此称成功入队。

唯一提交后的[只读快照](operator/snapshot_direct_preparation_20261105T174833Z/RESULT.json)确认Bio queue/accounting为空、无STARTED/FAILED/COMPLETE或预测；2025的4310575仍PENDING/Priority、运行零、0预测。Slurm accounting旧JobHeldUser为滞后字段，2025实际已首次释放，不重复释放。15份轻量回执重新编码SHA、本次失败与源代码/原九份运行及数据成员等价、两端收尾时间独立验收通过。

CPU请求保守上界221<user_zqchen1000，18在队<partition_a100 MaxSubmitPU100、a100 UP，**这些是并发资源限额，不是可用CPU机时额度**；原预检不代表机时余额足够。快照文件系统余744630517760 bytes=693.4912109375 GiB，quota0空，非数值账号存储配额或实测峰值。存储余量不代表可提交机时足够。

远端失败UTC `2026-11-05T17:35:59.861887+00:00`；远端快照UTC `17:37:11.226463`；本地提交finally `17:47:27.214180`、快照finally `17:48:39.036081`、复核 `17:49:44.276688`分别记录，不宣称时钟同步。用户VPN保留，未下载安装/修改原环境/模型、未读MRI/GT或回传模型影像，未新增R3/自动retry或监控。

后续须先由管理员核验并恢复账号可用CPU机时，再由用户授权新的独立提交；不得改变时限/CPU/模型/数据绕过本次拒绝。本次R2已消耗一次提交授权，原失败永久保留。

归档后本机回放曾有一项错误路径测试未固定历史Git HEAD，实际先触发合法的授权HEAD检查而不符该测试预期；只修正本机测试夹具，原失败留见[回放审计](LOCAL_R2_POSTCOMMIT_REPLAY_20261106.json)。最终[复验](operator/verify_transport_20261105T175432Z/VERIFICATION.json)仍292发现/290pass/2原版本skip、100生产3.10 AST。生产操作器/冻结包/服务器状态均不改，无再次SSH或提交；本机测试源码仅本地保留不提交。

环境：本机Conda base运行操作器，brats2023_seg跑测试；凭据仅从约定env加载，显式`ECNU_SERVER_ENV_FILE`精确路径、Chrome Bridge。session显式调用分流脚本start/check，finally只关闭本次自建VPN，用户既有会话保留。

科学状态保持`experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed`；GPU smoke和实测耗时/峰值门保持`waived_by_user_not_pass`。推理完成须`sacct COMPLETED 0:0`、同JobID/合同COMPLETE及103预测native/标签/SHA验收；本次不是统一评分或科学质量结论。
