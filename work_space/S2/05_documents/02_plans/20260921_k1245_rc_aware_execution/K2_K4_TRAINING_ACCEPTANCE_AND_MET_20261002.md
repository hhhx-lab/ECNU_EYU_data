# K2/K4 175轮训练验收与官方MET推进

artifact_status=experimental_unvalidated
operator_approved=false
formal_gate_status=not_run_not_passed

验收时间：本机UTC `2026-10-02T03:25:09Z`、远端UTC `2026-10-02T03:12:52Z`。同一fixed103用于checkpoint选择和报告，结果仅作exploratory。组合实验不能把差异归因给单个机制。

## 训练结果

K2 job `3827555`、K4 job `3827580`均已 `COMPLETED 0:0`，TRAINING_VALIDATION status=pass、COMPLETE存在、FAILED不存在。逐个核验两路共14个候选checkpoint：文件SHA与训练验证记录一致，current_epoch正确、trainer正确，网络、优化器、AMP、日志均finite。模型plans及dataset.json在候选间一致，冻结标签为background=0、NETC=1、SNFH=2、ET=3、RC=4。

按候选EMA最大且并列取更早epoch，两路最佳均为175。

| 累计epoch | K2 EMA fg Dice | K4 EMA fg Dice |
|---|---:|---:|
|25|0.729281|0.621915|
|50|0.745226|0.744624|
|75|0.723971|0.738427|
|100|0.746882|0.751560|
|125|0.755079|0.749779|
|150|0.763073|0.750922|
|175|0.768438|0.754988|

K2 selected SHA256：`6f82992015854e5b4a85bd5ccfe4891260d990bbb69003a27729cdeec8973f91`。

K4 selected SHA256：`8bfb5bfdfc6a8b578e2ba1489bfa958655f960e9f8d6a9dfaede29af4b340be8`。

nnU-Net训练结束时全病例验证 Mean Validation Dice：K2 `0.5527734486`、K4 `0.5566582739`。这是nnU-Net验证口径，与EMA不同；不能当作官方MET病灶Dice、F1、FP/FN或paired结果。

## 拉回证据

[只读验收与延长操作目录](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/extend_k1_accept_k2_k4_20261002_20261002T032211Z)保存：

- ACCEPTANCE_AND_EXTENSION.json：完整选点、候选SHA、finite与模型合同。
- PULLED_SHA256.json：32份轻量文件的SHA与大小，拉回后验证JSON可解析。
- pulled/K2、pulled/K4：计划、config、evidence、Slurm、stdout/stderr及完整training_log；不包含checkpoint、NIfTI、events JSONL。
- LOCAL_OPERATION_RECEIPT.json：本次VPN启停与本机UTC。

## K1时限

K1 job `3827543`在核验时仍RUNNING，实际TimeLimit为5天。用户选择延长原作业，执行唯一 `scontrol update JobId=3827543 TimeLimit=9-00:00:00` 请求，服务器返回 `Access/permission denied for job 3827543`，再次读取同一job合同确认仍为5天。因此未实现延长；需要集群管理员操作。

管理员可审核并执行：

```bash
scontrol update JobId=3827543 TimeLimit=9-00:00:00
```

原调度器EndTime为 `2026-10-04T10:40:49`（服务器当地时间，服务器时钟较本机慢约11分钟）；应在该期限前处理。本轮未提交K1续训作业。

## 官方MET批次

独立批次：[s2_met_aug_k24_selected103_official_met_20261002_r1](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k24_selected103_official_met_20261002_r1)。

使用上述两路选定checkpoint、冻结fixed103真实T2W四模态输入及原标签；预检校验515份输入/标签SHA、1035/103训练split对应关系、评价版本与CPU严格加载模型。每路单A100非exclusive；sequential_in_process、preprocess_workers=0、export_workers=0、multiprocessing_queue_used=false。官方评价器BraTS-evaluation=0.0.8、panoptica=2.1.0、numpy=1.26.4，mets、vol_threshold=27、overlap_threshold=0.2。

部署最初在远端执行前返回 `Argument list too long`，没有预检或job；原因是轻量文件内嵌命令过长。失败操作证据保留，部署器改为SCP目录传输后成功。远端UTC `2026-10-02T03:22:01Z` REMOTE_STATIC_PREFLIGHT status=pass，515份输入/标签SHA、训练split映射、两路CPU网络严格加载及评价版本通过。

冻结计划SHA256：`9d84282fec75fe37822764cd8d746e67aec721aa1769e352c6b5cc2f9decff32`。

| 路线 | 官方MET job | 名称 | 本次提交验收状态 |
|---|---|---|---|
|K2|3853018|s2k2met103r1|PENDING (Resources)|
|K4|3853019|s2k4met103r1|PENDING (Priority)|

[提交验收快照](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/k24_met_submission_acceptance_20261002_20261002T033325Z/SUBMISSION_SNAPSHOT.json)时间：本机UTC `03:33:33Z`、远端UTC `03:22:42Z`，两路无STARTED/FAILED。其pulled目录拉回了预检、模型快照摘要、launch/released及SHA索引；两个GPU作业各提交一次、写入held审计后释放，尚未报告官方MET完成。正式MET指标、paired及K5科学门均未通过本文件授权。
