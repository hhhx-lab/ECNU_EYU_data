# K1累计175轮续训终态即时快照

2026-10-08一次有界只读远端快照：远端UTC `03:52:03`（`login1`），本地UTC `04:03:01`。仅查询已有job `3919543`，没有提交、续跑或修改远端作业；本次启动的分流VPN在操作结束后关闭。Slurm返回的无时区时间字段不与上述UTC直接比较。

| 检查项 | 即时结果 |
|---|---|
| `squeue` | 作业已离队，返回空输出和 `Invalid job id specified` |
| `sacct` | `3919543` / `s2k1res175r1` 与batch均为 `COMPLETED 0:0`；分配单GPU、8CPU、64GB，历时 `1-06:33:09` |
| 恢复证据 | `RESUME_STARTED.json` status=pass，canonical job ID正确，从`current_epoch=100`、SHA `67911689e637557c5ee3b0c43c2d3bddc693a3eaf6e11fc39a9f7015b96773dd`恢复；GPU上AMP恢复核验为true，使用`--c` |
| 训练终态 | `RESUME_FAILED.json`不存在；`TRAINING_VALIDATION.json` status=pass、completed_epochs=175、nnunet_validation_cases=103；`RESUME_COMPLETE.ok`内容为`complete` |
| checkpoint | 25/50/75/100/125/150/175候选和final文件均存在；训练验证摘要选epoch175 |
| 日志 | 最新日志进入Epoch174（零基编号，即第175轮），随后完成nnU-Net验证；stdout末尾`Validation complete`，Mean Validation Dice `0.551110498438361`；stderr仅见locale和transformers弃用警告，未见关键错误 |

本次为训练进度与终态快照，不是全部checkpoint逐文件SHA复算或官方MET验收。`best_epoch=175`来自训练验证摘要，仍需按冻结合同复核checkpoint选择并做fixed103官方MET和paired比较。原作业`3827543`的TIMEOUT记录不改写；K5仍需K1/K2各自科学门通过。全部成果继续标记`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`；同一fixed103用于选点与报告，只能作exploratory。
