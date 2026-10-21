# K1 累计175轮 checkpoint 续训准备

新批次 `s2_met_aug_k1_175_resume_20261002_r1` 只恢复原 K1 job `3827543`，不停止、不重排当前作业。原作业尚在 RUNNING，不能提交续训。所有新产物保持 `experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 文件与执行顺序

| 文件 | 用途 |
|---|---|
| `EXECUTION_PLAN.json` | 原作业身份、冻结代码/合同SHA、累计175轮预算与超时后唯一提交规则 |
| `PACKAGE_SHA256.json` | 新运行包逐文件SHA，不包含可变checkpoint与作业输出 |
| `operator/inspect_source_once.py` | 已执行的一次源作业/完整checkpoint只读验收入口 |
| `operator/build_package.py` | 从已验收远端原Slurm派生环境和SHA合同；只构建一次 |
| `operator/access_resume.py` | 每次显式启动分流VPN，单次远端操作后只关闭本次VPN；stage或TIMEOUT后提交 |
| `runtime/research_environment.sh` | 复用冻结原K1研究合同与运行代码，只改变续训标志和新输出路径 |
| `runtime/resume_contract.py` | SHA、完整状态、真实trainer恢复、历史与下一轮LR验收 |
| `runtime/test_restore.py` | 新进程中的CPU恢复探针，隔离测试输出与训练结果 |
| `runtime/control_resume.py` | stage只测试不提交；TIMEOUT后拷贝最终checkpoint、保留原证据、唯一held launch审计 |
| `runtime/run_resume.py` | 调用安装版nnUNet CLI的`--c`，禁止缺checkpoint回退从头训练；GPU恢复后写STARTED |
| `slurm/01_resume_k1_175.slurm` | 单A100、8CPU、64GB、5天，日志与结果全在新根 |

本地入口使用Conda base及约定env：`operator/access_resume.py stage` 仅准备远端；`operator/access_resume.py submit-after-timeout` 每次仅取一次源squeue/sacct，RUNNING时直接返回不提交，TIMEOUT后才验收与held提交一次。不得重复stage、重复sbatch或删除reservation；失败后保留审计，不自动新建attempt。

## 恢复与验收

保持原trainer、1035/103 split、seed20260724、LR0.001/poly0.9累计175衰减、SGD/momentum0.99/nesterov/weight_decay3e-5、原数据与RC27/G1/增强合同、workers0、compile0、非DDP、每25轮存点。命令语义为 `nnUNetv2_train 264 3d_fullres 0 -tr nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175 -num_gpus 1 --c`，不传`-pretrained_weights`。

原checkpoint与事件/日志绝不被续训写入。超时后对最终latest及已存候选逐个SHA/current_epoch/finite验收，复制到独立结果目录；latest副本0600，其余旧候选副本0400。保留原作业TIMEOUT、stdout/stderr、STARTED、研究合同及日志快照；原作业结果不改写。

CPU真实trainer测试只证明网络、优化器、logger、current_epoch和175曲线恢复；CPU不能证明CUDA AMP加载，AMP须在新A100作业启动时逐字段验证。没有GPU恢复PASS不得称训练已恢复。训练完成须候选25/50/75/100/125/150/175齐全、状态finite、103例nnUNet validation、最终TRAINING_VALIDATION与COMPLETE；官方MET和paired另行门控。

现有保存频率25轮意味着latest之后所有未保存轮次都重算，不限于超时中断的单轮。今天latest50只是诊断，不能作为最终冻结恢复输入。若原作业意外COMPLETED 0:0，不续训；若出现非TIMEOUT失败、关键错误、SHA/合同漂移或作业消失，停止并通知。

`F_STRICT_PREREG_GATE=false`保持；同fixed103选择与报告仅exploratory，K5科学门未满足。本批不授权K5、不更改原五路equal-budget结果、不自动延长225轮。
