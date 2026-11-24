# RC NOSHM 折重排记录

状态：`experimental_unvalidated`；`operator_approved=false`；`formal_gate_status=not_run_not_passed`。

## 结论

原 17 个折均因节点 `/dev/shm` 空闲不足，在训练启动前以 `FAILED 75:0` 退出。本次按冻结包 `s2_rc_train_batch1_20261112_r1` 的 `operator/submit.py noshm` 逐折新建一次 held 作业，排除各折此前失败节点；逐条核验冻结包 SHA、run 身份、排除节点、资源参数、held 状态后，使用原 `operator/release.py` 放行。

**重排已成功：17 个训练 JobID 和 17 个配套汇总 JobID 均已提交并释放。尚未进入训练：最近一次远端只读快照显示 17/17 个训练作业均为 `PENDING`、0 个 `RUNNING`，没有新 `STARTED` 或新 `NOSHM` marker。** 因此，内存盘问题尚未被实测证明解决；本次措施是避开每折原来失败的节点，不是修改节点配置或降低内存门。

## 根因与处理

原始记录显示启动器要求约 42 GiB 数据空间加 20 GiB 余量，即约 62 GiB `/dev/shm`；各折等待 1200 秒、21 次检查后仍不足。历史空闲量为 B0p 在 `gpu15` 约 44 GiB、BL 在 `gpu12` 约 17 GiB、BU 与 BUL 后两折在 `gpu13` 约 3 GiB。训练子进程没有启动。这个门检查的是计算节点共享内存盘，不是 GPU 显存，也不是账号 NFS 文件系统剩余空间。

本次保留冻结训练配置：A100 单卡、16 CPU、128G Slurm 内存、24 小时、原训练器/数据/折/300 epoch 与依赖链不变；未启用 NFS fallback、未改训练代码、未续训旧 checkpoint、未改动已完成的 B0 五折和 BUL 前三折。原失败 JobID 与 marker 永久保留。每折本次为第 1 次 NOSHM 重排，冻结包最多允许 3 次；若新节点再次写出 NOSHM，停止并报告，不自动再试。

提交前远端目录可用空间约 441.18 GiB，冻结提交器门槛为 30 GiB；这只是文件系统空间检查，不表示 `/dev/shm` 充足。当前训练作业因优先级或前序折依赖排队，未观测到新的共享内存检查结果。

## 作业对照

| Run | 原失败 JobID | 新训练 JobID | 汇总 JobID | 排除节点 | 最新状态 |
|---|---:|---:|---:|---|---|
| B0p_f0 | 4389556 | 4452757 | 4452758 | gpu15 | PENDING / Priority |
| BL_f0 | 4389561 | 4452760 | 4452761 | gpu12 | PENDING / Priority |
| BU_f0 | 4389566 | 4452762 | 4452763 | gpu13 | PENDING / Priority |
| B0p_f1 | 4389557 | 4452764 | 4452765 | gpu15 | PENDING / Dependency |
| BL_f1 | 4389562 | 4452766 | 4452767 | gpu12 | PENDING / Dependency |
| BU_f1 | 4389567 | 4452768 | 4452769 | gpu13 | PENDING / Dependency |
| B0p_f2 | 4389558 | 4452770 | 4452771 | gpu15 | PENDING / Dependency |
| BL_f2 | 4389563 | 4452772 | 4452773 | gpu12 | PENDING / Dependency |
| BU_f2 | 4389568 | 4452774 | 4452775 | gpu13 | PENDING / Dependency |
| B0p_f3 | 4389559 | 4452776 | 4452777 | gpu15 | PENDING / Dependency |
| BL_f3 | 4389564 | 4452778 | 4452779 | gpu12 | PENDING / Dependency |
| BU_f3 | 4389569 | 4452780 | 4452781 | gpu13 | PENDING / Dependency |
| BUL_f3 | 4389574 | 4452782 | 4452783 | gpu13 | PENDING / Priority |
| B0p_f4 | 4389560 | 4452784 | 4452785 | gpu15 | PENDING / Dependency |
| BL_f4 | 4389565 | 4452786 | 4452787 | gpu12 | PENDING / Dependency |
| BU_f4 | 4389570 | 4452788 | 4452789 | gpu13 | PENDING / Dependency |
| BUL_f4 | 4389575 | 4452790 | 4452791 | gpu13 | PENDING / Dependency |

每次 `noshm` 调用还按冻结包原行为提交一个 `rcb_sum` CPU 汇总作业；17 个汇总作业均已释放，快照时均在 `PENDING / Dependency`，等待各自训练作业结束。

## 核验留痕

- 冻结包 `MANIFEST.sha256`：`1a112a398b93fa9eb56c98bb7f81028e9b471c96e50450bdeeb3be01938c3e3b`；本机取得的 `submit.py`、`release.py`、`launch_run.py`、`p3.py` 与训练 Slurm 源码 SHA 均匹配清单。
- 远端重排前核验：UTC `2026-11-24T02:59:44.784922Z`；17 个原始 marker、JobID、节点、attempt=0 一致，无本包 live job 或先前 NOSHM 重排记录；可用空间 441.18 GiB，门槛 30 GiB。
- 本机提交操作：UTC `2026-11-24T03:22:48.742543Z` 至 `03:27:39.031819Z`。第一次 B0p_f0 held 任务因本机标签格式检查错误而停止在释放之前；随后单独读取并验证原始 held 记录，释放同一 JobID `4452757`，未重复提交。该停止审计保留。
- 最新远端状态快照：UTC `2026-11-24T03:23:01.605907Z`；17 个训练作业全为 PENDING、0 RUNNING；17 个汇总作业全为 PENDING；每个 run 仅见旧 NOSHM marker，未见新 STARTED/NOSHM。服务器目录空间约 441.18 GiB。
- 本机状态核验开始/收到：UTC `2026-11-24T03:34:24.154345Z` / `03:34:48.928328Z`。与远端 UTC 分别记录，不据此声称两台机器时钟已同步。
- 远端状态只读回执：[即时核验](operator/noshm_retry_verify_20261124T033424Z/RESULT.json)；提交/释放审计：[全部 17 折](operator/noshm_retry_submit_20261124T032248Z/FINAL_RESULT.json)；首折 held 恢复：[恢复回执](operator/noshm_retry_recover_20261124T031957Z/RELEASED_01_B0p_f0.json)；原始失败原因与逐折历史：[11 月 20 日失败时间线](../20261120_ecnu_account_review/FAILURE_TIMELINE.md)。

全过程未读取影像体素、标签、checkpoint 或预测文件，未更改远端冻结包/Slurm 源码、未管理其他用户作业、未删除旧失败记录。训练结果、内存盘通过、模型质量与科学门均未验收。
