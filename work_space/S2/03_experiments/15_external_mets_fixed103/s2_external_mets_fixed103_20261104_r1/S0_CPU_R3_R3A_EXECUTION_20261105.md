# S0 CPU R3失败与授权R3a首次提交

本机复核UTC：2026-11-05 04:51:14。科学状态仍为 `experimental_unvalidated / operator_approved=false / formal_gate_status=not_run_not_passed`。

## 当前结果

**R3a的CPU准备作业4304783已held核验并释放，唯一即时只读快照为RUNNING/node201。不是GPU推理已启动，也不是准备产物已完成。**

| 项目 | 实际证据 |
|---|---|
| attempt | 远端批次下 `attempts/s0_cpu_prepare_r3a` |
| JobName / JobID | `s2exts0p3a / 4304783` |
| 资源 | `cpu_192G / 8 CPU / 64 GiB / 04:00:00 / 无GPU / 无依赖 / Requeue=0` |
| sbatch / release | 各一次，均返回0；held核验后本机独立验收，再查同ID后释放 |
| 释放时 | `PENDING / Reason=None / RunTime=00:00:00` |
| 即时快照 | squeue、sacct allocation/batch、scontrol均为RUNNING；scontrol为node201、55秒，sacct为2秒，顺序查询非原子快照 |
| Python marker | `STARTED/FAILED/COMPLETE`均不存在；只有输入staging和源码构建要求审计 |
| 日志 | stdout/stderr均0 bytes；不是构建完成或完整运行健康证明 |
| SHA | 冻结runtime/合同/Slurm、父R1/R2、旧环境与失败R3四文件集合无漂移 |

远端根：`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_external_mets_fixed103_20261104_r1`。

## 输入与修正范围

用户明确确认“允许独立 R3a 核验后继续提交”，原话及限制见[S0_CPU_R3A_AUTHORIZATION.json](S0_CPU_R3A_AUTHORIZATION.json)。复用原 `envs/bio_build_glibc217_r2`、47包精确锁、三个已验收sdist和固定2025归档；不再安装环境或求解，不改模型/版本/资源/推理算法。该CPU作业仅构建同版 `numexpr2.14.1 / connected-components-3d4.1.1 / ndindex1.10.1` wheel和单次固定镜像SIF，转换总上限7200秒。

旧CPU4304740仍为FAILED1:0/21秒，nounset早于系统profile导致Python入口前退出，原包不改。R3生成的Slurm在system/Conda/module初始化后才开启nounset；真实干净shell已验证。R3随后在freeze阶段引用错误的 `operator/SOURCE_BUILD_REQUIREMENTS.json`，实际文件位于 `reports/`。错误发生在sbatch前，R3无JobID。修正版现在先核父引用存在、轻量大小、SHA，再创建独立attempt；R3a只接续此修正和新attempt身份。

## 原失败永久保留

- [R3原失败回执](operator/submit_s0_cpu_r3_20261105T043351Z/PREPARE_HELD_RESULT.json)：`FileNotFoundError / freeze_R3_package / job_id=null`。远端UTC04:22:49.669747，本地finally04:34:16.438690。
- 原诊断因漏import os停止，[失败审计](operator/inspect_s0_cpu_r3_freeze_20261105T043706Z/)保留；修正并补fresh-namespace测试后，一次[独立只读诊断](operator/inspect_s0_cpu_r3_freeze_20261105T043836Z/RESULT.json)确认reports文件SHA `27867735b456b703672e55de5622422c8d906b3c52fb03e1d0d570c616445f85`，operator同名文件不存在。六提交marker全无，squeue空、sacct仅表头；不因不存在失败marker判成功。
- 诊断远端UTC04:27:13.621448，本地finally04:38:40.459658；当时文件系统余835.357421875GiB，仅为历史。
- R3部分目录恰四文件，R3a在冻结、提交/释放、准备入口前后及快照均核固定完整集合。`SUBMISSION_FAILED.json` SHA `48af2f4f423d66ff8c8a8d2158f52842028a05de1a0b6e08b8b9f3df50c77127`，其余三SHA见新contract的 `failed_R3_attempt_files`，不清旧失败或补写原R3成功。

## 本机验证与实际执行

1. 最终在已有 `brats2023_seg`执行[验证器](operator/verify_local_transport.py)，[201项tests/70源码AST及源码SHA通过](operator/verify_transport_20261105T044734Z/VERIFICATION.json)。覆盖完整准备入口控制流、真实shell错误退出、真实输出回放、最终路径/脚本、held到本机核验再同ID释放、错误JobID/漂移/重复提交拒绝、fresh-namespace快照和实际main三次传输顺序。测试模拟不替代真实wheel/SIF/GPU验收。
2. 新增R3a快照测试首次遗漏测试环境变量而本机失败，原[200项测试失败日志](operator/verify_transport_20261105T044638Z/UNITTEST.log)保留；仅修fixture环境后200项通过，再补main传输记录覆盖得到201项通过，没有因此调用远端或创建新attempt。此前R3各轮本机失败/通过日志也保留。
3. [操作器](operator/submit_s0_cpu_r3a.py)只执行一次。显式加载原服务器env及精确 `ECNU_SERVER_ENV_FILE`；现有session调用VPN start/check，finally只关闭本次自建会话。本次 `vpn_owned=false`，用户会话保留，无logout/stop。
4. 真实远端先复核4304740的FAILED终态、原R3完整失败库存、R2精确环境/锁/sdist/归档；冻结新runtime/合同/完整Slurm并 `bash -n`。按新冻结脚本截取的真实干净shell bootstrap完成，七import、C/C++语法及容器CLI通过，未构建产物、读MRI或加载模型。
5. 当前分区/QoS/账号/节点及全部本账号作业的保守CPU资源上界验收通过。已有17作业，运行上界120、排队上界145、本次8，合计273<1000；不是实测GPU占用，不修改无关任务。
6. 单次sbatch --hold，保存JobID；远端held身份核验、本机JSON往返bytes/SHA和最终rendered Slurm核验通过，再远端核合同/脚本/容量/同ID并单次release。最后只一次即时只读快照，无循环轮询/持续SSH或监控恢复。
7. 另在本机复核十份轻量JSON、三份实际远端源码记录、snapshot报告JSON、同JobID和原R3库存，见[本机回执复核](operator/submit_s0_cpu_r3a_20261105T044747Z/LOCAL_RECEIPT_REVIEW.json)。只做本地证据核验，无第二次SSH。

执行环境配置沿用原env；入口为 `conda run --no-capture-output -n brats2023_seg python -B <本批次>/operator/submit_s0_cpu_r3a.py`。本次已执行成功，**禁止再调用此提交器**；后续只读查询不得重复sbatch/release或自动新attempt。

## 身份、时间与容量

- 合同SHA：`6615846dca72b6b0cd2dd861cfa8e656ae9e2c9f51ffbb952f566fc62481063c`。
- Slurm SHA：`cc76d642f087be1b5b83d3b7208ffb152ca256e303071017f7f7ee66438795d5`；准确全文在[held回执](operator/submit_s0_cpu_r3a_20261105T044747Z/PREPARE_HELD_RESULT.json)的 `frozen_slurm`，不是未执行模板。
- 准备入口SHA：`771c3bec7e959c91d9ff772ed635decb53a4d3eac579d4e5d1ee9bdc65b4b88b`；三个runtime SHA见contract `package_files`。原47包锁仍 `866d02c7106d809d4f14b87c0da101e2c3c722ab50f7a8f78e3ea89fb5fc2d77`。
- held远端UTC04:36:55.764448；本机held验收04:48:22.725902；release远端04:36:59.534791；即时快照04:37:03.195737；本地finally04:48:29.979731，全部2026-11-05。两端约11分差仍分别记，不冒称时钟同步；Slurm自身12:36:07保留原值。
- 最新路径所在文件系统余896712835072bytes=835.12890625GiB，高于仅本次CPU增量预留150GiB。quota返回0但无数值，不宣称账号扩容额度或全量推理峰值门已通过，不据旧余量放行其他数据。

## 尚未完成

CPU准备完成须 `sacct COMPLETED 0:0 + reports/COMPLETE.json`，三wheel和SIF完整bytes/SHA/声明身份一致且无FAILED。当前没有这些成功证据，也尚未看到Python STARTED。后续仍要消费三wheel建立完整Bio-Intel推理环境、严格加载七fold/BrainIAC、实测2025容器隔离与GPU映射、515输入/103 K1预测/native与MET身份、完整Slurm和两路真实GPU smoke；联合门通过后才提交J05-J07全量。J01-J07目前未创建，无新分数，不读locked test、不回传MRI/模型、不清旧数据、不恢复监控。原PLAN和公开训练暴露的描述性横评边界不变。
