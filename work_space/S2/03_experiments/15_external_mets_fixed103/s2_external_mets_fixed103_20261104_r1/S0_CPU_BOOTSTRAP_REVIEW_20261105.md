# S0 CPU 启动与执行链复核

本地复核 UTC：2026-11-05 04:18:27；远端观察 UTC：04:05:01。两端时钟分记，不称同步。

## 输入与范围

用户要求先认真检查代码再提交。本轮仅补本机执行测试和一次有界远端启动检查；没有新建 R3、sbatch、release、安装、solver、真实镜像转换、模型加载或影像读取。复用用户已连接的 VPN，显式 start/check，finally 保留非本次创建的会话。

事实源为失败的 R2 冻结 Slurm/合同/提交回执、独立构建环境、真实系统 profile 和容器 CLI。R2 Slurm SHA 为 `d879a1addf9756b3bfa8561b96e33f600deb87c1b14f9483e79217d8506620ff`，合同 SHA 为 `8a2246326f58d7351712785e2395c54fb81949ad7a4390f19e18e80980329980`，本轮前后均未变。

## 实际结果

| 检查 | 结果与边界 |
|---|---|
| 旧作业终态 | sacct 再确认 4304740 allocation/batch FAILED 1:0，21 秒 |
| 原始干净 shell | `/etc/profile: line 47: HISTCONTROL: unbound variable`，比真实 Slurm 的 COLORTERM 更早失败，同类可选变量问题 |
| 修正候选启动 | 初始化期间 `set -eo pipefail`，系统/Conda/module 完成后 `set -u`；真实远端退出 0，进入约定 Python |
| 严格错误退出 | 本机实际 shell 覆盖 profile、Conda、module 失败，初始化后 nounset/pipefail、空格路径及 bash -n |
| 构建环境 | 独立 R2 Python 3.10；numpy/pip/setuptools/wheel/Cython/tomli/packaging 七个真实 import 均成功；身份门保持 |
| 编译器 | 约定环境内 GCC/G++ 13.4.0，C 及含 vector 的 C++ 标准输入语法检查均 0，无警告；只 `-fsyntax-only`，未构建 wheel |
| 容器接口 | Singularity CE 4.1.1 build/sif new/sif add 帮助通过；4-Partition/2-PrimSys/1-Squash/2-amd64 与代码一致 |
| 压缩接口 | 远端 `/usr/sbin/mksquashfs -help` 返回 1 且提供完整帮助，符合该接口语义；双线程和 64/128/64 MiB 队列参数存在，非转换成功 |
| 完整准备入口控制流 | 本机模拟子进程覆盖三个 wheel、SIF、成功/失败 marker、SHA/父身份/GPU 漂移及拒绝重复执行；不替代真实构建 |
| 本机回归 | 162 tests / 65 源码 AST 通过；测试留本机，SHA/日志可索引 |

首轮新增测试因 macOS `/var` 临时路径与 resolve 后 `/private/var` 不一致失败，旧日志保留；仅规范夹具根路径，不放松生产路径门。首次远端汇总过度限定必须出现 COLORTERM，因原脚本更早报 HISTCONTROL 标记 requires_review；原 RESULT 不改，新增绑定其 SHA 的独立本机验收确认修正候选实际 bootstrap pass，没有再次 SSH。

## 证据

- `operator/review_s0_cpu_preparation.py`：有界检查；候选只改 nounset 顺序，实际准备 payload 替换成 probe。
- `operator/review_s0_cpu_preparation_20261105T041618Z/RESULT.json`：唯一远端原始观察，SHA `14afa3364227239f82222b3a757fb002248de04cb13ef9334a63a2ef8043cff0`。
- 同目录 `SOURCE_IDENTITY.json` / `LOCAL_OPERATION_RECEIPT.json`：实际源码身份、start/finally 时间和 VPN 归属。
- 同目录 `BOOTSTRAP_VERIFICATION.json`：独立本机验收，`pass_bootstrap_only=true`、`full_preparation_pass=false`、`GPU_pass=false`、`new_remote_execution=false`。
- `operator/verify_transport_20261105T041420Z`：首轮测试夹具失败；`041601Z`：夹具修正后 154 测试通过；`041825Z`：补分类拒绝测试后 162 测试通过。上述后两个时间均带前缀 `verify_transport_20261105T`。

只复核保存的观察、不联网的入口为 `review_s0_cpu_preparation.py --verify-saved <本批operator下观察目录>`。验收文件排他创建，已有验收时拒绝覆盖，不作为重复运行入口。

## 尚未验收

候选目前是已测试的初始化变换，不是已部署或提交的独立 R3 完整包。后续仍须绑定独立 attempt，冻结实际 runtime/Slurm/合同，核验鲜活资源、容量和无重复提交证据后单次 held 审计释放；不得改原 R2 或把 probe 当 CPU 构建通过。

三个同版 wheel、真实 2025 SIF、Bio-Intel 完整推理环境和两路 GPU smoke 尚未完成；J01-J07 没有新 JobID/结果。全 103 推理仍须两路真实 smoke 及联合容量/估时门。本次未刷新磁盘或 quota，835.58984375 GiB 只是前次观察，不是新余量、数值账号额度或全量容量许可。

保持 `experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。原计划、失败和环境不改，不恢复监控、不读 locked test。
