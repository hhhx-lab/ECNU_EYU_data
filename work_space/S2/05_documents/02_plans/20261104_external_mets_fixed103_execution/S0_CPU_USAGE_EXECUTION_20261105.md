# S0 实时 CPU 使用验收接续

## 15:25 全量请求因本机分流漂移尚未SSH

用户要求提交两外部成品同fixed103全量；先新增只读snapshot_s0_cpu_r3a入口绑定4304783，201tests/71AST通过后调用一次，UTC05:35:32.305492在VPN归属门停止/finally05:35:32.305803，无本轮SSH/远端UTC/容量/新JobID/安装。只读本机status显示aTrust online/tunnelAvailable，但默认网关不同于旧基线、三个公网探针重新走隧道；不改基线/降门/注销用户会话，私有网络明细只留原本机状态。

用户选择“我重新连接 aTrust 后继续”，等待完成确认；之后UTC07:25:36一次本机只读status仍同登录会话/default changed。补只读入口完整执行测试，202tests/71AST通过，非远端或GPU通过。4304783当前未知，旧RUNNING和835.13GiB不得当本轮验收；CPU完成、三wheel/SIF身份、完整环境及两真实smoke/联合门前，不创建两全量。无重装/solver/自动retry/监控恢复或影像/test读取。见[接入停止分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/FULL_INFERENCE_ACCESS_HOLD_20261105.json)，原PLAN不改，以下历史保留。

## 12:51 R3a单次提交释放与CPU运行

用户确认独立R3a核验后继续提交。201本机tests/70AST通过，覆盖准备完整入口/最终脚本/真实shell错误退出、完整held与main传输链。原R3引用operator路径错误、sbatch前无JobID，原诊断漏import os及本机fixture失败全部保留；正确reports SHA与原R3四文件集合严格绑定，未改原失败。复用R2环境/47包锁/模型/资源，不重装/solver/自动retry。

CPU4304783/cpu_192G/8CPU/64GiB/4h/无GPU首次sbatch0/held审计pass，本机验十轻量JSON及合同/冻结Slurm，再同IDrelease0；释放PENDING/None/零运行，唯一即时只读快照RUNNING/node201。Python STARTED/FAILED/COMPLETE均无、日志0bytes；非实际wheel/SIF或模型/GPU通过。SHA drift为空，用户VPN保留、无第二次SSH或监控恢复。合同6615846d...1063c，脚本cc76d642...795d5，原锁不变；CPU请求保守273<1000而非GPU使用实测。

held远端04:36:55.764448、release04:36:59.534791、snapshot04:37:03.195737，本机held验04:48:22.725902/finally04:48:29.979731/复核04:51:14UTC分别记录。最新余896712835072bytes=835.12890625GiB，仅本次CPU150GiB增量门通过，quota0无数值，非全量容量验收。CPU完成须COMPLETED0:0+COMPLETE+三wheel/SIF完整SHA；完整推理环境、两真实smoke/联合门、J01-J07仍未完成/创建，无新分数/读test/模型回传。见[完整R3/R3a记录](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_CPU_R3_R3A_EXECUTION_20261105.md)及[提交/快照回执](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/submit_s0_cpu_r3a_20261105T044747Z/)。以下均为历史，原PLAN与研究边界不变。

## 12:18 启动与执行链检查

用户要求认真检查后再提交。本轮补完整准备控制流模拟、真实 shell 错误退出和分类拒绝测试，162测试/65AST通过；一次有界远端检查修正初始化、七import、C/C++语法及容器CLI通过。原脚本在干净环境先报HISTCONTROL，与已失败COLORTERM同根因；原汇总仅认COLORTERM的限制通过独立本机验收解释，原RESULT不改、不二次SSH。R2/4304740失败和旧环境保持，用户VPN保留，无新R3/作业/安装/solver/真实转换/影像或容量观察，非wheel/SIF/模型/A100通过。见[完整复核](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_CPU_BOOTSTRAP_REVIEW_20261105.md)，以下为历史。

## 12:02 已提交释放但系统初始化失败

用户先确认恢复校园登录，本机UTC03:51:30调用仍在VPN归属门停止，未SSH；一次只读status和客户端标识仍为disconnect/未登录。用户选择手动客户端重连后再确认“连好了”，同一个尚未远端执行R2才首次进入SSH，没有自动第二次启动或新建远端R3。

CPU4304740首次sbatch0/held身份pass/同ID release0，资源cpu_192G/8CPU/64GiB/4h/无GPU。九份轻量JSON往返bytes/SHA及冻结Slurm/合同、owner/account/QoS/资源/日志/无依赖身份本机独立核验通过；释放时PENDING/None/零运行。随后唯一即时只读快照发现allocation/batch FAILED 1:0、21秒，stderr为`/etc/profile.d/256term.sh: line 9: COLORTERM: unbound variable`。Slurm先启用`set -euo pipefail`再读取系统profile，系统对可选变量的引用触发nounset；是我的初始化顺序错误，不是模型/GPU/solver或实际wheel/SIF转换失败。Python STARTED/FAILED/COMPLETE均没有，不能因未出FAILED marker视为健康。

冻结Slurm SHA d879a1ad...6620ff、合同8a224632...29980及旧两失败保持，快照drift为空；新回执SHA `592c93f24b50993446ff38210c5e0ef352a554beb7c0c3dea147b41ee4f71eda`。提交远端UTC03:47:09.683180/本地finally03:58:36.443474，快照03:49:32.882755/04:00:59.611979分别记录，不称同步；Slurm原始起止11:47:49/11:48:10保留原值。文件系统余897207762944bytes=835.58984375GiB，quota0无数值，非额度或全量峰值门。实时CPU保守上界120+145+8=273低于1000，仅请求/分配上界，不是GPU使用量。用户现有VPN复用不注销。

新增`snapshot_s0_cpu_usage_r2.py`只绑定本次R2回执路径，保持仅读日志/marker/静态身份和完成产物验收；138测试/64源码AST通过，非模型/A100 pass。已询问独立R3只改Slurm初始化期间nounset顺序、环境/资产/版本/资源不变，先核验再单次held；尚未授权或提交，不原位改R2或自动重提。J01-J07、真实smoke和全量仍未创建，无新分数/影像/test读取。

证据：[提交与失败分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_CPU_USAGE_R2_SUBMISSION_FAILURE_20261105.json)、[提交回执](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/submit_s0_cpu_usage_r2_20261105T035804Z/RESULT.json)、[即时快照](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/snapshot_s0_cpu_usage_r2_20261105T040052Z/RESULT.json)、[本机快照验收](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/verify_transport_20261105T040015Z/VERIFICATION.json)。以下均为历史。

## 11:15 已授权但接入未就绪

用户已确认“允许独立R2修正解析后继续首次提交（推荐）”，登记 `S0_CPU_USAGE_R2_AUTHORIZATION.json`。本机136测试/63源码AST通过，证据 `operator/verify_transport_20261105T031456Z`。随后本机调用 `submit_s0_cpu_usage_r2.py` 在VPN归属门停止，尚未进入SSH或执行远端R2，没有本次sbatch/held/release/JobID；本地停止UTC03:15:11.880249、finally03:15:11.880458。

一次只读status返回disconnect/tunnelAvailable，aTrust界面标注“未登录”；旧的“登录完成”卡片不是当前连接证据。没有调用stop/logout或修改VPN适配器，不绕过官方登录/在线验收门；没有新远端UTC、最新配额或存储观察，旧851.736328125GiB只保留历史标记。已请求用户按学校流程恢复浏览器登录，不发送密码；恢复后继续同一个尚未远端执行的R2，不建立新提交attempt，不自动重复start/SSH/sbatch。

见[接入分析](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_CPU_USAGE_R2_ACCESS_CONTINUATION_20261105.json)、[本机停止回执](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/submit_s0_cpu_usage_r2_20261105T031511Z/OPERATION_STOPPED.json)和[收尾回执](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/submit_s0_cpu_usage_r2_20261105T031511Z/LOCAL_OPERATION_RECEIPT.json)。两次远端失败、冻结Slurm/构建合同/环境与模型资源不改；CPU准备、真实smoke和全量仍未提交。

## 以下为前次解析修正历史

北京时间2026-11-05，远端/本地UTC日期仍为2026-11-04。原PLAN、冻结Slurm、构建环境、模型/版本/资源和全部失败保持不变。本次没有创建作业。

## 实际执行

用户批准的 `submit_s0_cpu_usage_r1.py` 只执行一次，在提交前分区比较停止，未调用sbatch，没有JobID/held/release。失败SHA `9fa1bd3fd29656328d8d35b6fd9ae48f375bf79a197e665714b28ac5d11c5f75`；原聚合使用门失败SHA `df7ca183f829e0c896002896f1b12e54115c75902842f3d14991f8a2b52dac03`也保留。

一次独立只读诊断对照默认/宽字段squeue、逐本账号作业scontrol和末次队列，确认队列不变。根因是原 `job_fields` 正则漏识别带冒号的复合字段：`AllocNode:Sid`被拼进Partition，`ReqB:S:C:T`被拼进CPUs/Task；旧式 `NtasksPerTRES:0` 也需要独立分隔。不是分区迁移、Slurm拒绝或模型/GPU失败，不能删除身份核验来放行。

| 已有作业 | 状态 | 真实分区 | NumCPUs / TRES cpu |
|---|---|---|---|
| 4303995 | PENDING | a100 | 24 / 24 |
| 4303996 | PENDING | cpu_96G | 1 / 1 |

真实输出本地回放得到保守请求上界25 CPU，加本次8为33，低于 `user_zqchen MaxTRESPU cpu=1000`；不是实测GPU使用或已分配运行量。两处提交审计RESERVATION/SBATCH_RESULT/HELD_CREATED/HELD_AUDIT/RELEASE_RESULT/SUBMISSION_RECEIPT均不存在。R2 runtime/build、父R1和旧环境/失败身份未变；脚本SHA `d879a1addf9756b3bfa8561b96e33f600deb87c1b14f9483e79217d8506620ff`，合同SHA `8a2246326f58d7351712785e2395c54fb81949ad7a4390f19e18e80980329980`。

失败远端UTC `15:45:37.560662` /本地finally `15:57:04.620463`；诊断远端 `15:47:18.064896` /本地finally `15:58:45.511692`。钟差分记、不称同步。最后文件系统余 `914544918528 bytes = 851.736328125 GiB`；quota返回0但无数值，非账号额度或全量峰值验收。两次均复用用户VPN，未注销。

## 修正版与后续门

独立 `s0_slurm_fields_r2.py` 支持复合key=value及观测旧式后缀，重复字段拒绝。`submit_s0_cpu_usage_r2.py` 只替换生成操作器的解析函数，不改冻结源码/Slurm/runtime；同一函数用于实时使用与held身份审计。执行前必须消费新的用户确认、两次失败SHA和两处尚未提交证据，之后仍仅一次held提交、核验再释放同ID。

136本机测试/63源码AST通过，含两份真实输出回放、复合字段held模拟和资源/分区真实漂移拒绝。此前130测试中的四项夹具错误为漏传job_fields，只修本机夹具后通过，失败日志保留。测试不证明远端预检、模型加载或A100通过。

R2首次执行已询问新确认，当前未授权或执行。不重复R1，不重新求解/安装，不自动retry。仍须CPU `COMPLETED 0:0` 与三个wheel/SIF身份、完整Bio-Intel推理环境消费、两路真实完整smoke及联合容量/估时门；未通过前不提交全103。J01-J07无JobID，无新分数、影像/test读取或监控恢复。

证据：[分析JSON](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/S0_CPU_USAGE_R1_FAILURE_ANALYSIS_20261105.json)、[失败回执](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/submit_s0_cpu_usage_r1_20261104T155631Z/RESULT.json)、[只读诊断](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/inspect_s0_cpu_usage_failure_r1_20261104T155833Z/RESULT.json)、[本机验收](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/verify_transport_20261104T160021Z/VERIFICATION.json)。

`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。
