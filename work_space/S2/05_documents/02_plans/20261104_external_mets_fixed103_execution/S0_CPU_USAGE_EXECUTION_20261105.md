# S0 实时 CPU 使用验收接续

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
