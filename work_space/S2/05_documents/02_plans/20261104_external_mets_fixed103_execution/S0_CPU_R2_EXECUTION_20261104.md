# S0 独立 CPU 准备 R2 执行记录

原 PLAN/PLAN.json 和全部失败记录保持不变。本记录消费用户已确认的独立调度验收、标准 backend 解析修正及一次 CPU 准备 R2 授权，不重新求解、改版本/模型/资源或覆盖失败。

## 已完成阶段

- 15 项模型/容器资产先前已在 ECNU 逐 SHA 验收，Mac 本批中转已清理；资产存在不等于推理可运行。
- R3 官方工具链为 47 包，solver SHA `68099499cdaee295940740b6144d1f9c0b73e9822338f310a05b8a44386d481c`，锁 SHA `866d02c7106d809d4f14b87c0da101e2c3c722ab50f7a8f78e3ea89fb5fc2d77`。后续 QoS 字段失败 SHA `7ab5ed495a16d03e697425e7c217eddbad2461f9b39c5716f92b925bad403894` 保留。
- 独立调度验收按服务器实际默认表头核本账号和 `cpu_192G`，COMPLETE SHA `d3057846d363c4850bffdb8c3c25ed9e6b9af5529f783740917f6319e123c014`，没有再次 solver。
- CPU staging R1 在安装前因 `connected-components-3d` 缺省可选 backend 而报 `KeyError`，没有创建构建环境/作业。失败 SHA `be077b174c4975aa0ee7daa26dfd562816d726f0c71be433f19fcf0c54960ee6`；一次只读诊断确认 setuptools legacy 标准默认、有效 setup.py 和三份 sdist SHA。
- R2 仅补标准默认及隐含 `setuptools>=40.8.0` 要求。原 47 包仍满足要求；R1 源码按 SHA 固定，R2 单独生成 runtime，保护 R1 全目录身份，复用原锁和已下载 sdist。
- 既有本机 `brats2023_seg` 中 115 测试通过、55 源码 AST/SHA 记录完成，证据为 `operator/verify_transport_20261104T143358Z`。不是 ECNU 环境/模型/A100 通过。

## 本次首次 R2 执行

当前终态：`S0_R2_BUILD_ENV_PASS_PRE_SBATCH_AGGREGATE_QOS_USAGE_HOLD`。

1. 首次 `stage_s0_cpu_r2.py` 成功，审计 `operator/stage_s0_cpu_r2_20261104T143559Z`。新 attempt 为 `attempts/s0_cpu_prepare_r2`，构建环境 `envs/bio_build_glibc217_r2` 的实际47包安装/身份及三源码标准backend要求通过。构建合同 SHA `8a2246326f58d7351712785e2395c54fb81949ad7a4390f19e18e80980329980`，原锁 SHA 不变，没有重新 solver 或下载源码。
2. 首次 `submit_s0_cpu_r2.py` 在 CPU 包与鲜活调度预检停止，审计 `operator/submit_s0_cpu_r2_20261104T145258Z`。断言为 `aggregate QoS TRES requires current usage audit`；失败 SHA `df7ca183f829e0c896002896f1b12e54115c75902842f3d14991f8a2b52dac03`。未调用 sbatch，没有 JobID/held/release。Slurm 已冻结且 bash -n 通过，SHA `d879a1addf9756b3bfa8561b96e33f600deb87c1b14f9483e79217d8506620ff`。
3. 一次 `inspect_s0_cpu_r2_submission.py` 只读诊断，审计 `operator/inspect_s0_cpu_r2_submission_20261104T145541Z`，确认原提交 RESERVATION/SBATCH_RESULT/HELD_CREATED/HELD_AUDIT/RELEASE_RESULT/SUBMISSION_RECEIPT 均不存在；冻结 Slurm、R2 build/runtime、父 R1 和旧环境/失败身份未变。轻量结果内保存冻结脚本正文、资源/QoS 表与 SHA，无模型或影像下载。

资源快照中，账号另有4303995/4303996，均 PENDING，请求24+1 CPU；连本次8为33，低于 `user_zqchen` 的 `MaxTRESPU cpu=1000`。这是请求资源的保守合计，不是实测运行使用；不能据此称实际超额或没有GPU。原验收器只允许队列为空时消费聚合TRES，现需独立、分状态的实时资源使用验收，不能直接删除断言或用旧零队列快照放行。已向用户询问接续确认，尚待答复；不自动执行独立提交。原 stage/submit 不重复调用，后续获确认只复用环境与冻结Slurm，不再安装或求解。

staging 远端 UTC `14:40:59.929648` /本地收尾 `14:52:26.302507`；提交前失败 `14:41:58.990705` / `14:53:25.290902`；诊断 `14:44:19.441865` / `14:55:45.741141`，日期均2026-11-04，约11分钟钟差保留、不称同步。诊断文件系统余 `914544918528bytes=851.736328125GiB`；quota -s返回0但空，数值账号额度/全量峰值未验收。三个调用均复用用户VPN（vpn_owned=false）且未注销；仍按原env精确 `ECNU_SERVER_ENV_FILE`、显式aTrust start/check/finally归属，不复制凭据。

最新本机 `operator/verify_transport_20261104T145530Z` 为118测试/57源码AST通过；base catalog维护9测试通过，仅本机准备。新 `snapshot_s0_cpu_r2.py` 供以后单次作业观察/轻量日志/产物身份核验，本轮未调用，没有循环、安装或作业动作。

| 运行准备 | 当前实际状态 |
|---|---|
| 15模型/容器资产 | 已验收并保留ECNU |
| 独立Conda构建环境R2 | 安装/47包/标准源码要求通过，不是完整推理环境 |
| CPU Slurm | 已冻结/bash -n/SHA通过；提交前使用审计hold，未sbatch |
| 三个同版wheel与2025 SIF | 未构建，旧转换超时保留 |
| Bio-Intel完整推理R4/trainer/import/pip check | 未完成 |
| J01-J04与两真实完整smoke | 完整包/Slurm仍未完成，无JobID |
| J05-J07全103推理/统一MET | 两smoke/联合容量估时门之后，未提交，无新分数 |

## 后续放行顺序

1. R2 staging pass 后冻结 `prepare.slurm`，`bash -n`、鲜活资源/容量和 SHA 验收；一次 held 提交，核验后释放同一 ID。CPU 为 `cpu_192G`、8 CPU、64 GiB、4 小时，无 GPU。
2. 获配 CPU 内离线构建三个同版 wheel，对同镜像执行一次总限 7200 秒转换。须 `COMPLETED 0:0` 且 COMPLETE/wheel/SIF SHA 完整。
3. 消费 wheel 至独立 Bio-Intel 推理 R4，完成 trainer/import/pip check 和隔离门；J01-J04 完整运行包/Slurm 仍需完成。
4. 两路真实完整 smoke 和联合容量/估时门通过后，才创建 J05-J07 全量 103 推理/统一 MET。S0 入队或完成不等于模型运行通过。

旧 SIF 超时、R1-R3 环境与全部失败保留。数值账号 quota 和全量峰值门未通过；不读 test、不恢复监控、不自动重试、不清旧数据。fixed103 真实四模态、K1 selected175/native/MET 与公开训练暴露边界不变，研究状态仍为 `experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

`operator/` 证据相对于[执行批次](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/README.md)；[STATUS](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/STATUS_20261104.md)及实际回执为事实源，两端 UTC 分记。
