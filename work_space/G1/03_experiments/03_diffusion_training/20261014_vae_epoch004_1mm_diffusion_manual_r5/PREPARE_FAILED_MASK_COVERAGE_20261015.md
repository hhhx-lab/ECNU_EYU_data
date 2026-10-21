# R5预处理真实共同mask覆盖门失败：训练未启动

## 输入与唯一快照

按本线程原失败停止规则，仅调用一次既有operator/snapshot.py即时只读快照。先读总纲README、G1专题和替换提交记录；凭据仅使用约定env，显式启动分流VPN，操作结束关闭本次VPN。没有第二次远端快照、sleep、轮询、部署、预检写入、提交、release或新attempt。

- [完整原始快照](operator/snapshot_20261015T074133Z/SNAPSHOT.json)。本地文件SHA256：09fab7a6f2f966b6520904b98073053c4e14be84660c655fdce85e01c33dd5c4。
- 本地完成UTC：2026-10-15T07:41:37.975094+00:00（北京时间15:41:37）。
- 远端快照UTC：2026-10-15T07:30:32.530180+00:00；hostname=login1。两端时间分别记录。

## 已确认终态

| 阶段 | job | sacct状态 | 退出码 | 运行时间 |
|---|---|---|---|---|
| 新预处理 | 4086739 | FAILED | 1:0 | 00:05:46 |
| T1C训练 | 4069916 | CANCELLED | 0:0 | 00:00:00 |
| T1N训练 | 4069917 | CANCELLED | 0:0 | 00:00:00 |
| T2W训练 | 4069918 | CANCELLED | 0:0 | 00:00:00 |
| T2F训练 | 4069919 | CANCELLED | 0:0 | 00:00:00 |
| 旧CPU | 4069915 | CANCELLED by 2159 | 0:0 | 00:00:00 |

squeue为空；这些job的scontrol已返回Invalid job id，不能单凭消失判故障，但本次sacct、日志和失败marker共同确认预处理FAILED。四训练原有afterok依赖及KillOInInvalidDependent=Yes审计，与失败后的零运行CANCELLED相符；没有训练STARTED/RUN_CONTRACT/FAILED/checkpoint/progress/COMPLETE，不把CANCELLED 0:0当训练成功。旧CPU主动取消仍是已知历史，不当作新失败。

stdout显示START 4086739 UTC 2026-10-15T07:09:26Z、计算节点hostname=b142，顺序汇报病例00001至00005共5例处理结果。远端有15个病例目录，但并行worker可产生额外部分输出；不能称15例完整成功或全量926例通过。

dataset/STARTED.json存在，dataset/FAILED.json存在（SHA=d06806b2743c4e3f0257aed051c0372ec398e7e8f3b58ce6d276a18382bf5d78），dataset/COMPLETE.ok不存在。快照只记录失败JSON的存在/SHA与日志节选，未独立拉取失败JSON全文或全部病例manifest，不编造其内容/完整数据验收。

## 失败点与未核实原因

stderr明确：`ValueError: BraTS-MET-00006-000: authentic common mask does not cover segmentation`，位于runtime/prepare.py的process原生标签覆盖断言（line129），由ProcessPoolExecutor传播至主进程。

本地冻结代码核对：读取并校验真实T1N/T1C/T2F源SHA与原生几何/finite后，common_mask对每路按其0/100百分位、下限max(0,min)构造正值union并binary_fill_holes；随后检查任一label>0位于mask之外时立即报错。该断言在bbox裁剪及strict1mm重采样之前，因此不能将本次报错归因于已发生的1mm重采样、训练CUDA或优化器。仅已确认mask覆盖合同未满足，尚未测缺失体素数量/坐标/label分布、原图三模态取值、空间边界或纠正标签背景关系；不宣称单一科学根因。节选未见OOM/CUDA/nonfinite/磁盘满，不能据此保证全数据不存在这些问题。

不能删标签、改病例split、用label直接扩mask、任意膨胀mask或跳过失败病例来冒充原门通过。后续需新的明确只读诊断授权，先逐体素量化该病例的mask/label关系和来源合同，再决定独立修复批次；此轮不修冻结runtime、不继续提交。

## 身份、容量与停止动作

57文件逐SHA无漂移，packageSHA仍13f188b2609ba57544c1258aa368daeabe0201ecbdb81c3ec86700f6441b6ede。派生脚本SHA/active binding检查无错误，源CSV仍0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b。父epoch004重建3941042仍COMPLETED0:0、1030/4120/1030、split823/103/104；不把父成功当新1mm通过。

账号可用223836372992bytes（约208.47GiB），共享/public约26TiB。冻结198.85GiB容量预算不因此自动放行新attempt；本次失败不是容量门报错。部分数据和原失败证据原地保留，无删除或覆盖；仅本地轻量快照，不下载NIfTI/checkpoint或上传Drive。

按用户新失败停止规则，本分支停止，现有heartbeat g1-epoch004-1mm-diffusion已通过automation工具更新为PAUSED并写入此终态；不自动建立新attempt，不重复原submit/替换/release。此前“替换成功、入队”属于提交当时事实，现已新增运行失败，不继续沿用PENDING为当前状态。

共同源mask/严格1mm/finite/合法标签/消费SHA仍强制；用户P0/P1人工/P2等待豁免仅waived_by_user，不豁免本次标签覆盖合同，不伪造科学或工程pass。训练823/val103/test104锁定、四独立单A100各200k和原checkpoint/优化器/恢复预算不改。artifact_status=experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed，仅exploratory。总纲同步以Conda base catalog build/check通过为准。
