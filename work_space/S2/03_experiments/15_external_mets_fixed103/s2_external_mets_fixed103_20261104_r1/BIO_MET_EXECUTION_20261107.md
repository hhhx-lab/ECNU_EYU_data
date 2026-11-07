# Bio-Intel统一MET评分执行记录

## 当前状态

**后续启动验收已确认Python STARTED同JobID/合同SHA，4321579仍RUNNING/node214。** 最新squeue4:25、sacct allocation/batch各00:03:32/0:0；无FAILED/PROGRESS_Bio/COMPLETE，尚在输入校验阶段、没有Bio评分日志或新分数，四路表未完成。见[启动验收快照](operator/snapshot_bio_comparison_20261107T080716Z/RESULT.json)。下段为首次早期快照，不再代表当前缺STARTED。

新CPU评分作业 **4321579** 已一次held提交、独立核验、同ID首次release0。首次启动快照为 **RUNNING/node214**，squeue运行1:25、sacct allocation/batch各00:00:32，均0:0。两种计时按原样记录、不强行对齐。快照时Python STARTED/FAILED/COMPLETE均未出现、两个Slurm日志为空，尚未证明Python评分入口已到，也没有Bio新分数或完成四路表。

远端attempt：`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_external_mets_fixed103_20261104_r1/attempts/compare_bio_four_models_r1`。

- 合同SHA：`7b71f69302b690f086cbaa760ec8c5ab357487e170b367bd561eeec5cbc87b84`。
- 8文件运行包manifest SHA：`3d3712ef6bbe7a25ae89c482ec96ac5792124ef0e91f58e9fe59a09ce9b5e534`。
- Git基线：`7e06002c96b6d3530ff46be7bbf365129ccc290d`；本次新增源码精确字节另由冻结包绑定，不能声称新增代码已包含在该历史提交内。
- 资源：cpu_192G、8CPU、64GiB、24h、无GPU、无requeue。当前分区UP，适用QoS核验通过；本账号队列CPU加本次请求保守上界100，低于1000限制，没有改动其他作业。
- 当次预算门要求至少2GiB，提交时实际551490158592bytes；启动快照551520567296bytes，约513.644GiB。quota返回0但无数值，不当账号额度验收。

## 输入与实际动作

使用4319113完整Bio A5+B2推理/原作者后处理的103预测，manifest SHA `e63602a5e0d03ec1118a704a7bb247c2565ffef3baae1c2537986ce5f47120ad`。STARTED/COMPLETE/运行合同/预测清单与冻结包均验收；父4319117完整三路结果和103 GT参考压缩字节SHA通过。两父作业allocation/batch均COMPLETED0:0。

原run_official_met_eval.sh、compare_fixed103.py辅助算法逐字消费父冻结包。服务器既有brats_eval实际导入、Python3.10.18、BraTS-evaluation0.0.8、panoptica2.1.0、NumPy1.26.4、二进制及评价PY/YAML SHA通过；两次提交/释放前validate-only只读身份验收，没有在登录节点执行评分或打开GT体素。CPU作业负责GT/Bio native几何、finite和0–4标签检查，再评分103例。

最终同一作业生成Bio逐例72项、四路288均值/432配对及完整表。原三路216均值/216配对及CI原样保留，追加Bio与三路的216条共同finite配对；20000次患者组bootstrap、seed20260831。旧评分和CI不重算，不覆盖历史或更改NA/RC分母。

## 验证与证据

本机既有Conda brats2023_seg的318项批次测试：316通过、2个历史版本限定skip；其中新增9项无skip。提交前111源码AST、冻结脚本bash -n/3.10 AST、真实marker格式/漂移、RC/全NA、原216行保持、一次held/release及重复提交不写历史门通过。本机base无NumPy，只用于访问/结构校验；测试使用既有MRI环境，没有安装依赖。早期本机草稿的bootstrap键/manifest路径与实际schema不符、fixture通过文本模式复制CSV改变CRLF的问题均在SSH前修正，不是远端作业失败。

- [授权](BIO_MET_AUTHORIZATION_20261107.json)、[协议](BIO_MET_PROTOCOL_20261107.md)。
- [冻结payload](operator/submit_bio_comparison_20261107T080223Z/SUBMISSION_PAYLOAD.json)、[held](operator/submit_bio_comparison_20261107T080223Z/HOLD_RESULT.json)、[同ID释放](operator/submit_bio_comparison_20261107T080223Z/RELEASE_RESULT.json)。
- [本地独立验收](operator/submit_bio_comparison_20261107T080223Z/INDEPENDENT_LOCAL_ACCEPTANCE.json)、[会话回执](operator/submit_bio_comparison_20261107T080223Z/LOCAL_OPERATION_RECEIPT.json)。
- [本机测试](operator/verify_transport_20261107T080206Z/VERIFICATION.json)、[早期启动快照](operator/snapshot_bio_comparison_20261107T080416Z/RESULT.json)。

远端release UTC07:52:20.178414；本机提交会话结束UTC08:03:49.779774。快照远端UTC07:52:52.571427–07:52:54.754193，本机接收08:04:23.891472、finally08:04:23.892387。两端钟独立记录，不声称同步。两个会话均复用用户VPN、vpn_owned=false，未注销/关闭用户会话。

启动验收快照远端UTC07:55:52.235983–07:55:54.342988，本机接收08:07:23.469711。Python STARTED SHA为`29bbe0c073ef0da054cffd2c124ced34659223ad4d0121c6efa62790b31501e5`，同4321579/合同；最新余551463944192bytes，quota仍空。增加查询器后的完整复核仍318项316pass/2历史skip，112源码AST，见[最新本机复核](operator/verify_transport_20261107T080656Z/VERIFICATION.json)。本轮两个有界快照分别验释放后allocation与Python启动，不循环SSH/恢复后台监控。

## 后续手动验收

已实现`operator/snapshot_bio_comparison.py`，一次只读快照；只有sacct allocation/batch COMPLETED0:0、同JobID COMPLETE、103例/72列、288/432行和SHA通过，才拉回本次轻量成绩/JSON/报告，不重复下载旧结果或下载NIfTI/MRI/模型。沿用原env和base访问，未恢复监控或创建automation。失败保留并停止，不自动retry。

```bash
set -a
source '/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env'
source 'work_space/S2/03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/.env'
set +a
export ECNU_SERVER_ENV_FILE='/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env'
export ECNU_ATRUST_BROWSER=bridge
conda run --no-capture-output -n base python -B work_space/S2/03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/operator/snapshot_bio_comparison.py
```

fixed103仍为内部验证103扫描/69患者组，非锁定test104。公开训练暴露、开发选模、RC小样本和waived_by_user_not_pass保持；不称正式优越性或独立泛化。状态experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed。没有新推理/训练、安装、模型/MRI/NIfTI下载、删旧数据或管理其他作业。
