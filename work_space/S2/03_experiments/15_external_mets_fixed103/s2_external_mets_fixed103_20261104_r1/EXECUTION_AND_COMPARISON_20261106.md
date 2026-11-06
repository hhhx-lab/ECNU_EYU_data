# Bio全量R3提交与2025/K1/原E同fixed103比较

## 完成查询与回传（本机UTC12:20）

现有`operator/snapshot_current_comparison.py`本轮仅一次即时只读SSH快照，Conda base且凭据由精确`ECNU_SERVER_ENV_FILE`原路径加载，复用用户VPN并保留；没有sbatch/release/安装或远端写入。Bio4319113仍PENDING/Priority、0/103预测，STARTED/RUN_CONTRACT/FAILED/COMPLETE均无，不将排队称运行。

评分4319117 allocation/batch均COMPLETED0:0，node203，01:13:43。三路各103例/72项、K1-175和原比赛E-200历史逐例schema/NA/数值重放（max abs diff=0）、患者组paired bootstrap20000/seed20260831均完成，无FAILED/静态漂移。仅下载完成manifest选择的12份轻量结果248452 bytes；服务器和本机大小/SHA验收通过，不下载MRI/模型或追加未验收表。

本地标准库验收命令使用`conda run --no-capture-output -n base python -B -c`，解析完成清单/快照/CSV，交叉绑定12文件SHA和同JobID合同、三路相同103病例/72列，复算216条finite均值及216条共同finite配对分母/均值/raw delta。全部通过，不在本机重算bootstrap CI，不重读影像/GT，不重跑此前训练/模型测试。

成果见[下载结果中文README](results/comparison_light_20261106T121856Z/README.md)和[本轮完成复核](COMPARISON_COMPLETION_REVIEW_20261106_1223UTC.json)。WT病灶Dice2025/K1/原E为0.703347/0.630263/0.617613（各n87）；其余全部72项及配对CI保留原精度，n变化与NA不补0。点估计领先只作同fixed103公开成品描述，不升级为独立泛化或官方排名。

完成remoteUTC2026-11-06T11:41:36.142287；快照remoteUTC12:07:38.040352；本机接收12:20:36.451110/finally12:20:36.452238/复核12:23:51.667888分记。余588020449280bytes=547.63671875GiB，quota0空非数值账号配额。Bio尚不在三路完成表且未有CPU评分JobID，待103预测及完成身份验收后另建独立评分/四路汇总，复用三路，不重评/覆盖。原NSD/dtype告警、全部失败、科学三项及豁免门保持；下文评分中/待完成段落为历史。

## 本次输入与授权

用户要求机时恢复后提交Bio-Intel完整模型，并回传2025预测与K1、原E在同一队列比较。这里是103扫描/69患者组的fixed103验证集，不是锁定test104、master test130或官方179。用户本次允许2025预测NIfTI回Mac；不下载MRI、外部权重或镜像，不恢复监控。

## 已完成

- Bio唯一新作业 **4319113**，独立`full103_bio_r3`，单A100/8CPU/64GiB/48h。R1/R2失败、2025完成作业及模型/环境不改；原七运行源码和两manifest字节一致，只更新attempt/授权/合同身份。当前数组展开逐任务计数，避免用折叠array ID查询资源。已held审计、本机独立验收和同ID首次release0，释放时PENDING/None，后续PENDING/Priority，不能称RUNNING或完成。Slurm本次接受提交，不等于取得CPU机时余额数值。
- R3包SHA `8d55cf768ada8310d59d2d621bed21c9db64d66a5a668f467cab8361240c3e1c`，合同SHA `a51e9d1a475d01e3f89a402279462b1cbdfbaddaf9fae3eca0f84e81be5b5eb4`。实际远端释放UTC `2026-11-06T10:12:24.693582+00:00`，本地finally `10:23:53.008749`分别记录，用户VPN保留。
- 2025原作业4310575再次确认allocation/batch COMPLETED0:0；103当前预测均与完成manifest大小/SHA一致，总4260855 bytes。回传至[本地预测批次](results/2025_full103_download_20261106T102424Z/README.md)，下载验收通过，不称分割质量通过。
- K1的103现存预测、固定CSV/参考manifest、历史72指标CSV及既有brats_eval环境Python3.10.18/BraTS-evaluation0.0.8/panoptica2.1.0/NumPy1.26.4、三入口二进制SHA均通过只读当前核验；实际两评价模块全部py/yaml源码SHA作为新评分合同输入。
- **原E预测确实在本地存在，前次仅做特定远端/普通文件搜索不能解释为全局缺失。**本轮按实际本地文件发现`selection_evidence/E/prediction`全部103份。`preparation_summary.json`绑定原比赛E epoch200 checkpoint SHA `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`；不是E-SNFH175或FixV3。103预测合法标签/native几何、E历史参考与当前本地参考逐体素/网格相同，当前本地参考每份SHA均匹配固定参考manifest，全部通过。没有加载checkpoint或读MRI。

证据：[R3回执](operator/submit_bio_full103_r3_20261106T102000Z/RELEASE_RESULT.json)、[当前输入与环境验收](operator/collect_comparison_inputs_r2_20261106T102424Z/RESULT.json)、[2025回传验收](results/2025_full103_download_20261106T102424Z/DOWNLOAD_ACCEPTANCE.json)、[原E本地身份](operator/audit_local_comparison_20261106T102134Z/RESULT.json)。

## 评分执行与暂未完成项

### 19:52进度查询

2025与K1已各自完成103例/72列评分，CPU4319117在node203运行1:13:15，已进入原比赛E评价。K1评价日志记录CSV SHA与历史相同；原E进度、全量历史重放、总COMPLETE均尚无，不以部分结果判断谁更好，不回传未验收的最终表。Bio4319113仍PENDING/Priority，0预测、未开始。无FAILED或静态漂移，现有NSD默认阈值/dtype告警保留。本次仅现有操作器一次即时只读快照，未改源码/模型/环境/资源，不重跑测试/提交/释放/安装/影像GT读取或恢复监控；用户VPN保留。

remoteUTC11:41:09.264230/local finally11:52:37.668038/本地复核2026-11-06T11:53:52+00:00分记，余588107481088bytes=547.7177734375GiB，quota0空。见[本次复核](PROGRESS_REVIEW_20261106_1152UTC.json)与[原始快照](operator/snapshot_current_comparison_20261106T115223Z/RESULT.json)，后续段落中的K1未完成是此前快照。

### 本次进度查询

2026-11-06本次即时只读检查：Bio-Intel4319113仍PENDING/Priority、0预测、无STARTED/FAILED/COMPLETE，未运行。三路评分4319117 allocation/batch RUNNING/node203，squeue elapsed47:28（sacct46:35，保留实际观测差异）。EXT2025的PROGRESS已记录103例且日志确认完整72列CSV产生；K1已进入官方评价，PROGRESS尚无；原E、历史重放与总COMPLETE尚无。无包漂移或FAILED，未回传半成品评分CSV，不据单列日志值判优劣。K1日志保留原Panoptica NSD默认0.5及非整数dtype告警，不等于作业FAILED，也不声称无告警。remote UTC2026-11-06T11:15:22.205392+00:00、本地finally11:26:50.706159/复核2026-11-06T11:27:59+00:00分记；余588223873024bytes=547.826171875GiB，quota0空非数值账号额度。只使用现有只读入口，无新提交/释放/安装/影像或GT读取，用户VPN保留，不恢复监控。

本次仅读现有两job状态、marker、日志尾及静态包身份；2025的103份预测已在上次完整验收回Mac，本轮未下载MRI/模型或未完成的评分。EXT2025单路进度不等于三路历史复现/paired/bootstrap完成，也不等于Bio完成。源码未修改、依赖未安装，本次没有重跑模型或评分。见[进度复核](PROGRESS_REVIEW_20261106_1126UTC.json)和[原始只读快照](operator/snapshot_current_comparison_20261106T112642Z/RESULT.json)。最终评分完成后仍须allocation/batch COMPLETED0:0、COMPLETE、历史逐例72项复现及回传SHA。

**收尾快照：4319117在node203运行5分17秒，评分Python STARTED已出现并绑定同JobID/合同；无FAILED/COMPLETE或包漂移，三路PROGRESS尚无。** Bio4319113仍PENDING/Priority、0预测。快照remote UTC10:33:10.347387、本机复核UTC10:45:28分记；最新文件系统余613466243072bytes=571.3349609375GiB，quota0空。全部评分结果仍等待本作业完成，不生成虚构的2025对比数值。见[最终执行复核](EXECUTION_REVIEW_20261106.json)和[收尾快照](operator/snapshot_current_comparison_20261106T104425Z/RESULT.json)。下文STARTED尚无是10:29初始快照的历史。

最终104356完整309发现/307pass/2原版本skip，108生产源码Python3.10 AST与base三控制入口imports通过，含新增本地独立提交验收入口。索引第一次build与测试日志/VERIFICATION写入相撞，报告仅这两项变动，源码/curated链接无漂移；文件写完后再build/check。原测试、提交和索引失败均留证，不以工程验收升级科学结果。

**最新独立R2：评分4319117已首次held提交、独立核验并同IDrelease0，allocation/batch RUNNING/node203。** 8CPU/64GiB/24h/cpu_192G，无GPU。包SHA `dc54e00a8fb9b0f7bebb3e16d9dd85b9380076cbc69d3eb53f89b3c721e455e2`，合同SHA `ae679cbb723ed9bccbdfcda172056d7f0ab17371db2ffc67f3685d1bef058ed5`。相对R1只改合同/Slurm/包三个身份成员，原评分runtime、官方脚本、三路预测、GT、历史CSV及环境不变，原E沿用已上传文件。远端release UTC10:28:45.965867，本机finally10:40:14.375651、独立本地验收10:41:00.380186分别记录。

原R1问题已查明：分区`QoS=partition_cpu_192G`与QoS表`partition_cpu_192g`大小写不同，操作器误拒，不是资源不足证据。新操作器沿用旧S0的唯一canonical匹配，保持旧helper原SHA；核R1失败SHA、全部冻结成员、无SUBMISSION_RESERVED/HELD/RELEASE标记、当前queue/accounting无评分作业后首次提交。原R1永久保留，不自动重跑或覆盖。

即时快照remote UTC10:29:10.248115，STARTED/FAILED/COMPLETE尚无，**尚无2025新分数或完成比较表**。Bio4319113仍PENDING/Priority、0预测，未再次提交/释放；服务器文件系统余613506088960bytes，quota0空非数值账号配额。见[CPU回执](operator/submit_fixed103_comparison_r2_20261106T103950Z/RELEASE_RESULT.json)、[独立本地验收](operator/submit_fixed103_comparison_r2_20261106T103950Z/INDEPENDENT_LOCAL_ACCEPTANCE.json)、[即时快照](operator/snapshot_current_comparison_20261106T104033Z/RESULT.json)。下文无CPU JobID和诊断中属于此前阶段。

准备独立CPU8/64GiB/24h评分包，复用官方评价脚本，不在登录节点评分，不新建依赖。三路顺序完整103，ET3/RC4/TC1∪3/WT1∪2∪3、物理27mm³、parser DSC0.2、matcher1e-6；保留完整72列及NA语义。K1与原E均必须与历史逐例72指标/schema/NA复现，atol1e-12/rtol1e-9，不通过则比较hold。预定患者组paired bootstrap20000/seed20260831、共同finite病例/组框架与患者等权敏感性，不混G1影像SSIM。

本次评分提交前`fresh_cpu_resources`检查停止，**尚未sbatch，无评分JobID或新分数**；原冻结`compare_2025_K1_E_r1`及本地/远端失败回执保留，正在独立只读核实际CPU分区/QoS。不会将Bio入队、2025回传或原历史数值称为2025新成绩。两次无作业的本方错误亦保留：首次回传验收把模型专属reports路径漏写，修正后独立只读验收/回传成功；评分操作器顶层导入NumPy在base本机失败，移入仅本地数据审计main后base入口导入通过，无安装、当次未接VPN/SSH。评分资源门问题须另据原输出解释。

本机103927完整309发现/307pass/2原版本skip，107生产源码3.10 AST及base入口import通过，含真实QoS输出、完整CPU单次held/release控制流、父失败无修改/有提交证据拒绝、snapshot动态attempt及accounting滞后。较此前317净少8：移除BioR2重复类发现13项，新增5项回归，非删除测试。103828捕获旧helper冻结SHA漂移，已还原仅本次改动；103856有旧测试对子进程全部参数误搜pip/wheel的随机临时路径误报，再跑103927通过，两日志保留。随后104356及纯本地验收入口的最终108源码验证已通过，不称真实GPU/评分质量pass。

## 容量与研究边界

回传前服务器文件系统余613561663488 bytes（约571.424 GiB），`quota -s`返回0空，未取得数值账号额度；不是实测全量scratch峰值。两端UTC分记，远端10:13:55.690285；2025退出multiprocessing清理Errno39告警永久保留，作业0与输出SHA一致，不称无告警。

比较名称是**同fixed103公开成品描述性横评**。K1在这批开发集上选模；2025公开训练重叠未知；Bio完整七折103/103训练成员暴露，Bio尚未纳入本次三路完成表。不能升级为独立泛化、官方排名或临床有效性；原科学状态`experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed`及GPU实测门`waived_by_user_not_pass`保持。没有影响账号其他作业、删旧数据/失败、重新安装或修改原模型。
