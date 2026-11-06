# 2025、K1和原比赛E同fixed103统一评分结果

## 完成与身份

CPU评分作业 **4319117** 的allocation/batch均为 **COMPLETED 0:0**，node203，耗时 **01:13:43**。同JobID和冻结合同的COMPLETE、三路各103例/72项指标、K1与原E历史逐例重放、20000次患者组配对bootstrap均已完成，无FAILED或静态包漂移。本目录只拉回12份轻量文件，共248452 bytes；大小/SHA与服务器完成清单、本轮快照一致。

- 2025：固定2025 METS冠军公开镜像，原推理4310575完成的103份预测。
- K1：selected epoch175，checkpoint SHA256 `28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204`。
- 原比赛E：epoch200，checkpoint SHA256 `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`；不是E-SNFH175或FixV3。
- Bio-Intel：本轮4319113仍PENDING/Priority，0/103预测；不在本次完成表中，尚无Bio评分作业。

K1和原E的新评分各自与其历史逐例72项完全一致，最大绝对差异0，schema/NA一致；这是两路分别复现历史，不表示K1与原E相互成绩相同。

## 关键指标

同一fixed103验证集、同一冻结官方MET评价环境。下表为病例等权的病灶指标均值，不是全图体素Dice；资格与FP/FN惩罚沿用官方实现。有效分母n按指标定义，NA不补0。

| 指标 | 2025 | K1 epoch175 | 原比赛E epoch200 | 三路各自有效n |
|---|---:|---:|---:|---:|
| WT病灶Dice | 0.703347 | 0.630263 | 0.617613 | 87 |
| ET病灶Dice | 0.706513 | 0.637271 | 0.611652 | 85 |
| TC病灶Dice | 0.747662 | 0.688071 | 0.657715 | 86 |
| RC病灶Dice | 0.698966 | 0.322970 | 0.311187 | 15 |
| WT病灶NSD | 0.706655 | 0.627997 | 0.613113 | 87 |
| WT病灶HD95（评分器原值） | 62.144908 | 77.955327 | 85.047330 | 87 |
| WT小病灶F1 | 0.338859 | 0.261962 | 0.294659 | 46 |
| WT大病灶F1 | 0.743912 | 0.701538 | 0.688472 | 103 |

2025在以上指标的点估计领先两条本地路线；不能将此解释为正式优越性或独立泛化。WT全实例F1为2025 0.784637(n93)、K1 0.748435(n93)、原E 0.728257(n96)，分母不同；配对判断必须读取共同finite病例结果，不能当成三路均103例的均值直接比较。

## 全量结果与验收

- [全部72项三路均值报告](reports/COMPARISON_REPORT.md)：显示各模型各指标的有效n。
- [216条均值和有效分母](reports/MEANS_216.csv)：3模型×72指标，保留原精度。
- [216条配对结果与置信区间](reports/PAIRED_ALL_216.csv)：3模型对×72指标，只用每对共同finite病例；20000次患者组重采样，seed20260831。
- [K1/原E历史复现](reports/HISTORICAL_REPLAY.json)、[完成合同及输出SHA](reports/COMPLETE.json)、[下载验收](DOWNLOAD_ACCEPTANCE.json)。
- [本地独立轻量复核](../../COMPARISON_COMPLETION_REVIEW_20261106_1223UTC.json)：复算所有216条均值和216条配对共同分母/均值/原始差值；没有在本机重新计算bootstrap CI或重读MRI/GT。
- 三路逐例原始CSV在各自`evaluation/<model>/leaderboard_metrics.csv`，103病例后另有mean/std/median三个summary行，不把summary算成病例。

## 数据与结论边界

fixed103是103扫描/69患者组的**验证集**，不是锁定test104/master test130或官方179。Python3.10.18、BraTS-evaluation0.0.8、panoptica2.1.0、NumPy1.26.4；27mm³资格阈值、parser DSC0.2、matcher1e-6，ET3/RC4/TC1∪3/WT1∪2∪3，TC与WT不含RC。原NSD默认0.5及非整数dtype告警保留。

K1曾在本开发集选模；2025公开dataset登记含fixed103的103条，但实际fold训练成员/训练日志未知；Bio完整集成的103/103训练成员暴露已披露。比较仅称“同fixed103公开成品描述性横评”，不称独立泛化、官方排名或临床有效性。不读取其他测试集，不下载原MRI或模型，不恢复监控。

完成remote UTC `2026-11-06T11:41:36.142287+00:00`；快照remote UTC `12:07:38.040352`；本机下载接收UTC `12:20:36.451110`、finally `12:20:36.452238`；本地复核UTC `12:23:51.667888`。两端时间分别记录，不声称时钟同步。

服务器文件系统余588020449280 bytes=547.63671875GiB，`quota -s`返回0但无数值，不能据此声称账号配额已验收。用户VPN复用保留，本轮没有提交/释放/安装或远端写入，旧失败和冻结合同保留。

状态保持`experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed`；豁免门仍为`waived_by_user_not_pass`。Bio完成需COMPLETED0:0及103预测SHA/几何/标签验收后，另建独立CPU评分及四路汇总，复用本次已完成三路，不覆盖历史。
