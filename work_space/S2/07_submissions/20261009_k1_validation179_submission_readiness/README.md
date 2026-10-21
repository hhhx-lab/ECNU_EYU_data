---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# K1-175 官方 validation 提交入口核查

核查时间：2026-10-09 23:00（Asia/Shanghai），UTC 2026-10-09T15:00:10Z。

用户授权将K1路线预测提交至BraTS 2026 Task1并查看validation leaderboard。通过现有Chrome + Playwright Bridge登录会话读取[挑战页面](https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097)。主页状态显示“关闭”。项目`syn76260128`显示6个旧预测文件，其中原E为`syn76456272`，另有1个旧Docker条目；当前列表未见K1文件。

为核实可选评估队列，仅选择原E旧条目打开“Select Evaluation Queue”对话框。该步骤未产生提交。对话框的Evaluation Queues表格显示`No rows`，最终`Submit`禁用。没有上传文件，没有提交原E或K1，没有submission ID，也没有新评分。当前账号页面无可选评估队列，不能在此入口完成K1刷榜；本记录不将页面不可用原因进一步推断为账号权限故障或确认全部后台队列永久关闭。

本地S2文件检索未见K1对应官方validation179的推理/提交验收包；已验收K1的103份预测属于内部fixed103开发集。若未来重新开放队列，需使用冻结K1 selected175 checkpoint SHA `28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204`，先完成比赛validation179逐病例推理、病例ID/geometry/finite/合法label0–4及SHA验收，再按官方格式打包并唯一提交，不能提交fixed103或将原E旧包改名作K1。

本次只核查网页和本地已有证据；未连接ECNU、启动VPN、训练、GPU推理、打包或上传。旧实验和旧失败证据保留。工程入口核查不代表K1科学门或正式优越性通过。

## 刷新后的榜单

刷新观测UTC为`2026-10-09T15:01:53.152Z`。页面仍显示Closed，榜单标题为`Validation Leaderboard (unranked)`；页面计数为1,004条，当前页读取25条。没有本次K1提交ID或新的K1评分。

当前页可见以下旧团队记录，数值按网页显示精度转录：

| Submission ID | 团队 | 页面日期（时区未标明） | WT病灶Dice | ET病灶Dice | TC病灶Dice | RC病灶Dice | WT小病灶F1 |
|---|---|---|---:|---:|---:|---:|---:|
| 9774293 | ECNU_NYU_team | 7/31/2026 1:40 AM | 0.674 | 0.704 | 0.729 | 0.427 | 0.299 |
| 9774292 | ECNU_NYU_team | 7/31/2026 1:40 AM | 0.674 | 0.704 | 0.729 | 0.427 | 0.299 |

这两条不是本次K1成绩；仅凭团队名不能确认绑定原E，需查原提交审计。不能将官方validation榜单与内部fixed103表作同病例paired比较，也不能据此给出K1的官方名次。榜单标为unranked，不将页面行序当作排名。

## 证据与后续条件

### 上传与评分的澄清（2026-10-09 23:05 Asia/Shanghai）

用户截图正确显示“上传文件”可用。重新核查按钮`disabled=false`，浏览器已出现文件选择窗口；取消选择后未上传任何文件。此前“不能提交”的阻塞特指**无可选评估队列，不能提交评分**，不能表述为“上传按钮不可用”。上传到项目、选择评估队列、完成评估获得榜单成绩是不同阶段。可打开文件选择窗口不代表已验证服务器实际接收文件，更不代表评估队列已开放。见[上传按钮独立观测](UPLOAD_BUTTON_OBSERVATION.json)。

- [评估队列截图](QUEUE_DIALOG.png)与[可访问性快照](QUEUE_DIALOG_SNAPSHOT.md)：No rows，Submit disabled。
- [对话框及页面观测](PAGE_OBSERVATION.json)：含1,004条页面计数。
- [刷新后的原始观测](LEADERBOARD_REFRESH_OBSERVATION.json)：保留25条当前页记录。其`k1VisibleInTables=true`是过宽正则的启发式误匹配，不能作为K1文件/提交/评分存在的证据；`totalRows=null`仅表示该次提取正则未识别英文计数，不表示榜单为空。

当前需主办方确认是否重新开放Task1的validation评估队列，或提供有效提交入口。开放后仍须先完成K1官方validation179预测包的独立验收，再唯一提交；本次未启动这项推理。记录整理时间为2026-10-09 23:03（Asia/Shanghai），UTC约15:03。
