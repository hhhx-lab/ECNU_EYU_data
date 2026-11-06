---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# 2026比赛数据、2025包名与seed42划分的关系

复核UTC：2026-11-06T13:21:52Z，北京时间21:21:52。仅复核本地保存的2026官方页面整理、G2原manifest、既有split与外部模型登记；不重新下载数据、不读MRI/GT/checkpoint、不连接服务器、不重划split或提交作业。

## 1. 对上一条表述的补正

用户所说“我们从2026比赛获得数据，再按随机种子划分内部留出集”与本地记录一致。上一条只说“fixed103来自公开2025标注训练池”，省略了**2026比赛发放入口沿用2025命名数据包**这一关键背景，容易误导成我们另换了去年数据。完整口径应为：

> fixed103是2026 Task1所列带标签训练数据池，经本项目seed42患者组划分后的真实T2W验证子集；官方入口所列包名沿用BraTS2025，该病例全集与2025外部模型登记全集一致。年份不是我们替换了数据或改了划分。

“比赛提供年份”“包名/数据版本年份”“扫描采集年份”“模型参赛年份”必须分别记录。本次只验证前两者的本地文档与manifest对应关系，没有核验每张MRI的采集年份。

## 2. 2026官方入口究竟列了什么

本地[2026 Task1英文整理](../../../../../01_project_documents/01_competition_and_assignments/task1_2026/BraTS2026_Task1_English.md)开头注明：来源[2026 Task1页面](https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097)，访问日期2026-05-04，是结构化提取/转述，非完整逐字网页镜像。Data Access明确写BraTS2026 Data Access Team，Data Files列如下：

| 2026页面记录的文件 | Synapse ID | 页面所列Modified On |
|---|---|---|
| `MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip` | `syn64919665` | `4/29/2026 3:48 AM` |
| `MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip` | `syn64919141` | `4/29/2026 3:48 AM` |
| `MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip` | `syn65888166` | `4/29/2026 3:48 AM` |

训练包页面所列MD5为`a67d67f756c8ef14e8dbda08ca73688c`。这是保存页面中的字段，不是本次重下载/重哈希验收，也不是已取得2025历史原版本的跨年byte比较。G2原`real_train_manifest.csv`实际病例路径也使用上述BraTS2025训练目录及`UCSD - Training`子目录。没有找到当前仍可验收的原始下载传输回执；不把页面整理夸大成重新在线访问或原ZIP完整性复核。

## 3. 我们的103例怎么来的

既有split JSON和逐例membership CSV均记录`seed=42`、`patient_group_rule=remove_final_numeric_case_suffix`及`g1_v3_seed42_authentic_anchor_then_patient_group_balanced_missing`策略。按患者组划分，保证同患者不同时间点不跨组；不是随意抽103张扫描，也不是拿官方validation的无标签数据自行称测试。

| 数据范围 | Train | 内部Val | 内部Test |
|---|---:|---:|---:|
| QC后master1295例 | 1035 | 130 | 130 |
| 其中真实完整T2W的real-only1030例 | 823 | **103** | **104** |
| 其中缺失/异常T2W的265例 | 212 | 27 | 26 |

具体生成器先用authentic病例作G1 V3 seed42患者组anchor，再补充分配missing-only组，不移动anchor。上表展示最终既有划分，**本次没有运行生成器输出新split**。

Conda base标准库集合核对结果：

- fixed103 `source_case_id`集合 = real-only membership中`split=val`的集合 = master中`split=val AND t2w_status=authentic`的集合，103/103一致。
- 103扫描属于69患者组，与本项目master train患者组交集0，与master test患者组交集0。
- 来源44例`train_top_level`、59例`train_ucsd`。
- 用户泛称的“我们随机划的测试集”是内部留出之意；正式文件中本次横评用的是**val103**，而real-only **test104未用于本次横评**。

原始manifest1296例，经QC留1295。一个不合规病例排除的历史记录保留，不在本次修改QC。seed42保证的是本项目内部划分可复现；它不保证别人用不同划分或完整交叉验证训练的模型也没有见过我们的留出病例。

## 4. 2025和2026是否是同一套

本次进一步比较了**完整病例全集**，不只是103个留出病例：

| 比较对象 | 唯一病例ID数 |
|---|---:|
| 本项目原`real_train_manifest.csv` | 1296 |
| 2025冠军镜像`BraTS2025MET`登记 | 1296 |
| 两者ID交集 | **1296** |
| 仅存在于任一方的ID | **0 / 0** |

因此现有证据支持：**2026提供的这批公开训练病例池沿用2025命名，与2025模型登记是同一个病例ID全集。** 不是我们拿一套全新且外部未见的2026患者数据，去测试去年模型。更准确地说，“同一病例池”已通过ID核对，而“两个年度所有影像/标签文件完全相同字节、标签版本无变化”没有验收；不能据此推断两个年度官方隐藏test也是同一套。主办方沿用公开训练数据本身不是数据泄露。

## 5. 对泄露疑问的准确影响

我们的K1/E训练患者与内部val103不重叠，split并未因此失效；但val103曾用于开发选模，仍不是最终锁定测试。2025模型登记覆盖整个同源训练池，实际所用checkpoint各fold训练/选模/后处理名单仍未知；Bio-Intel完整集成按其公开split直接涉及103/103训练折。**对我们是留出的，不等于对外部成品也是留出的。**

所以训练污染风险存在于“拿同源公开训练池里我们的内部验证子集做外部权重泛化比较”，而不是仅凭2025文件名指控比赛数据泄漏、篡改我们的数据或推理读取GT。现有72指标/模型/预测/旧失败全部保留，不改正式比赛成绩、不解锁104，也不更新Bio远端进度。

本次各源文件SHA、全集交集和split核验见[结构化补正](COHORT_YEAR_AND_SPLIT_CLARIFICATION_20261106.json)。工具仅Conda base标准库，本地JSON/CSV/文本读取；旧[数据来源报告](DATASET_PROVENANCE_REVIEW_20261106.md)需与本补正共同阅读，不能继续把“2025公开训练池”理解为非2026发放来源。

## 6. Pre/Post是否分开处理

UTC2026-11-06T14:58:32本机补充核查：官方描述是**治疗前/治疗后**，不严格等于手术前/手术后；UCSD纵向资料还包括非手术治疗场景。T1N/T1C的造影剂注射前/后是另一含义，不能当作治疗状态。

| 层次 | 现有处理 | 边界 |
|---|---|---|
| 官方任务 | 同时覆盖pre-treatment和post-treatment，RC=4表示切除腔 | 未取得可核验逐例临床pre/post清单，不断言官方其他位置没有该元数据 |
| 本项目manifest/split | 检查原manifest、master/real-only membership及fixed103四张表，没有pre/post-treatment、surgery或timepoint临床状态列 | `split_source`是数据来源，`patient_group`和病例后缀用于身份/纵向分组，不是术前术后诊断 |
| 本项目划分 | seed42按患者组随机，authentic anchor及missing组分配；患者跨split交集0 | 没有按临床pre/post做比例分层，同患者时间点放在同split不等于分开训练 |
| 本项目S2训练 | 共用分割网络，保留标签4/RC；K1等有RC-aware增强 | 按RC类别处理不等于输入临床状态或分别训练术前/术后模型；本次不证明外部2025作者的内部训练策略 |
| 本次三路比较 | 各模型同103例，输出ET/RC/TC/WT区域指标及患者组配对统计 | 未报告真正pre/post临床子组；RC区域分数不是整个术后人群分数 |

既有G2 `labels_present_after_overlay`字段与fixed103 ID交叉显示**16/103有RC标注，87/103没有RC标注**。这是标签登记的本地统计，没有重新打开GT体素或验证当前临床病历；不能写成“16术后、87术前”。无RC可能是未经手术的治疗后、无可见切除腔等情况，不能仅靠这一代理赋临床类别。病例`-000/-001`后缀也不能自动映射到术前/术后；103后缀分布52/20/17/12/2（000至004）只是身份统计。

真实pre/post横评需要可追溯的病例级治疗状态，保持原fixed103与患者组关系，并对分层有效病例数/患者数、区域分母及置信区间单列。若只能取得RC信息，可做“RC-present/RC-absent”描述性敏感性分析，但不得冒充术前/术后成绩。现有MET某指标的finite分母也不是全部有RC病例数，不能拿它推算术后人数。

结构化证据和八份源SHA见[TREATMENT_STATUS_REVIEW_20261106.json](TREATMENT_STATUS_REVIEW_20261106.json)。本次不改模型、训练/评分代码、split或已有分数，不提交额外作业、不下载数据或读取MRI/GT/checkpoint，不更新Bio远端状态。

## 7. 当前服务器保留数据有没有逐例Pre/Post表

上述本机复核之后，按授权做了一次有界只读目录/元数据检查，使用[专用检查器](operator/inspect_official_clinical_metadata.py)，原始[RESULT](operator/inspect_official_clinical_metadata_20261106T150544Z/RESULT.json)及[本机回执](operator/inspect_official_clinical_metadata_20261106T150544Z/LOCAL_OPERATION_RECEIPT.json)保留。远端开始/完成UTC14:54:23.741618/14:54:30.002355，本机开始/完成UTC15:05:44.465368/15:05:58.569887分别记录，不宣称时钟同步。

- `/public/home/zqchen/data`当前规范化/保留目录：1298目录、7779条目；没有找到元数据文件、ZIP/tar归档或按临床pre/post命名的目录，元数据内容读取0bytes，错误和枚举截断均空。
- 项目`work_space/G1/data/raw`：2目录、3条目，仅保留corrected-labels目录；没有训练原ZIP、逐例临床状态表或pre/post目录。它不是完整未修改官方发行包。
- 因此**当前检查的服务器数据目录没有可直接使用的病例级pre/post表，本机主manifest/split也未接入这项临床状态**。不能推论原ZIP、Synapse平台其他表或管理员别处从未提供过；也不能据目录缺表说RC标签丢失。

检查仅枚举目录和候选元数据，不打开MRI/GT/checkpoint，不查询或变动作业；复用用户已连接VPN，`vpn_owned=false`、`preexisting_vpn_preserved=true`，未关闭用户会话。

## 8. RC有无是否进入loss与评分分母

后续用户询问presence indicator与权重归一化，已独立追溯原E合同、7月Git源码与K1历史SHA清单，详见[RC存在性/loss/有效样本复核](RC_PRESENCE_LOSS_REVIEW_20261106.md)及[结构化审计](RC_PRESENCE_LOSS_REVIEW_20261106.json)。focal按有效GT体素权重和归一化，无RC不额外加入权重3；没有单独的病例/类别presence开关。参考Dice保留全部前景通道、smooth避免0/0，完整历史已安装底层源码未独立验收。16有RC扫描只属8患者，病灶Dice实际n15；阴性87例的MET RC误报扫描数2025/K1/原E为0/4/3，不能仅靠阳性Dice判断全部RC能力。无新评分、训练/模型/划分改动或依赖安装。
