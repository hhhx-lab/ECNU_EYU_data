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
