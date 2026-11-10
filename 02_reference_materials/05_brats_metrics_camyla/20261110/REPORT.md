---
title: "BraTS METS 评分与 Camyla 深度调研"
subtitle: "2025/2026 论坛规则与公开分割手稿"
author: "ECNU 项目资料复核"
date: "2026-11-10"
lang: zh-CN
---

# 调研结论

本报告回答两个问题：两年 BraTS 论坛对 validation 和最终排名究竟说了什么；Camyla 的“40 篇手稿”能否找到，其中哪些对我们的脑转移瘤分割有帮助。

**最重要的结论不是发现了一个新的万能指标或可直接下载的转移瘤冠军模型，而是发现了几处必须区分的评价口径和复现缺口。**

1. **2025 METS validation 在比赛中途明确省略病灶级评价，改为病例级评价。** 2025-07-09 官方公告写明原因是耗时过长。不能把去年官方成绩直接与今年的病灶级 DSC/NSD、小灶 F1 对比。[B1]
2. **2026 官方 validation 榜单始终是 unranked；最终名次来自隐藏 test 的离线评价。** 参赛者制作的自动综合榜不是官方排名。[B3]
3. **2026-07-26 官方宣布最终分别报告检测、分割两张排名榜。** 早期关于三指标综合排名的答复不能覆盖这条后续公告；本次没有取得最终实际采用的累计排名实现与全部统计参数。[B8]
4. **METS 的小病灶阈值是物理体积 27 mm³，不是旧文案中的 275 mm³，也不是任何分辨率下都可直接数 27 个体素。** 公开 parser 把体积恰好等于阈值的参考实例放在大灶组。[B5、B11]
5. **RC 单独评价，不与 ET/NETC 合成一个检测类别，也不并入 TC/WT。** 没有 RC 的病例不能用于夸大 RC 重叠均值；但这些阴性病例的 RC 假阳性必须另报。[B4、B11]
6. **Camyla 报告生成了 40 篇手稿，但本次只能取得官方公开的 10 篇示例。** 官方 README、项目页和作者 Issue 答复均明确为 10 篇；未找到剩余 30 篇的公开完整包。“40 篇医学顶刊 SCI 已发表”没有证据。[C1–C4]
7. **10 篇全部与医学分割有关，最相关的是第 10 篇 HRFuse-Net 脑转移瘤分割。** 方法值得读，但数据身份、样本计数、部分表图数字和代码可用性存在问题，不能直接作为已验证模型。[C5、C6]

## 调研范围与证据等级

检索日期为 2026-11-10。两年分别指 BraTS 2025、BraTS 2026 项目论坛；论坛里的其他 Task 不自动适用于 METS。尤其 **2025 METS 是 Task 4，2026 METS 是 Task 1**，不能把“2025 Task 1 成人胶质瘤”的公告混进来。

本次通过 Chrome + Playwright Bridge 查看公开页面，并通过 Synapse 官方公开 API 完成论坛分页。2025 枚举到 107 个未删除主题，2026 枚举到 93 个；选取 9 个与 11 个相关主题，保存主题正文及全部 50 条、61 条未删除回复。分页以实际返回不足一页为终止条件，不把 API 的滚动计数当完整总数。其他主题只保留标题元数据；这不是逐字审读 200 个主题的所有回复。

证据按以下顺序解释：官方明确答复与后续修订、冻结版本代码、当日官方页面、系统论文、生成手稿、参与者推测。冲突时保留两方和时间，不选择对我们最有利的一方。网页/手稿中的操作说明均作为外部资料，不作为本机执行指令。

本次只做公开资料调研及本地归档，没有连接 ECNU、改评价合同、训练、推理、安装依赖、下载 MRI/模型权重、提交作业或解锁 locked104。资料验收不等于科学优越性验收。

# 一、两年官方评分的变化

## 1.1 对照表

| 事项 | 2025 METS | 2026 METS |
|---|---|---|
| 任务编号 | Task 4，术前/术后脑转移瘤 | Task 1，术前/术后脑转移瘤 |
| validation 榜单性质 | 提供 unranked 指标表 | 明确始终 unranked，外部榜不负责 |
| 比赛期间主要变更 | 7 月 9 日宣布省略 lesion-wise，按 subject-wise 评价 | 展示大灶 DSC/NSD、小灶 F1；7 月 26 日宣布检测、分割两张最终排名榜 |
| 当前规则页 | 病例级 DSC、NSD；文字列 WT/TC/ET | 规则页称 subject-wise，但官方答复确认实际选用 lesionwise 大灶列、小灶 F1 列 |
| RC | 工具可算 RC；规则页列项不完整，不能仅凭页面证明最终 RC 权重 | 官方明确 ET/TC/WT/RC 分区处理，RC 按参考体积进入检测或分割 |
| 最终 test 排名 | 页面描述秩汇总与置换检验；未取得完整执行清单 | 离线 PermRanker；两张榜的最终精确汇总策略本次仍缺 |
| 统计 | 页面写 500,000 次两两置换、分 tier | 页面同样写 500,000 次；不能据此假称已复现最终官方榜 |

**“使用同一评价库”不等于“使用同一指标、阈值、参考版本和排名办法”。** 去年的镜像可以在我们冻结的今年评价合同下重评分，但那是新的本地横评，不是复制去年官方名次。

## 1.2 2025：为什么后来不做病灶级

2025-07-09 19:45:55 UTC，METS co-organizer Nikolay Yordanov（公开作者 ID 3497218）在 Announcements 回复 34927 宣布：

> “due to the extremely long processing times for the validation phase, we are omitting the lesion-wise metrics ... only ... at the subject-wise level.”

同条公告给出 [Astarakee/brats_evaluator](https://github.com/Astarakee/brats_evaluator)。此前 validation 开放帖中，Mehdi Astaraki 解释：一个病例常有 20 多个病灶，逐病灶、逐区域计算尤其 NSD 很耗时。参与者先后报告 9 小时、两天、约六天等待；这些是平台评分耗时报告，不是模型推理速度。[B1、B2]

当前 2025 METS 规则页也写 DSC/NSD 为 Subjectwise。因此，“2025 首先尝试逐病灶，但 validation 后来改为病例级”的时间线有直接证据。公告正文说 METS 评价将仅按病例级进行，但本次未取得最终 test 的实际逐指标配置；不扩展成“已独立验证全部最终 test 指标及 RC 权重”。

还需区分 **工具能力** 与 **官方采用范围**：该仓库 README 仍说明能同时算 subject-wise、lesion-wise，多出来的列并不证明都进入官方排名。当前规则页只列 WT/TC/ET，而仓库又可定义 RC 等区域，说明不能靠一个 README 推出最终完整名次公式。

## 1.3 2026：官方不认可第三方 validation 综合排名

主题 13508 的发帖者搭建了 [BraTS2026-MET 自动榜](https://vineetjangiriitb.github.io/BraTS2026-MET/)，称每三小时刷新，将 12 个指标合成排名。参与者进一步猜测可能先按四区均值再排名，但本人写了“not 100% sure”。

2026-06-19 12:26:41 UTC，Mehdi Astaraki（ID 3450172）在回复 36364 明确说：

> “the validation leaderboard ... will remain explicitly unranked.”
>
> “assumes no responsibility for any unofficial rankings or external leaderboard interpretations.”

最终隐藏测试离线遵循 PermRanker，前三名在 workshop 公布。[B3]

所以我们此前看到的第三方“第几名”、当前 validation 的均值排序、最终 workshop 的 top teams 是三种证据，必须分开写。即便第三方代码自称仿照比赛，也不能升格为官方名次。

## 1.4 最终选哪些指标、后来为何分成两榜

2026-06-09 回复 36324：Mehdi 描述“病例级累计秩跨指标及测试病例汇总”，加两两置换检验；大病灶用 DSC/NSD，小病灶用 F1。

2026-07-09 回复 36449进一步点名：

> `lesionwise_dsc/nsd_mean_{region}` 和 `small_instance_f1_{region}` 用于排名。

其他输出列可以帮助调参，但不是所有列都参与官方排名。RC 小于阈值用检测，较大用分割；很小的“孤岛”也按阈值进入检测，并未公告统一删除。[B6]

参与者持续指出：三个指标等权会使分割占两项、检测只占一项；大灶/常见区域的有效病例更多；FP 还会同时影响分割与检测。官方 6 月 16 日承认 FP 确实惩罚分割，并表示检测/分割权重稍后宣布；7 月 11 日仍说下周宣布。[B7、B5]

**最终找到的后续答复是 2026-07-26，而不是“确定三项永远等权”。** 回复 36558（主题 14024）及 36559（主题 13146）相同公告：

> “two final ranking leaderboards - one for the detection and one for the segmentation arm”
>
> “After the submission deadline, we will employ different strategies ... select the approach that is both fair and informative.”

这是官方针对争议的实质修订。但它没有给出最终选用的全部权重、缺失指标处理、分区域/分病例归一化和最终秩矩阵。因此可以确认“两榜”，**不能声称本次已精确重建官方累计排名**。[B8]

## 1.5 27 mm³、275 mm³与默认参数陷阱

主题 13893 提问者发现文案从 275 mm³ 变成 27 mm³。2026-07-11 10:24:53 UTC，官方确认阈值为 **27 mm³**。[B5]

当日抓取的 Evaluation wiki 639602 的可见正文为 27 mm³；原 275 的句子仍位于 HTML 注释中。不能把注释当现行规则，更不能把文章阅读器抽出的隐藏文本当新阈值。

物理体积应按参考原始网格计算：

$$V=n_{\mathrm{vox}}s_xs_ys_z.$$

本次保存的公开 parser 在冻结提交中使用 `instance_volume >= vol_threshold`。因此配置为 27 时：**小灶 <27，大灶 ≥27**；网页“larger than”是简写，等号边界应按实现核验。不得根据训练裁块的碎片体积或统一重采样后的标签体素数重新分组。[B11]

另一个新发现是：当前公开 README 示例和 parser 默认参数为 **20.0、0.1**，不等于我们的 **27 mm³、DSC 0.2**。公开 config 的匹配器配置也不能直接代替我们冻结的 panoptica 环境。必须显式记录版本、配置 SHA、matcher、parser 阈值与 NSD 容差，而非只运行一个裸默认命令。

这不是本次修改官方代码或重新评分的授权。本项目原冻结 BraTS-evaluation 0.0.8 / panoptica 2.1.0 / NumPy 1.26.4、27/0.2 合同保持不变；公开源码只供研究对照。官方 pipeline 名称相同不证明 runtime 等价。

## 1.6 FP 的惩罚有多层，不只影响检测

在 2026-06-16 回复 36347，官方确认当前实现用参考实例决定大小；未匹配预测计为 FP，不再按预测大小细分。它们会惩罚分割指标。主题 14024 用构造例说明，由此可能鼓励保守预测。[B7、B8]

公开 parser 的关键行为是给大灶 DSC/NSD 列表追加每个 FP 的 0；给漏掉的大参考灶追加 0；小灶 F1 也使用 FP 计数。其概念可写为：

$$\mathrm{F1}_{\mathrm{small}}=\frac{2\mathrm{TP}_{\mathrm{small}}}{2\mathrm{TP}_{\mathrm{small}}+\mathrm{FN}_{\mathrm{small}}+\mathrm{FP}}.$$

$$\mathrm{DSC}_{\mathrm{large,penalized}}=\frac{\sum_{j=1}^{N_{\mathrm{large}}}\mathrm{DSC}_j}{N_{\mathrm{large}}+\mathrm{FP}},$$

其中漏掉的参考大灶取 0，FP 也贡献 0。这只是该 parser 汇总语义的说明，不替代完整实例匹配算法。

但有一条非常重要的条件：**没有参考大灶时，其分割列是 NA；没有参考小灶时，其小灶列也可以是 NA。** 不能把“每个 FP 都会加零”夸大成“所有 RC 阴性病例都进入 RC 病灶 Dice 分母”。恰恰相反，RC 阴性 FP 需要单列计数，才不会被重叠均值藏起来。[B11]

## 1.7 RC 应独立，空类别和指标支持集要分开

主题 13631 讨论设计书“ET+NETC+RC collective lesion”与实际 config 的冲突。2026-06-30 回复 36399，官方明确要求按 **config_mets.yaml 的逐区域实现**，并承认设计文案不对齐。[B4]

实际区域为 ET=[3]、TC=[1,3]、WT=[1,2,3]、RC=[4]。RC 不是 NETC，也不能为了复用三类模型把 RC 并进 TC/WT。各区域实例分别匹配，体积判断依据各自参考实例。

对我们最直接的启示是把“RC 真值存在”“该指标可计算”“模型有没有误报”区分开。RC 阴性病例不提供 RC 重叠质量证据，却提供误报证据。训练 loss 的 class-presence indicator、评分支持集 indicator 与临床术前/术后标签也不是同一个东西。

如后续研究重新设计 loss，应预注册有标签但为空、标签未标注、该类确实不存在三种情况，说明每项 loss 的有效分母；不能用 GT presence 在真实推理时直接决定是否输出 RC。本次未逐源码复核原训练 loss，也未修改它，不能凭论坛答复声称原模型已正确采用某种 indicator。

## 1.8 validation 真值有修订，选模与最终提交不是同一件事

2026-06-04 的主题 13294 公告：在 **June 4, 20:00 CET** 之前提交的受影响任务已自动用新参考重跑，之后提交从一开始使用新参考，无需参赛者操作。公告原文写 CET，本报告不擅自将它改成夏令时后再报另一个 UTC 时间。[B9]

这意味着跨日期、跨原始 CSV 比较必须确认 GT 版本及是否已重评分，不能把同 submission 的两个数字直接解释成模型变化。

2026-07-24 回复 36552 则说明：参赛者要自己选一个方案，构建、验证 Docker 后提交隐藏测试。不是比赛替我们从多次 validation 的每个区域各挑一个最好数字拼成“模型成绩”。[B10]

## 1.9 PermRanker 是统计排名框架，不是分割模型

[PermRanker 论文](https://link.springer.com/chapter/10.1007/978-3-032-16365-3_1)题为 *My Model Is Better Than Yours! Statistically-Aware Ranking for Fair Benchmarking of AI Models*，2026-04-01 出版。公开摘要描述两个阶段：病例级累计秩汇总，以及两两置换检验。

它不直接生成分割，也不是替代 DSC/NSD 的一个神奇分数。秩聚合对量纲敏感性较低，但名次仍取决于候选模型集合、指标、有效病例、缺失值规则和权重。我们现有患者组 bootstrap CI 也不是官方 PermRanker 的复现。

Springer 页面明确为订阅内容预览。本次保存了公开页面与摘要，**没有取得或绕过权限下载付费全文**；公开检索未找到可确认的完整免费副本。规则页的 500,000 次置换只能称官方描述，不代表本项目已实际执行该排名流程。[B12]

# 二、对现有四路统一比较的具体影响

## 2.1 哪些数字仍能使用

我们的 2025 冠军镜像、2026 Bio-Intel 完整 A5+B2、K1-175、原 E-200 已在 **fixed103 内部验证集**上完成统一 MET 评价；103 扫描、69 患者组，不是官方 validation179，也不是 locked104。[L1]

| 同合同指标 | 2025 镜像 | Bio-Intel | K1 / 原 E |
|---|---:|---:|---:|
| WT 病灶 Dice，n=87 | 0.703347 | 0.685620 | 0.630263 / 0.617613 |
| WT 病灶 NSD，n=87 | 0.706655 | 0.677676 | 0.627997 / 0.613113 |
| WT 小灶 F1，n=46 | 0.338859 | 0.327950 | 0.261962 / 0.294659 |
| RC 病灶 Dice，n=15 | 0.698966 | 0.710364 | 0.322970 / 0.311187 |

这些是 **同一合同下的观察性横评**，不会因为 2025 官方后来采用另一套指标而作废。但它们不能称为“去年官方冠军分数”和“今年官方排名”的直接比较。

Bio 的 RC Dice 稍高，却在 87 个 RC 阴性扫描中出现 31 个 FP 实例，涉及 17 个扫描；2025 为 0，K1 为 4，原 E 为 3。只有 16 个 RC 阳性扫描、8 患者组，其中 15 个支持该病灶分割指标。用 0.710364 单独代表 RC 综合表现是不充分的。[L1]

## 2.2 为什么不应该造一个“总分冠军”

将 WT/TC/ET/RC 的 DSC/NSD/F1 共 12 个均值简单平均，有四个问题：量纲/权重不是官方最终定义；每列有效病例不同；RC 样本少；FP 重复惩罚可改变分割与检出的权衡。HD95 更不能与 [0,1] 指标直接平均。

建议继续保留分割、检测、RC 阴性误报三组结论。报告指标有效 n、共同 finite 支持集、患者组配对区间。若需要探索性综合排名，必须另行命名和预注册敏感性分析，明确不是官方复现，不原位替换现有冻结表。

2025 相对 Bio 的 WT Dice 描述性差为 +0.017727，原患者组区间跨 0；这不是“2025 显著碾压 Bio”。K1 的 Dice 高于原 E，但 WT 小灶 F1 反而低于原 E；两张榜的思路与这种多指标权衡一致。

## 2.3 数据暴露问题得到更直接的官方解释

2026 官方 Data wiki 639600 明写：

> “we are utilizing the dataset used in 2025”

今年做标注 QC，并计划新增非标注数据。7 月 21 日左右的主题 14058 又公告 UCSD 训练标签修订。**“2026 比赛下载”与“沿用 2025 数据池”可以同时为真**，不是把我们的随机分出的内部集变成官方 test。[B13]

因此，对去年训练模型而言，今年的内部留出并不自动是未见患者。是否某一具体病例训练暴露，需要模型训练清单与病例/患者身份、字节 SHA 证明；共享年度数据池不是每个文件百分百相同的证明，也不是已经证实某队作弊。

2026 的规则和论坛还明确禁止将外部有标注数据或其他分割任务预训练权重用于参赛模型；可单独做外部研究并在论文说明。[B14] 因此，**PROTEAS/HRFuse-Net 可作为当前研究参考，不能仅因公开就默认可用于当年正式参赛训练**。本次也未开展新增外部训练。

# 三、Camyla 的“40 篇”究竟是什么

## 3.1 正确入口与公开范围

用户给出的 `github.com/yifangao12/camyla` 返回 404；实际官方用户名多一个“1”，为 **yifangao112**。

- [Camyla 官方仓库](https://github.com/yifangao112/Camyla)。
- [系统论文 arXiv:2604.10696](https://arxiv.org/abs/2604.10696)。
- [官方项目页](https://yifangao112.github.io/camyla-page/)。
- [10 篇 PDF 原目录](https://github.com/yifangao112/Camyla/tree/main/assets/paper_pdf)。
- [CamylaNet 分割框架](https://github.com/yifangao112/CamylaNet)、[nnPrep 数据转换工具](https://github.com/yifangao112/nnPrep)。

本次冻结 Camyla 提交 `df4434f9d4aef5b7394ed03a4e877a8130c1b6cf`，2026-04-14，完整树未截断。系统论文、10 篇 PDF、31 个数据/想法 JSON、双语 README、LICENSE 共 **45 个文件、31,821,359 bytes**，逐项核对远端 Git blob 身份、下载大小、SHA256。另保存项目页、Issues、Releases 和数据原始论文。

对完整 40 篇的追查包括：完整仓库树、README、项目页资源区、全部现有 3 个 Issues、有关公开手稿/划分的作者回复、Releases，以及公开搜索“Camyla 40 papers”。结果如下：

| 入口 | 本次能确认的结果 |
|---|---|
| 仓库目录 | 01.pdf–10.pdf；README 明说 “A 10-paper subset” |
| Issue #1 作者答复 | 已上传 10 篇示例到 assets/paper_pdf |
| 官方项目页 | Sample generated papers 为 10 篇；Trace 仍写 will follow shortly |
| GitHub Releases | 返回空列表，无完整论文包 |
| Issue #3 作者答复 | 默认 Fold 0，预处理后 splits_final.json；各基线结果仍需整理 |
| 公开搜索 | 找到系统论文/项目页/转述，没有取得剩余 30 篇完整稿包 |

所以本地目录不是“40 篇已全部收齐”。**剩余 30 篇未取得，不用推测标题或重复 PDF 充数。** 对公开范围的结论只限本次检查入口，不断言全互联网绝无其他副本。[C1–C4]

## 3.2 对宣传数字的逐项核验

| 常见说法 | 原始证据与更准确表述 |
|---|---|
| 28 天、8 卡 | 论文报告两个独立 run、31 数据集、合计约 28 天；附录记录 RTX 4090 48 GB、5,461.9 GPU-hours，约 227.6 GPU-days。48 GB 是论文原文规格，非本机硬件验收 |
| 2700+ 模型 | 报告模型实现数量；不是 2700 个互相独立验证的可下载最终权重，也不一定是 2700 个不同训练种子 |
| 40 篇医学顶刊 SCI | 实际是两个 run 的成功任务生成 18+22=40 篇完整手稿；T1/T2 是评审质量对照，不是录用/发表通知 |
| 24/31 超越 14 强基线 | DeepSeek run 18/31，Sonnet run 22/31，两者并集 24/31；不是单个 run 赢 24 个 |
| “超越” | Dice 比最强基线高超过 0.5 个百分点，或 Dice 在 0.5pp 范围内而 HD95 更低；有些赢家不是 Dice 更高 |
| 数据完全无污染 | 作者称来自 2025 Scientific Data、避免传统基准污染；本次未独立证明 LLM 从未见过相关数据/论文，也未证明原医学图像均始于 2025 |
| 26 个 blind-test | 是系统开发层面留出 26 个数据集、另 5 个开发数据集；不能直接等同于模型搜索从未接触逐病例评估反馈 |

正文主结果中，两 run 的具体提案生成模型为 DeepSeek V3.2、Claude Sonnet 4.6；项目页还宣传其他后端的自动代理比较，不应把两类实验混为同一次 28 天试验。[C2]

## 3.3 基准能不能严谨复现

Camyla 用 nnU-Net 式预处理/训练基础，主要比较前景类体素 Dice 和 HD95；14 基线覆盖 CNN、Transformer、Mamba。它有完整方法的长周期探索、消融和稿件编排能力，但以下边界值得重视：

1. 论文明确按 evaluation fold 的 Dice 选择最强基线，并在实验搜索中反复读取评价反馈。Issue #3 作者说默认用 Fold 0；本次没取得数据对应的病例/患者划分文件。因此不能把搜索后的最优成绩直接解释成一次性独立 test 泛化。
2. 附录写基线可由代理根据收敛选择 100 或 1000 epochs。不要把“相同训练基础/预算”理解成已经独立证明所有模型训练更新数、随机种子和超参完全一致。
3. 附录 H 的部分 3D 显著性检验使用轴向切片样本；作者自己承认邻近切片相关、p 值可能偏乐观。对纵向脑转移瘤应优先以患者组为重采样单位，而非把几百张切片当几百名患者。
4. 40 篇来自成功任务，包含筛选效应；最佳 run 的并集也包含“从两个结果择优”。它可展示自动研究能力，但不证明任意新数据集都优于 nnU-Net。

**值得借鉴的是严谨记录实验候选和失败、固定数据合同、组织消融；不能照搬的是反复用内部 test 调参后仍称完全独立测试。**

# 四、10 篇公开分割手稿逐篇筛选

以下数字是生成手稿主对比表的报告值，未在我们服务器复现。HD95 的二维任务可能以像素而非 mm 报告，不跨任务比较。全部 PDF 原件保存在 `camyla/manuscripts/`，搜索文本另存，不改原稿。

## 4.1 手稿与方法清单

| 编号/模型 | 目标与数据 | 方法要点 | 对本项目的优先级 |
|---|---|---|---|
| 01 CDFL | 颅内生殖细胞瘤，GDMRI-CT，50 例 | 沿三正交方向分解、特征格交互与自适应融合 | 中：脑 MRI/方向上下文；不是转移瘤，也不是胶质瘤 |
| 02 SFA-Net | 多发性硬化病灶，MSLesSeg，115 扫描 | 多尺度空洞卷积、频域重建、双注意力 | 高：小病灶尺度差异；病理和标签不相同 |
| 03 HCF-Net | 新生儿 HIE 病灶，BONBID2023 | 低频上下文门控、高频细节支路、可分离卷积 | 中高：轻量高频细节；不是肿瘤 |
| 04 CSM-Net | 支气管肺泡灌洗细胞，PW-BALFC，2105 图 | 相邻尺度双向互细化、稀疏采样 | 低：可以借鉴融合，不直接拿二维细胞模型跑 MRI |
| 05 SAC-Net | 全景牙齿，STS-Tooth，850 图 | 中线对称注意力、级联对称细化 | 低：强对称先验不适合分布不对称的转移灶 |
| 06 SRD-Net | 腹腔镜场景，Endoscapes2023 | 结构/反光残差双支路、少数类融合 | 中低：类别不平衡思路，但反光模型不适用 MRI |
| 07 ScaleGate-Net | 肝病灶 CT，PLC-CECT，278 扫描 | 差异引导可变形跳连、各向异性上下文 | 中：尺度/各向异性融合，需防错配和几何偏差 |
| 08 BHD-Net | 胎脑组织 MRI，SMRI-FB，70 扫描 | 结构/边界拆分、边界引导可变形聚合 | 中：薄边界与高分辨率，但不是病灶检出 |
| 09 VBR-Net | 皮肤 OCTA 血管，DERMA-OCTA | 血管引导边界残差 | 低：微细结构参考，二维 OCTA 不等价 MRI |
| 10 HRFuse-Net | 脑转移瘤，LMD-BM，声称 44 例 | 并行保分辨率残差流、轴向条带上下文、门控融合 | 最高：直接主题相关，但复现门尚未通过 |

完整英文题名和远端 PDF 永久版本链接见 [手稿索引](camyla/MANUSCRIPTS.md)。

## 4.2 主对比表数字与应该怎样读

| 手稿 | 最强 Dice 基线 | 本方法 Dice / HD95 | 不能忽略的说明 |
|---|---|---|---|
| 01，表 3 | STU-Net 0.7233 / 8.4485 | 0.7387 / 5.5790 | Dice +1.54pp；正文“2.12%”是相对增幅，非 2.12pp |
| 02，表 4 | U-Mamba 0.6957 / 16.3300 | 0.7063 / 15.2980 | 消融主模型另报 0.7147 / 13.6329；不能混用 |
| 03，表 2 | MedNeXt 0.5585 / 15.8881 | 0.5823 / 11.9295 | 新生儿数据，不证明脑转移瘤改进 |
| 04，表 6 | nnU-Net 0.7666 / 52.4164 | 0.7663 / 47.8435 | Dice 略低，靠边界指标；不是“所有指标赢” |
| 05，表 3 | U-Mamba 0.9367 / 2.6072 | 0.9416 / 0.7943 | 更好的基线 HD95 为 UKAN 0.9394，必须分指标选参照 |
| 06，表 5 | nnU-Net 0.4753 / 39.0041 | 0.4813 / 45.1510 | Dice +0.60pp，但 HD95 更差 |
| 07，表 5 | U-Net++ 0.5322 / 107.4067 | 0.5544 / 85.3516 | 消融又报 0.5660 / 84.9302；主比较不是那个数 |
| 08，表 4 | MedNeXt 0.8530 / 1.2305 | 0.8574 / 1.1851 | Dice +0.44pp，低于 0.5pp，按系统规则主要是 HD95 胜 |
| 09，表 3 | U-Mamba 0.8391 / 3.9159 | 0.8381 / 3.0712 | Dice 略低；比次低 HD95 的 UTNet 3.2803 更好 |
| 10，表 4 | nnU-Net 0.4619 / 32.5388 | 0.5069 / 25.4869 | Dice +4.50pp；其他模型可能 HD95 更好，详见下章 |

这个清单不是同一数据集下的模型排行榜。差异大的病种、图像维度、类别数和单位不能混合平均。手稿中“best”“significant”“clinical”之类措辞也不构成我们的独立复核结论。

# 五、脑转移瘤 HRFuse-Net 专项审读

## 5.1 核心设计：对小灶可能有帮助的两部分

第 10 篇完整题名为 *HRFuse-Net: Hierarchical Resolution-Retentive Feature Encoding for Brain Metastasis Segmentation*。[C5]

第一部分 **PHRS：Parallel High-Resolution Residual Stream**。在 nnU-Net 3D 编码主干之外保留一条较窄的全分辨率残差支路（文中通道宽度 16），在每级编码阶段用门控融合。目的在于主干下采样时仍携带小结构的高分辨率信息。

第二部分 **ASCA：Axial Strip Context Aggregator**。在后段编码阶段 3/4 沿各轴聚合上下文，池化到长度 L=4，使用一维 depthwise 卷积、插值回配，配合 SE 门控残差融合。目的是在高分辨率细节之外获得较便宜的长程上下文。

这是可测试的架构假设。它不能“恢复”原始预处理已经丢失的 GT 标签，也不能直接证明我们 P0 的漏灶来源就是下采样。应先做稳定病灶身份、训练 patch 暴露和采样覆盖诊断；之后只改变一个结构变量，不同时换预处理、loss、阈值和采样。

## 5.2 手稿采用的数据与训练

手稿声称 LMD-BM 为 44 个 patient cases、四模态 FLAIR/T1/T1c/T2，35 训练、9 测试，三前景：坏死核、瘤周水肿、强化灶，**没有 RC**。训练为 1000 epochs、batch 2、patch 96×160×160、单 RTX 4090；SGD/Nesterov 0.99、初始学习率 0.01、weight decay 3e-5。

主指标是体素 Dice、HD95，不是官方的 27 mm³ 大灶 DSC/NSD + 小灶 F1。表中 0.5069 不能与我们 Bio 的 WT 病灶 Dice 0.685620 直接相减，也不能据此说模型更差或更好。

## 5.3 主表报告的改进与边界

表 4：nnU-Net 为 Dice 0.4619、HD95 32.5388；HRFuse-Net 为 0.5069、25.4869。Dice 绝对 +0.0450，即 **4.50 个百分点**，相对约 +9.75%；HD95 减少 7.0519，约 21.67%。

但表中 **nnFormer 的 HD95=24.5692，比 HRFuse-Net 更低**。因此“HD95 优于所有基线”与表格不符，应只说相对 nnU-Net 改善。

此外，消融表 1–3 的 full model 是 **0.5101 / 20.2752**，不是主对比表的 **0.5069 / 25.4869**。系统论文表 13 又给两 run 的 LMD-BM 最佳 Dice 51.51%、50.69%；附录逐样本统计出现 51.01%。这些可能涉及不同 run/选点/统计口径，但本次未取得 checkpoint/原始逐例结果来解释，不能自行拼成一个“真实最佳模型”。

手稿小体积组（<0.006cc，即 <6 mm³）声称 Dice 增加 0.126，图示每组仅 n=3。只有 9 个测试病例、且可能含纵向相关，不能把这个微小子组当成强临床小灶证据。

## 5.4 原始数据论文复核发现：40 人，不是手稿的 44 人

31 个 ideas JSON 中的 904 将数据称为 PROTEAS 子集，但 reference/license 字段为 null。为此本次另查原始 Scientific Data 论文：

[Flouri 等，A longitudinal MRI dataset of brain metastases with tumor segmentations, clinical & radiomic data](https://www.nature.com/articles/s41597-025-06131-0)，2025-11-19，DOI 10.1038/s41597-025-06131-0。[C6]

原论文描述 **40 名患者、65 个转移灶、744 MRI 序列扫描（基线与随访）、45 套放疗计划和 45 CT**；六家塞浦路斯诊断中心，2019–2024 收集，有纵向随访。它不是“744 个独立患者”，也不是 BraTS 官方 2026 hidden test。

原论文还写 **小于 3 mm 的病灶被忽略/排除**。直径 3 mm 与体积 27 mm³并非同一阈值；它的微小灶标注协议和我们的 METS 检测目标不能直接视为一致。这也使手稿对极小体积病灶的宣传更需要检查真实病例/标签。

因此，44 是否来自筛选后的时间点、重排后的病例、其他数据源或手稿误写，本次无法确认。手稿还把该数据引用指向 Ramakrishnan 等 2024 的另一篇开放脑转移瘤数据文献，而非 Flouri 等 2025；这进一步说明**数据血缘没有充分闭合**。不能仅凭 904 JSON 宣称下载的数据就是经 SHA 验证的 PROTEAS 子集。

原始数据公开存于论文提供的 Zenodo，本文仅保存公开论文与链接，未下载任何 MRI/患者明细。若未来用于独立外部验证，必须先核验 patient/timepoint 划分、标注与预处理身份、许可、与模型训练病例的交集。

## 5.5 稿件、代码和复现的具体缺口

1. **独立代码/权重入口未可用。** 摘要写 `https://github.com/yifangao112/HRFuse-Net`，本次官方 GitHub API 返回 404。这只证明当时公开入口不可取得，不推断私人仓库不存在；不能称已下载代码或 checkpoint。
2. **作者/期刊身份不足。** PDF 是 Anonymous Author / Anonymous Organization，页眉仍为 Short Title of the Article；“Preprint submitted to Elsevier”是模板文字，不是录用证明。
3. **页码模板错误。** 实际 16 页，末页却标 Page 16 of 15。
4. **引用质量问题。** nnU-Net 在主表引用的是 Niu 等腕部 X 光 nnU-Net 应用，而非原框架；数据引用与原始数据描述不一致。
5. **表图数字未完全一致。** 主比较、消融和部分图中基线均值不同；缺少可绑定模型 SHA 的逐例原始结果，暂不能可靠解释。
6. **效率不能混单位。** 文中 126.7ms 是每 patch，不是全 MRI 病例推理；没有完整全量体积滑窗、显存和后处理的实测验收。

以上不是断言论文造假，而是足够明确的“目前不能拿来直接比较或写成已验证优势”的理由。图表和引用问题也说明 AI 生成稿需要真实证据复核，不能只因为排成了论文就接受全部结论。

# 六、最值得借鉴的路线与后续建议

## 6.1 对本项目的研究优先顺序

| 优先级 | 建议 | 应先满足的条件 |
|---|---|---|
| 1 | 完善检测/分割/RC 阴性误报的分组报告 | 复用原冻结逐例表，支持集和方向明确；不造官方总分 |
| 2 | HRFuse 的高分辨率支路做单变量架构消融 | 获取真实代码/权重或明确自行重实现；先检查 P0 小灶曝光和来源 |
| 3 | SFA/HCF 的轻量多尺度/高频门控 | 单变量、固定 split/seed/预算、显存和病例耗时实测 |
| 4 | PROTEAS 作为纵向外部验证候选 | 真实患者划分、许可、训练暴露、标签兼容；不能比较 RC |
| 5 | Camyla 的研究编排方式 | 把日志、模型身份、失败和原始证据当基础，不以自动写稿代替独立 test |

不建议当前直接安装全套 Camyla、下载 31 个数据集、跑 2700 个候选。更适合先围绕我们已知的小灶召回/RC 误报瓶颈提出一个有明确对照的实验。任何新增下载、环境、训练/评分作业仍须另行按项目授权和计划执行。

## 6.2 想做“同一套权威评价”的最低合同

冻结病例和患者身份；固定 GT 修订版本、MRI 通道和空间几何；严格区分 METS 四区与外部数据三类；固定 evaluator、panoptica、matcher、parser、NSD 容差及 SHA；显式 27 mm³、overlap 0.2 等当前约定；完整报告有效 n、FP/FN、全空/阴性病例；患者组配对 CI；检测和分割分开；独立 test 不反复选模型。

若要真正复现官方最终 ranking，还需取得官方实际 rank 输入矩阵、指标/区域权重、缺失值规则、两榜累计策略、tie 与 permutation 参数。本次报告只确认了公开原则和修订，不能用我们自己写的平均函数替代官方流程。

# 七、资料清单与复核入口

## 7.1 本地目录

本批位于 `02_reference_materials/05_brats_metrics_camyla/20261110/`：

- `REPORT.md / REPORT.docx / REPORT.pdf`：可维护源稿、可编辑版与阅读版。
- `camyla/manuscripts/01.pdf–10.pdf`：10 篇公开原稿，10 为脑转移瘤。
- `camyla/Camyla_system_paper.pdf`、`camyla/PROTEAS_dataset_descriptor.pdf`：系统与原始数据论文。
- `camyla/MANUSCRIPTS.md`：10 篇英文题名、下载链接、优先级和报告值。
- `camyla/PUBLIC_ASSET_MANIFEST.json`：45 项大小/SHA256/Git blob 身份；明确 full_40=false。
- `camyla/searchable_text/`、`PDF_TEXT_INVENTORY.json`：可检索文本与页数；某些 PDF 连字抽取不完整，以原 PDF 为准。
- `brats/FORUM_2025_INDEX.json`、`FORUM_2026_INDEX.json`：完整未删除主题元数据索引。
- `brats/threads/`：20 个主题与 111 条回复，作者 ID/创建/修改时间和原文 SHA 回执；压缩传输原件另保留。
- `brats/wiki/`、`brats/sources/`、`brats/code/`：当日公开规则、API/公开网页、冻结指标源码参考。
- `ARCHIVE_MANIFEST.json`、`EXECUTION_AUDIT.json`：资料身份、执行边界、缺失和检查记录。

第三方全文、PDF、源码参考保留在本机实体目录并登记 SHA；Git 提交只保留原创报告/索引、轻量身份及必要回执，不将本地参考副本误认作项目运行代码。没有下载医学影像、模型、镜像或整份数据集。

## 7.2 主要来源

以下论坛链接直接打开主题；回复编号用于在本地归档中精确定位，日期均来自 Synapse 元数据。

**BraTS 官方与论坛**

- B1：[2025 METS Announcements](https://www.synapse.org/Synapse:syn64153130/discussion/threadId=11896)，回复 34927，2025-07-09，Nikolay：省略病灶级评价；本地 `brats/threads/2025/11896/`。
- B2：[2025 Validation queues open](https://www.synapse.org/Synapse:syn64153130/discussion/threadId=12068)，回复 34682/34721/34800：评分耗时与 NSD 瓶颈。
- B3：[2026 Challenge1 ranked leaderboard](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13508)，官方回复 36364，2026-06-19；非官方排序不代表官方。
- B4：[Computing RC class Scores](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13631)，回复 36397/36399，2026-06-30：按四区 config，设计书文字不对齐。
- B5：[Size categories confusion](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13893)，回复 36460，2026-07-11：27 mm³。
- B6：[2026 METS Announcements](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13146)，回复 36324（6/9）、36449（7/9）：累计秩与准确指标列。
- B7：同 B6，回复 36346/36347（6/16）：FP 惩罚与权重未定；36451/36481 是参与者追问，不是官方结论。
- B8：[Metrics Mismatch/Quirk](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=14024)，回复 36558（7/26），及 B6 回复 36559：最终检测/分割两榜。
- B9：[Validation Set Refinement](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13294)，2026-06-04，官方参考标签修订/重评分。
- B10：[How final submission is selected](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=14124)，回复 36552，2026-07-24：自己选方案并提交容器。
- B11：[官方 BraTS_evaluation](https://github.com/BraTS/BraTS_evaluation)，本次冻结提交 [273f79f](https://github.com/BraTS/BraTS_evaluation/tree/273f79fa3da1681b63e0e0d3c1d14a317293e8e5)；[parser](https://github.com/BraTS/BraTS_evaluation/blob/273f79fa3da1681b63e0e0d3c1d14a317293e8e5/brats_evaluation/metrics_parser.py)、[METS config](https://github.com/BraTS/BraTS_evaluation/blob/273f79fa3da1681b63e0e0d3c1d14a317293e8e5/brats_evaluation/configs/config_mets.yaml)。只做源码研究，不证明与我们旧环境实测等价。
- B12：[PermRanker 公开摘要/出版页](https://link.springer.com/chapter/10.1007/978-3-032-16365-3_1)，2026-04-01；[2026 Evaluation wiki](https://www.synapse.org/Synapse:syn74274097/wiki/639602)、[2025 METS wiki](https://www.synapse.org/Synapse:syn64153130/wiki/631058)。
- B13：[2026 Data wiki](https://www.synapse.org/Synapse:syn74274097/wiki/639600)、[UCSD QC 更新帖](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=14058)。
- B14：[External data/pretrained policy](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13409)，回复 36339/36362，及 B13 当前规则。
- 补充：[2026 Segmentation Evaluation Pipeline 公告](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13311)、[small instance threshold 问答](https://www.synapse.org/Synapse:syn74274097/discussion/threadId=13999)。

**Camyla 与数据原始文献**

- C1：[Camyla 官方仓库](https://github.com/yifangao112/Camyla)，冻结 [df4434f](https://github.com/yifangao112/Camyla/tree/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf)。
- C2：[系统论文](https://arxiv.org/abs/2604.10696)，重点正文 4.1、附录 E/H/I、表 3/13/22/24；[项目页](https://yifangao112.github.io/camyla-page/)。
- C3：[Issue #1 作者确认 10 篇](https://github.com/yifangao112/Camyla/issues/1#issuecomment-4241395950)、[Releases](https://github.com/yifangao112/Camyla/releases)。
- C4：[Issue #3 作者说明 Fold 0/splits_final](https://github.com/yifangao112/Camyla/issues/3#issuecomment-5066148896)。本次未下载数据，未取得这些实际病例 split。
- C5：[HRFuse-Net 原稿永久版 PDF](https://raw.githubusercontent.com/yifangao112/Camyla/df4434f9d4aef5b7394ed03a4e877a8130c1b6cf/assets/paper_pdf/10.pdf)，重点物理第 8–10 页和第 15–16 页；本地 `camyla/manuscripts/10.pdf`。
- C6：[PROTEAS Scientific Data 原论文](https://www.nature.com/articles/s41597-025-06131-0)、[公开 PDF](https://www.nature.com/articles/s41597-025-06131-0.pdf)，DOI 10.1038/s41597-025-06131-0。

**本项目依据**

- L1：[fixed103 四路统一 MET 结果](../../../work_space/S2/03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/results/comparison_four_light_20261108T034412Z/README.md)。说明：本链接以仓库根目录关系维护，阅读正式 DOCX/PDF 时请从总纲的 S2 入口定位；原结果/模型不被本报告覆盖。

## 7.3 仍未取得的材料

剩余 30 篇完整手稿；HRFuse-Net 公共代码与 checkpoint；Camyla 原始逐例结果、病例/患者 split 和全部基线复现包；官方最终两榜累计排名的实际精确参数；每个外部模型与 fixed103 的完整训练患者交集证明。上述缺口不会被“下载 SHA 通过”消除。

研究状态保持 `experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。结论只针对当前可见公开证据，不对官方未公开内容作推断。
