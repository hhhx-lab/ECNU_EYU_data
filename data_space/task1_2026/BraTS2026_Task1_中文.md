# BraTS 2026 Challenge - Task 1 中文整理

来源页面：https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097
访问日期：2026-05-04
范围：整理 Task 1 页面中的任务、数据、数据文件、评估、提交与结果信息。本文件是完整结构化整理与准确中文翻译，不是第三方网页的逐字镜像复制。

## 页面背景

挑战名称：BraTS 2026 Challenge
状态：进行中
页面显示的注册参与者数量：52
页面可见挑战标签页：Overview、Instructions、Task 1、Task 2、Task 3、Task 4、Task 5、News、Community
Task 1 页面的目录项：Description、Data、Data Files、Evaluation、Submission、Results

页面页眉说明：Brain Tumor Segmentation Challenge，即 BraTS，自 MICCAI 2012 发起以来，通过算法评测、提供高质量标注数据集，以及设置贯穿疾病过程的临床相关任务，持续推动脑肿瘤影像分析。MICCAI 2026 的第 15 届 BraTS 挑战集群与 AI-RANO、RSNA、ASNR、NIH、ASFNR、CBTN 等临床组织合作，继续推进这项工作。

## 任务描述

### 临床问题

Task 1 聚焦脑转移瘤疾病监测。页面指出了三个主要问题：

1. 脑转移瘤监测费时费力，尤其是在需要管理多个脑转移灶、且工作流程依赖人工测量或人工标注时。
2. 根据 RANO-BM 指南，脑转移瘤通常通过最大单维直径进行评估。页面强调，病灶及其周围水肿的体积估计对于有效临床决策以及改进治疗结局预测非常重要。
3. 脑转移瘤常常很小。检测和分割小于 10 mm 的病灶具有较高难度，以往相关任务的 Dice 相似系数较低。

### 拟议解决方案

拟议方案是使用机器学习方法自动检测并分割：

- 脑转移瘤；
- 病灶周围水肿；
- 切除腔。

预期收益包括提升时间效率、提高可重复性，并增强算法对不同人工标注者之间差异的鲁棒性。

### 预期影响

该挑战旨在产生可用于当前治疗场景和治疗后场景的关键算法。页面所述目标是改善，并可能变革，脑转移瘤患者的管理与监测。

## Task 1 任务内容

参赛者需要开发一种通用的自动分割算法，能够检测并精确勾画不同大小的脑转移瘤。该算法应适用于治疗前病例和治疗后病例。

### 四标签标注系统

BraTS 2026 Brain Metastases 使用以下四标签系统：

| 标签 | 缩写 | 名称 | 含义 |
|---:|---|---|---|
| 1 | NETC | 非强化肿瘤核心，Nonenhancing tumor core | 被强化肿瘤包绕、但自身无对比增强的所有肿瘤核心部分。它代表通常被认为适合手术切除的肿瘤主体。 |
| 2 | SNFH | 周围非强化 FLAIR 高信号，Surrounding non-enhancing FLAIR hyperintensity | 肿瘤周围水肿和浸润组织，由 T2 FLAIR 图像上异常高信号包络定义。它包括浸润性非强化肿瘤，也包括肿瘤周围区域的血管源性水肿。既往梗死或微血管缺血性白质改变等与肿瘤无关的 FLAIR 信号异常不包含在内。 |
| 3 | ET | 强化肿瘤，Enhancing Tumor | 在增强后 T1 加权图像上具有明显对比增强的所有肿瘤部分。邻近血管、出血或内源性 T1 高信号不包含在该标签中。 |
| 4 | RC | 切除腔，Resection Cavity | 治疗后病例中脑内的切除区域。 |

2026 年任务将新增一个检测排行榜。其目的在于促进对病灶检测敏感的算法。页面特别指出，小于 27 mm^3 的小病灶具有临床相关性，因为它们可能需要被计为独立实体，或需要单独定量。

## Task 1 组织者

| 姓名 | 角色 | 单位 |
|---|---|---|
| Mariam Aboian, MD/PhD | 首席联合组织者 | Department of Radiology, Children's Hospital of Philadelphia |
| Nikolay Yordanov, MD | 联合组织者 | Faculty of Medicine, Medical University - Sofia, Sofia, Bulgaria |
| Nazanin Maleki, MD | 联合组织者 | Department of Radiology, Children’s Hospital of Philadelphia (CHOP) |
| Raisa Amiruddin, MBBS | 联合组织者 | Department of Radiology, Children’s Hospital of Philadelphia (CHOP) |
| Fabian Umeh | 联合组织者 | Teesside University, UK |
| Crystal Chukwurah | 联合组织者 | Medical Student, Yale School of Medicine |
| Monika Pytlarz | 联合组织者 | PhD Student, Sano - Centre for Computational Personalised Medicine |

## 数据

BraTS 2026 Brain Metastases 数据集是来自多个机构的治疗前和治疗后脑转移瘤多参数 MRI，mpMRI，扫描的回顾性汇编。这些扫描是在标准临床条件下获得的。由于数据来自不同机构、设备和成像协议，因此该数据集反映了广泛的图像质量范围和真实世界中多样化的临床实践。

本次挑战使用 2025 年的数据集。页面说明，训练集和测试集中的所有图像均已标注。2026 年，组织方计划对数据集内所有标注进行质量控制。此外，组织方还计划向参赛者提供未标注病例，以促进半监督学习方法创新。

### MRI 序列

该数据集包含以下多参数 MRI 序列：

- 增强前 T1 加权成像，T1W；
- 增强后 T1 加权成像，T1C；
- T2 加权成像，T2W；
- T2 加权液体衰减反转恢复成像，FLAIR。

2025 年，T2W 在 BraTS-METS 中变为非强制序列。有些病例有原生 T2，有些病例有合成 T2，也有些病例没有 T2 序列。所有成像体数据均使用多个脑转移瘤分割算法的 STAPLE 融合结果进行初始分割。融合标签随后由不同职级和经验水平的神经放射学专家按照统一沟通的标注协议进行人工精修。经验丰富且获得专科认证的主治级神经放射科医师批准了这些精修后的标注。

### 挑战包含的数据集

| 数据集 | 训练 | 验证 | 测试 | 配准空间 | 贡献者 | 机构 | 已标注 |
|---|---:|---:|---:|---|---|---|---|
| Duke | 37 | 15 | 30 | SRI24 space | Devon Godfrey PhD; Scott Floyd MD/PhD | Duke University | yes |
| NCI | 35 | n/a | 1 | SRI24 space | Ayda Youssef MD | National Cancer Institute | yes |
| Missouri | 22 | 25 | 35 | SRI24 space | Nourel hoda Tahon MD, Msc; Ayman Nada MD/PhD | University of Missouri | yes |
| WashU | 39 | 2 | 12 | SRI24 space | Satrajit Chakrabarty | Washington University | yes |
| Yale | 195 | n/a | 12 | SRI24 space | Mariam Aboian MD/PhD | Yale university | yes |
| UCSF | 322 | n/a | n/a | Native space | Jeffrey Rudie MD | University of California, San Francisco | yes |
| NW | n/a | 46 | n/a | SRI24 space | Yuri S. Velichko PhD | Northwestern University | yes |
| UCSD | 646 | 91 | 213 | Native space | Maria Correia de Verdier MD; Jeffrey Rudie MD/PhD | University of California, San Diego | yes |
| Ulm | 200 | 0 | 0 | Native space | Nico Sollman MD/PhD | Ulm University, Germany | no |
| In total | 1496 | 179 | 303 | n/a | n/a | n/a | yes |

### 附加数据说明：UCSD 纵向 MRI 数据集

University of California San Diego Brain Metastases Longitudinal MRI Dataset 目前包含 646 个训练病例。该数据集包含进展性纵向数据。有些病例可能接受过非手术治疗，因此部分病例可能存在空掩膜。

在发布该数据集前，组织方将进行质量控制。他们会在训练数据上运行一个 BraTS 2025 获胜者算法，以及一个从未在 BraTS 数据上训练过的算法，页面引用为 Rudie et al., 2021。Dice 小于 1 的病例将被识别出来并重新标注。

### 图像配准

BraTS 2025 Metastases 数据集包含以下混合类型：

- 原生空间病例；
- 与 T1C 1 mm^3 共配准的病例；
- 配准到 SRI24 空间的病例。

Ulm University、UCSF 和 UCSD 提供的所有病例均处于原生空间，总计 1268 例。其余病例配准到 SRI24 空间，总计 328 例。

页面解释，将神经影像病例配准到 SRI24 这样的公共空间，可以提供一致的解剖参考，从而便于跨受试者、研究和数据集进行比较。然而，对放射科医师来说，在原生空间阅片更加自然，因为插值可能扭曲图像并遮蔽小病灶。

### 数据访问

除注册挑战外，参赛者还必须申请数据访问权限。

1. 提交 Data Access Google form：https://forms.gle/UiCpXos2zKFPdMnK6
2. 五个挑战任务及其训练集和验证集只需要提交一次数据访问表单。
3. 详细信息验证后，BraTS Service Account 会通过邮件邀请申请者加入 BraTS 2026 Data Access Team。
4. 参赛者需要接受邀请，才能解锁页面列出的文件。

BraTS 2026 Data Access Team：https://www.synapse.org/Team:3586605

注意：测试数据集和验证集真实标签不会向公众发布。

## 数据文件

| 文件名 | Synapse ID | 修改时间 | 大小 | MD5 |
|---|---|---|---:|---|
| MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1.zip | syn64919665 | 4/29/2026 3:48 AM | 31.17 GB | a67d67f756c8ef14e8dbda08ca73688c |
| MICCAI-LH-BraTS2025-MET-Challenge-ValidationData_batch1.zip | syn64919141 | 4/29/2026 3:48 AM | 5.06 GB | ab2e253de48f11e8e8c4da3dc96ba113 |
| MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1.zip | syn65888166 | 4/29/2026 3:48 AM | 20.25 KB | dbad47aca8d27bff93d3f6436b6f2cfa |

页面还显示了“Add To Download List”操作。

## 评估

### 分割评估指标

该任务使用按受试者计算的分割指标：

1. Dice Similarity Coefficient，DSC，Dice 相似系数，是常用的分割性能评估指标。
2. Normalized Surface Distance，NSD，归一化表面距离，它引入容差参数，是 DSC 等传统指标的互补指标。

### 病灶检测评估指标

按病灶计算的检测指标包括：

1. F1 score，即精确率和召回率的调和平均数，用于判断算法是否倾向于过分割或欠分割。
2. 在不同检测阈值下得到多个 F1 分数，并基于这些 F1 分数计算 AUC。

检测评估会应用于一个 MRI 研究中的每一个单独病灶。对于挑战的检测分支，页面将病灶定义为一个集合性术语，包含强化肿瘤、非强化肿瘤核心和切除腔。

分割评估指标只应用于大于 275 mm^3 的病灶。页面说明，该数据集中的图像已共配准到 1 mm 层厚。

### 排名细节

排名将遵循基于 DELPHI 的图像分析验证建议。该流程包含：

1. 算法排名；
2. 统计显著性检验。

对于多维结果或多指标结果，每个团队会根据上述指标平均值获得相应排名。随后对这些排名求和，形成一个单变量总体汇总度量，该度量决定每个团队的总体排名。

所有团队会按排名顺序排列。随后，以成对方式对平均排名进行随机置换，共进行 500,000 次置换。对应的成对 p 值将被计算出来，用于判断成对统计显著性，并报告有序方法之间的实际差异。

p 值会以一个上三角矩阵报告。该矩阵将显示统计上不显著的潜在团队分组，这些团队会被归入同一层级；同时也会明确标出其他团队之间显著优越的关系。页面将该方法描述为以往 BraTS 和其他挑战中使用的系统排名方法的演进版本，并表示该方法将被打包和发布为独立工具，以支持可重复性并供其他挑战使用。

如果某个算法未能为某个特定测试病例产生结果指标，页面说明不会施加惩罚。也就是说，该指标不会被设为其最差可能值，例如 DSC 或 NSD 不会被设为 0。

### 评估参考文献

1. Reinke et al. Understanding metric-related pitfalls in image analysis validation. Nature Methods. 2024 Feb;21(2):182-194.
2. Maier-Hein et al. Metrics reloaded: recommendations for image analysis validation. Nature Methods. 2024 Feb;21(2):195-212.

## 提交

页面包含 Submission Dashboard 和 Your Submission Directory 区域。

Project SynID：syn74773222

提交文件表包含以下列：

- File Name
- Updated On
- ID

访问时，该表没有任何行，并显示 0-0 of 0。

页面显示的提交操作包括：

- Upload File
- Submit Selection

## 结果

Results 区域显示：Coming soon。

## 页脚与相关链接

- Terms of Service：https://www.synapse.org/TrustCenter:TermsOfService
- About：https://sagebionetworks.org/
- Help：https://help.synapse.org/docs/Getting-Started.2055471150.html
- Version Number / 源代码仓库链接：https://github.com/Sage-Bionetworks/synapse-web-monorepo
