# 来源、检索路径与完整论文索引

调研日2026-10-27。URL/UTC/原始响应/SHA见证据JSON；本表便于人工阅读。只使用公开访问，不绕过登录、验证码、访问控制或限流。

## 检索路径

1. 用户四种子：BraTS Orchestrator、MICCAI2026 Satellite METS目录、Graylight2026博客、Ujjwal Baid GitHub。
2. 官方Task1 `syn74274097`核赛道、模态、数据规模、标签、27mm³、排名规则；其公开validation未排名，不作为最终榜。
3. Synapse官方公开API按截图名/空格/点号变体查询teams，再读指定teamMembers，逐作者交叉；aymuos继续核public userProfile，确认是个人而非误填team ID。实际API见TEAM_EVIDENCE。
4. 32篇METS公开PDF全文抽取，追作者/机构/code availability；FLAIR跨分类补PEDs014。PDF大小/SHA和读取UTC见PAPER_EVIDENCE。
5. 论文代码链→GitHub元数据/tree/README→重点配置/训练与推理→releases→HF model card/LICENSE/LFS文件元数据。Bitbucket另查公开目录和ensemble代码，不执行。
6. Google学术查询`"BraTS" "2026" "metastases"`首屏；arXiv API查询`all:brats AND all:2026 AND (all:mets OR all:metastases)`返回7项，精确标题OR查询返回8项但未新增本次目标论文。搜索命中仅作线索，不把无关论文收为参赛队。
7. HF `BraTS`/`brats2026`/`brats26`/`Metastases`搜索；只看前100条时不能声称全平台穷尽。确切模型URL由作者repo指向，搜索名未命中不能抹掉真实权重。
8. Synapse Docker三个manifest匿名401，停止，不获取凭据或镜像层。Zenodo搜索403后停止；OpenReview403/网络失败转用会议同文公开PDF。

团队别名、作者、机构与方法词存于[TEAM_MAP.json](TEAM_MAP.json)，不是猜测来源。搜索可组合：`"BraTS 2026" + 别名/作者`、`"BraTS-METS" + 方法名`，再限定GitHub、arXiv、OpenReview、MICCAI、Hugging Face、Bitbucket或机构主页；检索建议不冒称每种组合都已逐一发起。

## 论文目录

下面表格由本次MICCAI相关分类链接机械整理。编号不连续是目录本身如此，不擅补003/012/024/032；本次32篇目录项全文均已读取。PEDs014另列，不能算32篇内。

| 编号 | 技术论文原题与原始页面 |
|---|---|
| METS001 | [Clinically Informed, Dataset-Adaptive Lightweight Segmentation of Brain Metastases within the nnU-Net Framework](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_001.html) |
| METS002 | [A Multi-Network Ensemble Strategy for the BraTS 2026 Brain Metastasis Segmentation Challenge](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_002.html) |
| METS004 | [Lesion-Aware Hybrid State-Space and Residual-Encoder Fusion for Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_004.html) |
| METS005 | [Integrating nnDetection and nnU-Net for Brain Metastasis Segmentation in the BraTS-METS 2026 Challenge](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_005.html) |
| METS006 | [Rare-Region Optimization and Ensemble Composition for Pre- and Post-Treatment Brain Metastases Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_006.html) |
| METS007 | [Runtime-Aware Ensemble Optimization for Brain Metastasis Segmentation under Strict Deployment Constraints](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_007.html) |
| METS008 | [Region-Aware Residual Encoder nnU-Net for Multi-Region Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_008.html) |
| METS009 | [Pre- and Post-Treatment Brain Metastases Segmentation Using nnU-Net with Post-Processing for BraTS 2026](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_009.html) |
| METS010 | [Beyond Aggregate Dice: Size-Stratified Failure Analysis of an nnU-Net Baseline for Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_010.html) |
| METS011 | [Gated Frequency-Enhanced Class-Wise Ensemble for Brain Metastases Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_011.html) |
| METS013 | [Brain Metastasis Segmentation Using an Ensemble of U-Mamba Models](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_013.html) |
| METS014 | [A Region-Based Residual-Encoder nnU-Net Ensemble with a Matched-Normalization Subtraction Channel and Resection-Cavity False-Positive Suppression for Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_014.html) |
| METS015 | [A Transformer Pipeline with Uncertainty-Guided Curation and Failure Recovery for Brain Metastases Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_015.html) |
| METS016 | [Ensemble nnU-Nets for pre- and post-operative Brain Metastases Segmentation in BraTS 2026 Task 1](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_016.html) |
| METS017 | [Region-Wise Logit Fusion and Utility-Gated Refinement for Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_017.html) |
| METS018 | [Scale-Aware 3D Deep Learning for Robust Brain Metastasis Detection in Multimodal MRI](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_018.html) |
| METS019 | [Resection-Cavity-Aware Segmentation of Brain Metastases for BraTS 2026](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_019.html) |
| METS020 | [Segmentation of Pre- and Post-treatment Brain Metastases Using an Ensemble of MRI Foundation Models](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_020.html) |
| METS021 | [CarveMix-RC: Addressing Rare-Class Imbalance Through Lesion-Aware Synthetic Augmentation for Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_021.html) |
| METS022 | [Detege et Delinea: A Two-Stage Framework for Brain Metastasis Detection and Segmentation in Multimodal MRI](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_022.html) |
| METS023 | [Region-Aligned Training and Target-Specific 2D–3D nnU-Net Fusion for BraTS 2026 Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_023.html) |
| METS025 | [A Residual-Encoder nnU-Net Baseline for Pre- and Post-Treatment Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_025.html) |
| METS026 | [Seeing Lesions, Not Voxels: A Connected-Component-Weighted Dual-Map Focal Loss and Per-Group Oversampling for Brain Metastases Segmentation and Detection](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_026.html) |
| METS027 | [Brain Metastases Segmentation for BraTS 2026 Task 1: A Multi-Architecture Comparison](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_027.html) |
| METS028 | [Missing Modality Imputation and Brain Metastasis Segmentation via Latent-Space Ensemble Synthesis and Size-Aware Focal Loss](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_028.html) |
| METS029 | [Precision Over Recall: Brain Metastasis Segmentation with a Learned False-Positive Rejector](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_029.html) |
| METS030 | [ACORN: Asymmetric Conditional Overlapping RAVSS Network for Segmenting Small Brain Metastases](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_030.html) |
| METS031 | [Epistemic Uncertainty-Guided Supervision Weighting for Brain Metastasis Sub-region Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_031.html) |
| METS033 | [METS-PRO R2: Lesion-Aware Positive Residual Refinement for Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_033.html) |
| METS034 | [Segmentation of Pre- and Post-treatment Brain Metastases Using Ensemble Model](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_034.html) |
| METS035 | [Inference-Time Decoding for Brain Metastasis Segmentation](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_035.html) |
| METS036 | [A Detection-Aware SegResNetDS Ensemble and a Measured Null Space for Brain Metastasis Segmentation in BraTS 2026](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_036.html) |
| PEDs014 | [Mechanism-Aware Post-Processing Under a Submission Budget: FLAIR-Viaevum at the BraTS 2026 METS, PEDs and GoAT Challenges](https://papers.miccai.org/miccai-2026-sat/BraTS_PEDs_014.html) |

## 已核arXiv版本

- [2609.11477](https://arxiv.org/abs/2609.11477)：bin_JLU / Haobin Liu、Xin Wang，对应009。
- [2609.24769](https://arxiv.org/abs/2609.24769)：IUB / Mahdi Islam、Musarrat Tabassum，对应027。
- [2609.35195](https://arxiv.org/abs/2609.35195)：Vision_Lab / CarveMix-RC，对应021。
- [2604.24276](https://arxiv.org/abs/2604.24276)：Kundu等《Instance Awareness of Multi-class Semantic Segmentation Loss Functions》，相关基础研究，**不是035同文，也不自动等于其2026提交**。

其余3项宽查询结果是报告生成、MRI合成/超分等相关但非本次前列队方案；精确标题查询8项均未额外确认目标同文。未匹配不等于从未另发论文。

## 额外方法线索

不绑定用户截图的队伍名，避免强行填空；这些来自公开PDF代码链接，除标明已读tree者外仅定位线索。

- 001：[Williamwhy/brats-mets-2026](https://github.com/Williamwhy/brats-mets-2026)，轻量nnU-Net；Free State/SPARK等。
- 004：[METS-HybridFusion](https://github.com/takosainsbury59-hash/METS-HybridFusion-BraTS2026)，state-space+ResEnc，tree已读；`LICENSE_PENDING`不等于开放复用许可，公布SHA不等于权重下载。
- 006：[brats2026-met-uniandes](https://github.com/afromerogr/brats2026-met-uniandes)，Universidad de los Andes，RC rare-region优化。
- 010：[NeuroSeg HFA](https://github.com/moisescunha77-arch/neuroseg-hfa-brats2026)，Brazil HFA，size-stratified failure analysis。
- 016：[Humanitas v18](https://github.com/allindemann/brats-2026-mets-task1-v18)，递归tree已取得，未重点审阅完整训练/许可；容器名在公开资料中可见，匿名manifest401。
- 018：[BraTSDetector](https://github.com/ailabrepo/BraTSDetector)，MIT/Harvard/Boston Children's，scale-aware多FOV检测。
- 019：[brats26-mets](https://github.com/kagozi/brats26-mets)，University of South Dakota，MedNeXt/RC专家。
- 022：[minnelab/BraTS-2026](https://github.com/minnelab/BraTS-2026)，Linköping/KTH/Karolinska，两阶段检测/分割。
- 025：[BraTS2026_METS](https://github.com/ujjwalbaid0408/BraTS2026_METS)，用户种子baseline，tree/release已读；不要挪用其2025GLI历史分数。
- 030：[91XG40/BraTs2026](https://github.com/91XG40/BraTs2026)，National Central University，ACORN。
- 033：[METS-PRO-R2](https://github.com/ThanSin02426/METS-PRO-R2)，IIT Jammu/DiceMed，ET非负残差refinement。
- 036：[SegResNetDS ensemble](https://github.com/MahmoudGhoraba/brain-tumor-segmentation-brats)，tree已读；GIU/Arkansas/Mississippi，MONAI Auto3DSeg，论文称4×H200，非我们的七月服务器。

## 新闻、基础设施与旧版线索

- [MicroBT作者新闻](https://franciszchen.github.io/)：2026年10月第一名声明，已读取；自家新闻为局部佐证，非完整官方最终表。
- [Graylight2026](https://graylight-imaging.com/blog/brain-metastasis-segmentation-using-ai-graylight-imaging-presents-new-results-at-miccai-2026/)：作者/方法/2检测4分割声明，与013及成员交叉。
- [FLAIR作者主页](https://ai4viaevum.com/team/long-lai/)：当前个人介绍与历史论文单位分开，本次归属取论文/Synapse/README。
- [Orchestrator算法表](https://brats.readthedocs.io/en/latest/reference/algorithms/) / [backends](https://brats.readthedocs.io/en/latest/guides/backends/)：可找历史容器入口，METS当前登记2025与2023，不是2026冠军代码发布证明。
- [AURORA](https://github.com/BrainLesion/AURORA)、[2510.25058](https://arxiv.org/abs/2510.25058)、[Zenodo15927391](https://zenodo.org/records/15927391)：Orchestrator关联历史/基础设施线索；本次未验证该Zenodo文件或新2026队模型内容，不纳入已核权重列表。
- [Angelus README所列Zenodo DOI](https://doi.org/10.5281/ZENODO.19714728)：仅从已读取README定位；访问受限，未核记录与当前论文/模型的对应关系，不纳入已核权重列表。

## 访问不完整

Google普通搜索部分ERR_ABORTED/CONNECTION_CLOSED，Bing有无关音乐结果，不作为证据；Google学术首屏成功。OpenReview不能完整读，Zenodo403停止；GitHub REST额度耗尽导致Pixel_Perfect只读成功的raw README，不能断言其余资产不存在。精确失败/响应见SEARCH/CONTAINER/其他证据，未规避限制。

本次不接入ECNU、不访问比赛hidden-test/validation GT、不拉模型/镜像，不安装或运行未知代码。公开文本/元数据可复核≠科学质量/复现pass。
