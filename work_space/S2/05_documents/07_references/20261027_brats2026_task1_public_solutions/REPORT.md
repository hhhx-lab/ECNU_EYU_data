# BraTS 2026 Task 1 前列队伍公开方案深度调研

调研日期：2026-10-27，Asia/Shanghai。各来源实际读取UTC保存在同目录证据JSON中。

配套入口：[21队论文/PDF/代码/模型直达表](LINK_INDEX.md)；[33篇完整论文标题与额外方法](SOURCES.md)。本文第4节详解10条重点路线，第5节逐一补齐剩余11队，第8节区分对本项目有用的研究问题和未经验证的建议。

## 1. 主要结论

**已经找到足够多的一手资料，可以基于方法与代码开展研究，而不必只凭排行榜猜。** 截图21个名字均形成身份到论文的公开证据链，其中aymuos是个人用户名，其他20个是公开队伍。MICCAI2026 Satellite的METS分类32篇论文全部读取正文；另补到1篇在PEDs分类、实际同时介绍METS的FLAIR-Viaevum论文。

六条最重要线索：

1. **MicroBT / COMPASS-METS**：冠军有作者主页佐证。源码、训练、OOF学习门控、最终配置和冻结资产哈希清单公开；**冻结网络权重和学习模型没有下载URL**，不能立即复现冠军输出。
2. **buckeye / Ohio State**：训练与学习式ET假阳性拒绝器公开，小型拒绝器二进制确在GitHub，但五个分割checkpoint不在仓库。
3. **DEEPNEURO-MET / DEEPNOID**：nnU-Net+nnDetection、RC专门处理、容器配方公开；不是名字相似的DeepNeuro工具项目。
4. **MIA_Lab / Bern**：Primus transformer、实例均衡目标、小灶专家、失败恢复及独立人工标签QC工具公开；未找到直接公开权重。
5. **TheRealBoys**：三套HF权重均public/non-gated，有真实文件、大小和LFS SHA256，合计约**4.215 GiB**。GitHub只有极简推理入口，不是完整训练开源。
6. **Bio-Intel**：另有约**2.303 GiB**的公开模型包，模型卡称含7个fold。源码Apache-2.0，但BrainIAC衍生权重限**非商业学术研究/教学**，推理仍需另行取得基础BrainIAC checkpoint。

总体趋势是nnU-Net/residual encoder主干加上区域融合、小灶候选、假阳性控制、RC稀疏性和推理预算；并非所有队伍都靠更大网络。不能从这批文献推导“扩散增强无效”或“某架构必胜”。

## 2. 比赛与名次边界

[官方Task1](https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097) 是pre-/post-treatment脑转移瘤METS，不是GLI/PEDs/GoAT/Missing MRI。互斥标签NETC=1、SNFH=2、ET=3、RC=4；TC=NETC∪ET、WT=NETC∪SNFH∪ET，RC独立。官方表列训练1296、validation179、test303；T2W可真实、合成或缺失。

官方说明对>27mm³病灶采用分割指标，另强调<27mm³检测。名次用多指标排名聚合、500000次置换及统计不显著tier分组，**并列不是独占第二/第三名**。他队论文/本地实现中50voxels、275mm³或不同连通性不能替代当前官方合同。

本报告“分割/检测/综合”转录用户截图。公开门户本次仍是**Validation Leaderboard (unranked)**，1004条提交，不是完整最终tier表。[Zhen Chen主页](https://franciszchen.github.io/) 2026年10月新闻称MicroBT获BraTS-METS第一；Graylight博客称其在35队中检测第二/分割第四，提供局部佐证，**不构成对整张最终榜的独立验真**。

核验链：截图原名→Synapse公开队伍/用户→成员→论文作者/机构→code availability→README/文件树→权重/镜像实际状态。身份“强”是作者/成员交叉或直接队名声明；公开源码不等于许可齐全、权重齐全、已运行或复现原输出。下文成绩/算力/包内容均为作者报告，非本项目实测。

## 3. 21队完整索引

论文编号链接到MICCAI原始页面；PEDs014是跨赛道论文。机构按论文署名，不猜作者籍贯。

| 截图名 | 分割/检测/综合 | 机构与作者锚点 | 论文 | 代码/方法入口 |
|---|---|---|---|---|
| MicroBT | 1/1/1 | 香港理工DSAI、Nottingham、Lincoln；Zhenye Lou/Qing Xu/Wenting Duan/Zhen Chen | [017](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_017.html) | [COMPASS-METS](https://github.com/leonlou921/COMPASS-METS) |
| buckeye | 2/1/2 | Ohio State/OSU癌症中心；Hari Girish、Simeng Zhu等 | [029](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_029.html) | [FP Rejector](https://github.com/Hari-Girish/-Precision-Over-Recall-Brain-Metastasis-Segmentation-with-a-Learned-False-Positive-Rejector) |
| DEEPNEURO-MET | 2/1/2 | DEEPNOID Neuro AI，首尔；Jun-Ho Kim/Ki Bok Nam | [005](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_005.html) | [DEEP-MET](https://github.com/deepnoid-ai/DEEP-MET) |
| Mr_Cancer_Defeater | 2/1/2 | University of South Carolina；Roger Newman-Norlund/Pranesh Rajesh Kannan/Raghav Pallapothu | [014](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_014.html) | [CPU pipeline](https://github.com/rnorlund/bratsmets-cpu-pipeline) |
| aymuos | 2/2/3 | King's College London/King's College Hospital；Soumya Kundu | [035](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_035.html) | [brats2026-mets](https://github.com/aymuos15/brats2026-mets) |
| MIA_Lab | 3/2/4 | Bern ARTORG/Inselspital；Francesco Nitti等 | [015](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_015.html) | [分割](https://github.com/nittifra/brats-met-2026) / [QC](https://github.com/nittifra/brats-met-qc) |
| bin_JLU | 3/3/5 | 吉林大学软件学院；Haobin Liu/Xin Wang | [009](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_009.html) | [brats2026-met](https://github.com/hornbeamliu/brats2026-met) |
| GLI2026 | 4/2/5 | Graylight/Silesian University of Technology，Gliwice；Piotr Rudzki/Maria Bancerek/Jakub Nalepa | [013](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_013.html) | [Bitbucket ensemble](https://bitbucket.org/fphealthcare/brats2026-ensemble) |
| Vision_Lab | 3/3/5 | Old Dominion University Vision Lab；Md Shibly Sadique等 | [021](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_021.html) | [CarveMix-RC](https://github.com/shiblyg/carvemix-rc) |
| Bio-Intel_Team | 4/3/6 | SNU Hospital/SNU、Chungnam；Doye Kim/Yeon Gyu Han/Dongheon Lee | [020](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_020.html) | [Bi-Intel](https://github.com/thekeramik05/MICCAI-METS_challenge_bi-intel) |
| FLAIR-Viaevum | 4/3/6 | 中山大学肿瘤防治中心神经外科；Long Lai/Minli Hu等、Chao Ke | [PEDs014](https://papers.miccai.org/miccai-2026-sat/BraTS_PEDs_014.html) | [FLAIR/NeuroForge](https://github.com/B664577/brats2026-flair-viaevum) |
| WIN_BRATS | 5/2/6 | Mansoura University，埃及；Abdelrahman El-Gaghous等 | [007](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_007.html) | [winbrats-docker](https://github.com/Abdelrhman-Elgaous/winbrats-docker) |
| Core_Neuro_AI | 5/4/7 | University of Technology Sydney；Sri Sai Hemanth Bollepalli | [008](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_008.html) | [brats2026](https://github.com/sshemanth/brats2026) |
| Latent_Minds | 6/4/8 | IIT Bombay；Vineet Jangir/Bhavesh Sandbhor | [026](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_026.html) | [BraTS2026-MET](https://github.com/vineetjangiriitb/BraTS2026-MET) |
| Pixel_Perfect | 6/4/8 | IIT Delhi/AIIMS New Delhi；Abdul Rahman/Abhishek Raj等 | [034](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_034.html) | [BraTS-MET](https://github.com/Rossette-46/BraTS-MET)，本次仅README核验 |
| Brain_exe | 7/4/9 | Semmelweis、Obafemi Awolowo、Shenyang Medical、HIT Shenzhen；Amirmahdi Mojtahedzadeh等 | [023](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_023.html) | [Brain.exe](https://github.com/BrainExeBrats2026/brain-exe-brats2026-task1) |
| IUB | 5/6/9 | Independent University, Bangladesh；共同作者为独立研究者；Mahdi Islam/Musarrat Tabassum | [027](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_027.html) | [BraTS_METS_2026](https://github.com/mahdiislam79/BraTS_METS_2026) |
| TheRealBoys | 5/6/9 | San Raffaele/Vita-Salute、Politecnico di Milano；Nicolò Pecco/Lorenzo Veronese | [002](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_002.html) | [Segmentation_METs_BraTS2026](https://github.com/NicoloPecco/Segmentation_METs_BraTS2026) |
| Wangwangwang | 6/5/9 | 山东科技大学；Ruifeng Wang/Yanjun Peng/Xue Chen | [011](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_011.html) | [GFCE](https://github.com/qqqwwweressaa/GFCE) |
| Angelus | 7/5/10 | 台湾National Health Research Institutes；Ajesh Saviour Paravila | [031](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_031.html) | [BraTS-2026-metastasis](https://github.com/zephyr-9598/BraTS-2026-metastasis) |
| ECNU_NYU_team | 7/5/10 | 华东师大统计学院/NYU生物统计；Jianbao He/Zhuoran Sun等 | [028](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_028.html) | [ECNU_NYU_brats](https://github.com/hhhx-lab/ECNU_NYU_brats)，本文只核论文公开链接 |

### 身份核对中最重要的三处纠错

- Mr的真实公开名是`Mr. Cancer Defeater`，ID[3600885](https://www.synapse.org/Team:3600885)，三成员恰是论文014作者；下划线匹配会漏。
- [userProfile/3495063](https://repo-prod.prod.sagebase.org/repo/v1/userProfile/3495063) 明确返回aymuos、Soumya Kundu、KCL。仓库称“team3495063”，实际为用户ID；其创建的队伍是[CAI4CAITumours3495064](https://www.synapse.org/Team:3495064)。找不到队伍3495063不代表队伍撤回。
- FLAIR的METS方法在PEDs014跨赛道论文；PEDs/GoAT获奖不能挪给METS。当前个人主页和历史论文单位可能不同，本文机构采用论文与队伍当次声明。

## 4. 重点方案拆解

### 4.1 MicroBT：structured anchor与有约束的ET恢复

[论文017](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_017.html) / [COMPASS-METS](https://github.com/leonlou921/COMPASS-METS)：三个来源各五折，共15网络：ResEncXL、ResEncM，以及对应M fold初始化的lesion-sensitive focal/Tversky fine-tuning。

先对齐WT/TC/ET/RC概率；XL与FT做**logit融合**，TC/WT从FT直接解码，ET/RC经融合概率/组件条件，形成保守几何anchor。ET候选以组件分类器加至少2/3来源对ET及父TC/WT的支持决定，仅增加、不删anchor。最后Utility V4分别预测病灶存在与几何安全，70固定特征，accept0.75/reject0.50/其余abstain，只处理断开的ET additions。

已读 [xf12.json](https://github.com/leonlou921/COMPASS-METS/blob/main/configs/fusion/xf12.json)：XL/FT权重WT0.85/0.15、TC0.80/0.20、ET0.85/0.15、RC0.70/0.30。[final配置](https://github.com/leonlou921/COMPASS-METS/blob/main/configs/final/n03_utility_v4.json) 明确add regions仅ET、delete为空、preserve_anchor=true。`calibrated`命名不是独立概率校准，论文明确logit fusion。

作者报告复用179开发集宏平均LW-DSC0.712/NSD0.714/AllF1 0.779/SmallF1 0.354；XL的SmallF1为0.367，**不是全指标上升**。分析训练examination1295而官方1296，不能自动解释成和我们同样排除。

阅读链：`data/prepare_dataset501.py`→`scripts/03–07*`→`fusion/*`→`postprocessing/final.py`/`utility_v4.py`→输出合同，配合特征词典。公开tests未本次运行。

[ARTIFACTS.md](https://github.com/leonlou921/COMPASS-METS/blob/main/docs/ARTIFACTS.md) 外置15推理裁剪checkpoint、5学习bundle、schema；有角色/大小/hashes，**没有URL且不承诺下载发布**，releases为空。源码重训≠比赛冻结输出复现。

[COMPUTE.md](https://github.com/leonlou921/COMPASS-METS/blob/main/docs/COMPUTE.md) 据显式epoch日志累计约1091.220 GPU-hours，RTX5090/3060，非连续wall time、不含全部探索。作者RTX3070容器run179例5.534h；相对历史参考61例共196voxel不同，不能称逐体素一致。显存峰值未测。本次未运行任何模型。

论文说明179被反复用于选择，global operating points不是完整nested CV；读懂方法不等于得到无偏冠军增益估计。小灶/合成增强尝试非全部保留，不能替作者说冠军靠diffusion增强。

### 4.2 buckeye：强分割基础上学习删ET假阳性

[论文029](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_029.html)：五折nnU-Net、Dice+TopK10%、5000epochs；OOF组件形态/位置/置信度训练gradient-boosted rejector，推理仅删除ET。

ET LW-DSC增加0.0224至0.7187；小ET F1下降0.0170，拒绝110/1039 ET components。宏平均分割改善和小灶检测损失同时成立；precision-limited是该模型局部经验，不是所有转移瘤模型都应删小灶的定律。

公开训练Slurm、`train_rejector.py`、`src/build_components.py`/`apply_rejector.py`、Dockerfile及`model/rejector_R37b_leakfree.joblib`（390527bytes）。**拒绝器绑定该分割输出特征，不能直接套我们的E/K就当收益已验证**。五个大checkpoint外置，镜像匿名manifest401。

### 4.3 DEEPNEURO-MET：分割与检测协同

[论文005](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_005.html) / [DEEP-MET](https://github.com/deepnoid-ai/DEEP-MET)：nnU-Net分割/RC专项分支与nnDetection辅助证据。实际`run.sh`串联input准备、nnU-Net、两个nnDetection task、`boxes_2_seg_4.py`、flat输出。

作者ET/RC/TC/WT LW-DSC0.745/0.671/0.765/0.720，ET smallF1约0.501；检测协同有overlap优势但也有检测取舍。此列不能与他队宏平均/内部104分数排冠军顺序。

源码含框架/入口/镜像配方，模型需另放`models/`；没看到冻结权重。GitHub license元数据空，README自有扩展声明Apache-2.0，第三方许可各自保留，不把全仓统一算一个许可证。

### 4.4 Mr. Cancer Defeater：匹配归一化的T1c−T1n

[论文014](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_014.html)：ResEncM五折+ResEncL三折、区域监督、差分enhancement通道和RC FP suppression。

关键是**matched-normalization subtraction**；不能任意独立z-score两图相减后宣称等价。公开[CPU pipeline](https://github.com/rnorlund/bratsmets-cpu-pipeline)含QC/split/差分/后处理/评价，GPU训练沿residual presets与论文参数。未见实际冻结成品权重下载。

### 4.5 aymuos/KCL：解码改变，而非继续堆网络

[论文035](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_035.html) / [源码](https://github.com/aymuos15/brats2026-mets)：三ResEncL变体概率融合，seeded hysteresis低阈值扩展、高阈值可靠seed，RC从connected-component-loss来源单独取得。

README示例lo0.25/hi0.70、RC独立阈值/keep-largest是其配置，不是普适最优。负结果、sweeps、slides、诊断和论文源码公开；旧Dataset名字含2025，不能只据旧目录名否定2026论文。

本地`fold_all`开发harness可能含记忆训练病例，组件/评分规则可让收益反向；不替代OOF/官方179。Docker目录含2025旧配方与重建中的2026配方，本次没有判成已可直接拉取的成品；根许可未核到，复用先确认授权。

另外已定位[演示幻灯片](https://github.com/aymuos15/brats2026-mets/blob/main/slides/build/main.pdf)、[论文源码](https://github.com/aymuos15/brats2026-mets/blob/main/paper/main.tex)、[三路组装入口](https://github.com/aymuos15/brats2026-mets/blob/main/src/decode/assemble_3way.py)和[开发交接](https://github.com/aymuos15/brats2026-mets/blob/main/docs/BraTSHandoff2026.md)。文件树确认这些文件存在，幻灯片未另作逐页视觉审阅。README曾把某RC保留规则的本地+0.028与validation−0.016并列报告，值得阅读其失败诊断，但“融合阈值安全”等作者经验不应提升为普遍定理。

### 4.6 MIA_Lab：细token transformer、标签QC和失败恢复

[论文015](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_015.html) / [训练推理](https://github.com/nittifra/brats-met-2026)：Primus body与小灶specialist，Voronoi/实例均衡目标，减少大灶淹没小灶梯度，再做failure/recovery。公开训练说明body四通道，specialist仅T1c/T1n，各有label/plans/trainer，**不是同一模型换阈值就叫专家**。

独立 [brats-met-qc](https://github.com/nittifra/brats-met-qc) 是标签复核工具：预测/GT分歧、fold-union与ensemble的Semantic Spatial Uncertainty产生review queue，经UI人工inject/swap/dismiss并留审计。**这里QC是工作流，不是名叫QC的独立分割模型，也不是我们旧G1的tumour_ssim QC算法。**

QC用any-overlap、低SSU和尺寸等候选条件，README明确非官方评分器。flag不证明标注错；train-only范围、人工确认、患者隔离和标签版本须另管，不能自动改frozen val/test GT。源码Apache-2.0、QC工具MIT；模型说后续发布/请求获取，releases空，未见直接下载。

### 4.7 GLI2026/Graylight：按尺寸选U-Mamba与CATMIL

[2026博客](https://graylight-imaging.com/blog/brain-metastasis-segmentation-using-ai-graylight-imaging-presents-new-results-at-miccai-2026/)→[论文013](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_013.html)→[Bitbucket](https://bitbucket.org/fphealthcare/brats2026-ensemble)，作者/成员吻合。普通U-Mamba强于大灶，CATMIL增强小灶sensitivity，按尺寸融合并处理RC。

源码继续指向 [U-Mamba](https://github.com/bowang-lab/U-Mamba)、[SmallLesionMRI/CATMIL](https://github.com/luumsk/SmallLesionMRI)及[2604.08015](https://arxiv.org/abs/2604.08015)，它们是基础方法线索，不是该队完整冻结模型。

本次Bitbucket commit `d7ae160f4b48839826ea3703b7d364ee42bf2f29`仅**三个文件**：69byte README、`ensemble.py`、requirements；无训练/checkpoint/Dockerfile/LICENSE。脚本默认12720mm³分流、20mm³high-pass，RC来自普通U-Mamba，不用底座仓库填补缺项。

静态风险：`ensemble()`里的`partial`绑定`ensemble`自身而非`ensemble_single_file`，Pool.imap传filename疑似参数冲突/错误递归；RC参数说明与真实覆盖来源不一致。本次未运行/修原仓，不称其比赛容器失败，但当前公开脚本不应当成已测试即用实现。

2025冠军、2026检测第二/分割第四、Orchestrator2025条目不同，不混成“2026冠军源码”。

### 4.8 Vision_Lab：与我们最相关的CarveMix-RC增强

[论文021](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_021.html) / [CarveMix-RC](https://github.com/shiblyg/carvemix-rc)：针对稀缺RC，把真实donor cavity及影像局部移到target，配RC-weighted/oversampling、集成和后处理。

已读generator有signed-distance carve mask、局部mean/std强度harmonization、可容纳bbox的随机位置与RC overlay。这是**真实腔体copy-paste，不是diffusion生成**。论文没有atlas放置约束，也不排除与target既有肿瘤重叠，不能称具有我们Route A的严格不覆盖事务。

优点是成本低、直接针对RC；解剖合理性、donor/target患者隔离和标签覆盖需要独立核查。MIT，生成/trainer/FP-aware ensemble/postprocess/bootstrap有，未见权重或Docker成品。本次不新增增强实验。

### 4.9 Bio-Intel：公开模型与预训练效益的限制

[论文020](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_020.html) / [源码](https://github.com/thekeramik05/MICCAI-METS_challenge_bi-intel)：Triad初始化encoder模型A；BrainIAC冻结ViT作辅助特征模型B；RC oversampling、类别融合、尺寸与置信度同时低才删组件。A五fold，B仅1/3两fold；RC A/B权重0.15/0.85，其他0.5/0.5。

README的重要反例：单fold控制下BrainIAC相对同形状random features无显著优势，RC ΔDice=-0.0137、CI[-0.0417,+0.0098]。有证据支持辅助通道设计，**不支持“基础模型表征必然有增益”**；Triad初始化未完整消融。

[模型卡](https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model)及LICENSE已读。`weights.zip`元数据2472704349bytes、SHA256 `d308d3dd5dff4ec8fa899d64f056d8a18651b4d4e56f6618c66a2d8998e334f3`。模型卡称7fold与plans/dataset，本次未下载ZIP核内容。

模型B内嵌BrainIAC权重，但wrapper构造仍依赖另取`BRAINIAC_CKPT`；ZIP公开≠依赖齐全。模型B限非商业学术研究/教学，商业与临床workflow需另授权。**比赛规则与第三方许可分别核对**：官方禁止其他分割任务预训练权重，仅凭论文/排名不能替组织方认定某基础encoder用法获批或违规。

### 4.10 TheRealBoys：三套真实成品权重线索

[论文002](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_002.html) / [GitHub](https://github.com/NicoloPecco/Segmentation_METs_BraTS2026)：MedNeXt-L、nnU-Net、STU-Net。公开`predict.py`第一阶段权重0.25/1/1，RC阶段MedNeXt/STU-Net=1/0.25、阈值0.1，另brain mask和小组件过滤。

| 模型仓库 | 文件 | bytes | LFS SHA256 |
|---|---|---:|---|
| [MedNeXt](https://huggingface.co/NicoloPecco/MedNeXt_BraTS_Metastases) | `model_best.model` | 756781752 | `d8819d00bdf5f2cc044c4a208666a4128455a2a071c4dc71288f9d92f482988c` |
| 同仓库，必须一起取得 | `model_best.model.pkl` | 401532 | `46b7546e8e5ebdc2f36b2a4659fe5861fd23c5d69313ad8975fab41afdc5cf02` |
| [STU-Net](https://huggingface.co/NicoloPecco/STUNet_BraTS26_Metastases) | `checkpoint_best.pth` | 3521470590 | `d5c7c04f8255531af58322ab28edd7972e279a46c667037c4705a18c9b1fc07f` |
| [nnU-Net](https://huggingface.co/NicoloPecco/nnUNet_BraTS_Metastases) | `checkpoint_best.pth` | 246962786 | `5dcd99861935cdb953e05449fc8b492ff1bf9444b853179292ff00b1574fa6cd` |

三API均200、private=false/gated=false；SHA是LFS服务器元数据，**非本地传输后byteSHA复验**。MedNeXt模型卡CC-BY-4.0，GitHub源码Apache-2.0；另两卡许可未逐字核。

GitHub只有LICENSE/README/environment.yml/predict.py四文件，非完整训练/eval开源。模型卡留github.com/TODO/citation TODO，以实际GitHub README链获取。还需custom trainer、plans/dataset、v1/v2/MedNeXt环境与目录验收，本次未加载/推理；有权重≠即用成功。

## 5. 其余方案与可复现性

| 队伍 | 关键区别 | 当前公开程度/风险 |
|---|---|---|
| bin_JLU | 五折ResEncL1000epochs；RuleClean/边界扩展/RC brain-fraction；13负结果/OOF | Apache-2.0，Docker/推理/postprocess/OOF；未见权重；某stage榜单收益不在OOF复现 |
| FLAIR-Viaevum | offline metric harness+官方输入firing-volume parity迁移门；METS hysteresis/top-1RC | MIT，多赛道源码/论文/配方；未见权重；历史“准备未提交”文档和更晚README声明不能当同一时点验收 |
| WIN_BRATS | 5fold选3，persistent predictor避免重复装载，作者21.1→6.4h | Docker/分析有；5个约820MBcheckpoint排除；README许可与GitHub元数据另核 |
| Core_Neuro_AI | 区域ResEnc/asymmetric Tversky+BCE | trainer/plans/配方；未见根LICENSE/权重；后续Primus链接不是原提交 |
| Latent_Minds | CC-weighted双FN/FP focal maps，label×size12组采样 | loss/trainer/sampling/eval/Docker；未见根许可/权重，不等于简单提高gamma |
| Pixel_Perfect | MedNeXt供tumor、nnU-Net供RC硬覆盖；填洞/物理阈值 | 论文/README；tree API限流，权重/许可/镜像未完成核验，不说全部没有 |
| Brain_exe | region-aligned训练、target-specific2D/3D fusion | 配置/代码/Docker；模型bundle非公开、未见根LICENSE |
| IUB | plain/ResEncL/region/Primus比较 | training/eval代码；权重on request；README称MIT但根LICENSE未见，架构比较仍混训练配置 |
| Wangwangwang | gated frequency/adaptive Laplacian+class-wise ensemble | 代码/配置/Docker；README仅8bytes、根许可/权重未见，需论文补配置 |
| Angelus | epistemic uncertainty reliability supervision weighting | trainer/目标有；Apache badge与根LICENSE缺项分开，旧URL不否定2026身份 |
| ECNU_NYU_team | missingT2W的MT-ED/MT-BBDM latent ensemble；RC CE×3+focalγ2 | 论文028与公开代码链；内部104、补全与后续R9/G1增强不是同一模型/评分 |

### 5.1 bin_JLU：标准训练、可拆分后处理与负结果

吉林大学Haobin Liu/Xin Wang。[论文009](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_009.html) / [arXiv](https://arxiv.org/abs/2609.11477) / [代码](https://github.com/hornbeamliu/brats2026-met)。五折ResEnc-L，各1000epochs，常规Dice+CE，不把所有收益归给新损失或新架构。

RuleClean清理小组件并检查邻接；BoundaryExpand按概率对RC边界扩展；brainF70以脑mask内占比抑制RC假阳性。作者通过1296例五折OOF复查：前两项收益较稳定，第三项在榜单上改善却不在OOF复现。官方validation ET/TC/WT/RC LW-DSC约0.733/0.751/0.713/0.549，是该作者该阶段的结果，不是隐藏测试分数。

先读[容器后处理](https://github.com/hornbeamliu/brats2026-met/blob/main/docker/postprocess.py)、[推理](https://github.com/hornbeamliu/brats2026-met/blob/main/docker/predict.py)、[评价侧后处理](https://github.com/hornbeamliu/brats2026-met/blob/main/evaluation/postprocess_brats.py)，再看论文13项负结果。Apache-2.0；无已定位的公开成品权重。其论文称所用四模态均native完整，这是**作者对自身数据的声明**，与官方允许合成/缺失T2W不能视为同一数据身份，不能据此认定我们数据处理错或所有队伍输入相同。

### 5.2 FLAIR-Viaevum：跨赛道论文、迁移门与提交预算

中山大学肿瘤防治中心Long Lai/Minli Hu等。[跨赛道论文PEDs014](https://papers.miccai.org/miccai-2026-sat/BraTS_PEDs_014.html) / [源码](https://github.com/B664577/brats2026-flair-viaevum)。重点不是一个“FLAIR网络”，而是nnU-Net系列集成和**候选后处理能否迁移到官方输入**的开发流程。

METS用seeded hysteresis：可靠高阈值seed连接低阈值邻域，再做top-1 RC选择。README配置lo0.2/hi0.7，与KCL的lo0.25/hi0.70不同，不能照抄成统一最佳参数。第一道门是离线评分，第二道门比较官方输入上的预测激活体积；作者保留了9项拒绝机制与收益反向案例。

阅读[迁移门](https://github.com/B664577/brats2026-flair-viaevum/blob/master/docs/GATE.md)、[hysteresis应用](https://github.com/B664577/brats2026-flair-viaevum/blob/master/task1_met/t1_hyst_apply.py)、[公开技术稿](https://github.com/B664577/brats2026-flair-viaevum/blob/master/docs/lncs/flair.pdf)和[审稿回复](https://github.com/B664577/brats2026-flair-viaevum/blob/master/docs/lncs/response_mets.pdf)。MIT，权重未定位，Synapse镜像匿名401。论文2026-07-23的METS proxy平均0.554、22/70不是截图最终综合第6；PEDs/GoAT的结果也不转计METS。其离线50voxels抑制等规则须与当前官方评价版本逐项核验，不能因论文写“exact”就默认兼容。

### 5.3 WIN_BRATS：选择少数fold并消除重复加载

埃及Mansoura University。[论文007](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_007.html) / [源码](https://github.com/Abdelrhman-Elgaous/winbrats-docker)。五折ResEncL训练完成后枚举10种三折组合，最后部署fold1/2/4；persistent predictor把权重常驻内存，每fold只加载一次。

公开README给出L4上的测量/303例预算折算：五折CLI约21.1h、三折CLI12.6h、三折常驻6.4h；L4是作者用作挑战A10G代理的设备，**不是本项目A100上实测耗时**。作者567例分析主要用于集成选择/等价性研究；即使论文报告TOST通过，也不能自动解释成当前官方2026隐藏测试完全等价。

可读[fold选择](https://github.com/Abdelrhman-Elgaous/winbrats-docker/blob/main/analysis/fold_selection.py)、[训练暴露审计](https://github.com/Abdelrhman-Elgaous/winbrats-docker/blob/main/analysis/leak_coverage.py)、[统计](https://github.com/Abdelrhman-Elgaous/winbrats-docker/blob/main/analysis/statistics.py)、[推理](https://github.com/Abdelrhman-Elgaous/winbrats-docker/blob/main/infer_logic.py)及[逐病例结果](https://github.com/Abdelrhman-Elgaous/winbrats-docker/blob/main/results/per_case_lesionwise.csv)。源码/配方/结果CSV存在，checkpoint外置；README称Apache-2.0，GitHub识别为NOASSERTION，使用前读实际LICENSE。对我们最直接的启发是把模型数量与加载开销分开核算，而不是直接删fold。

### 5.4 Core_Neuro_AI：区域目标与非对称Tversky-BCE

University of Technology Sydney，Sri Sai Hemanth Bollepalli。[论文008](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_008.html) / [源码](https://github.com/sshemanth/brats2026)。直接预测WT/TC/ET/RC四个重叠区域，五折概率平均，配区域阈值、层级一致性与组件过滤。

论文相对标准nnU-Net报告RC Instance F1从0.238到0.586、LW-DSC从0.459到0.578；它比较的是整个新配置，不是单独Tversky的因果收益。非对称FN/FP惩罚与区域重建顺序都影响结果，不能只替换loss名就认为实现等价。

读[ResEncL trainer](https://github.com/sshemanth/brats2026/blob/main/nnunet_trainers/nnUNetTrainerCoreCavityV3ResEncL.py)、[RC配置](https://github.com/sshemanth/brats2026/blob/main/configs/v3_rc_postprocessing.json)、[推理](https://github.com/sshemanth/brats2026/blob/main/docker/infer.py)。仓库也含后续Primus/其他实验，不能把所有文件合成当次提交；未见根LICENSE和公开权重。

### 5.5 Latent_Minds：组件加权双图focal与12组采样

IIT Bombay，Vineet Jangir/Bhavesh Sandbhor。[论文026](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_026.html) / [源码](https://github.com/vineetjangiriitb/BraTS2026-MET)。小病灶按数量占多，但大病灶占体素多，作者用GT组件大小的逆权重压缩这种差距，并设置上限；另由预测假阳性在线构建第二权重图，逐点取maximum，使漏检与幻觉同时受到约束。

采样不是仅“多抽RC”：NETC/SNFH/ET/RC四标签乘三尺寸层共12组。README列fold0/1与fold2–4使用不同trainer/训练epoch，因此最终ensemble不等于五次完全同参训练。作者报告类别整体RC oversampling会造成precision collapse，也报告某小灶重加权反而降低小灶检测，均是特定配置的负结果。

重点读[双图loss](https://github.com/vineetjangiriitb/BraTS2026-MET/blob/main/trainer/nnUNetTrainer_CCwFocalDualMap.py)、[采样](https://github.com/vineetjangiriitb/BraTS2026-MET/blob/main/trainer/nnUNetTrainer_CCwFocal.py)、[patient-group split](https://github.com/vineetjangiriitb/BraTS2026-MET/blob/main/scripts/data/make_splits.py)。正文validation ET/TC/WT/RC LW-DSC约0.685/0.714/0.675/0.560；不要与另队宏均值直接排序。代码/数据准备/推理存在，README明确无权重，根LICENSE未见；其训练标签QC改动需要单独版本与审计，不能套用到冻结验证集。

### 5.6 Pixel_Perfect：不同网络负责肿瘤与术腔

IIT Delhi/AIIMS New Delhi。[论文034](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_034.html) / [代码入口](https://github.com/Rossette-46/BraTS-MET)。MedNeXt-M主要提供肿瘤标签，nnU-Net提供RC，随后按硬标签覆盖、孔洞填充和物理体积阈值完成输出，**不是对所有类统一软概率平均**。

论文报告后处理后RC LW-DSC nnU-Net0.476、MedNeXt0.448，因此将RC交给前者；最终宏LW-DSC约0.639、small-instance F1约0.311。尺寸阈值对检测的影响大于对重叠指标的影响，是具体阶段实验结论，不是通用最佳阈值。

本次论文全文与raw README已读，但GitHub REST额度耗尽前未取得该仓递归tree。**目前不能判断权重、许可证、Docker或其他文件全部不存在**；把它保留为“方法明确、资产未完成核验”，而不是“只发布了README”。

### 5.7 Brain_exe：按区域选择2D与3D的贡献

Semmelweis等多机构合作。[论文023](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_023.html) / [源码](https://github.com/BrainExeBrats2026/brain-exe-brats2026-task1)。五个2D+五个3D full-resolution模型，直接学习重叠区域；最终3D/2D权重WT0.65/0.35、TC0.5/0.5、ET0.75/0.25、RC0.5/0.5。阈值严格>0.50，写回顺序[2,1,3,4]；TTA和组件后处理关闭。

读[融合配置](https://github.com/BrainExeBrats2026/brain-exe-brats2026-task1/blob/main/configs/fusion_rc50.json)、[训练说明](https://github.com/BrainExeBrats2026/brain-exe-brats2026-task1/blob/main/docs/TRAINING.md)、[推理及几何核验](https://github.com/BrainExeBrats2026/brain-exe-brats2026-task1/blob/main/run_inference.py)。作者排除1个未修非法标签病例后用1295开发例；我们不能自动复制排除。

作者OOF宏DSC0.5605、official validation0.6456；两个数据群不同，差异不是同一患者训练增益。类目标→区域目标和融合方案同时改变，论文也承认不能从官方比较隔离两者贡献。公开8个文件级blob加目录项，含配方/配置但无checkpoint，根LICENSE未见；不把“小仓库”解释成完整训练源码公开。

### 5.8 IUB：架构比较里的分割-检测取舍

Independent University, Bangladesh。[论文027](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_027.html) / [arXiv](https://arxiv.org/abs/2609.24769) / [源码](https://github.com/mahdiislam79/BraTS_METS_2026)。比较plain nnU-Net、ResEncL、区域训练及Primus，UCSD纵向病例按patient group划分。

Primus在作者报告的aggregate DSC/NSD较强，却在同文lesion-wise F1明显弱于ResEncL；直接概率平均未保留全部检测优势。这说明“宏Dice最高”与“发现更多小灶”不是同一个选模目标，不能把不同指标名混为官方综合最优。

读[顺序说明](https://github.com/mahdiislam79/BraTS_METS_2026/blob/main/RUN_ORDER.md)、[区域标签配置](https://github.com/mahdiislam79/BraTS_METS_2026/blob/main/configs/dataset002_dataset.json)、[融合](https://github.com/mahdiislam79/BraTS_METS_2026/blob/main/scripts/15_ensemble_final.py)。plans/训练/eval/后处理/公开结果和论文PDF存在；权重on request，README称MIT但根LICENSE未见。架构比较同时含训练配置差异，不能当作严格同预算架构消融。

### 5.9 Wangwangwang：频率增强与类别专属融合

山东科技大学Ruifeng Wang/Yanjun Peng等。[论文011](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_011.html) / [GFCE源码](https://github.com/qqqwwweressaa/GFCE)。标准residual encoder-decoder与门控Adaptive Laplacian Convolution输入分支形成两路，按WT/TC/ET/RC分别融合，再做阈值/组件处理。重点是两个分支提供何种互补，不是“加频域模块就一定提高小病灶”。

已定位[融合搜索](https://github.com/qqqwwweressaa/GFCE/blob/main/scripts/classwise_ensemble_search.py)、[运行入口](https://github.com/qqqwwweressaa/GFCE/blob/main/run_model.py)、[两分支plans/dataset](https://github.com/qqqwwweressaa/GFCE/tree/main/models)。论文validation LW-DSC ET/TC/WT/RC约0.709/0.685/0.652/0.586。

README实际仅`# GFCE`，仓库包含大段nnU-Net源码副本与少量自有入口；不能用总文件数充当完整自有方法说明。根许可证和权重未见，先依论文与custom trainer核实现再考虑复用，不能把底座许可证自动覆盖所有新增代码。

### 5.10 Angelus：把不确定性用于监督加权

台湾NHRI，Ajesh Saviour Paravila。[论文031](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_031.html) / [源码](https://github.com/zephyr-9598/BraTS-2026-metastasis)。架构不变，利用fold预测的分歧为每病例/每类生成可靠性权重，分歧越高监督权重越低；作者在该配置下认为纯epistemic优于entropy/频次校正组合。

读[权重生成](https://github.com/zephyr-9598/BraTS-2026-metastasis/blob/main/scripts/reliability_weights.py)和[trainer](https://github.com/zephyr-9598/BraTS-2026-metastasis/blob/main/src/nnUNet_epistemictrainer.py)。论文承认每例由全部fold预测时包含训练见过该例的模型，因此不确定性是**有in-sample偏差的可靠性启发式**，不是纯样本外uncertainty；这点必须保留。

README写Apache badge/许可声明，但当前tree无根LICENSE、无权重或容器。其表中“Core (1+2+3)”与“All Lesions (1+2+3+4)”不是本文官方TC/WT定义，不能直接按表头搬进统一四区域成绩；需核原始评价字段。README给出[Zenodo DOI](https://doi.org/10.5281/ZENODO.19714728)作为相关资源，本次Zenodo访问受限，未核记录内容、版本或权重，**不列为已确认模型下载**。

### 5.11 ECNU_NYU_team：本队论文也是公开链的一部分

[论文028](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_028.html) / [公开代码](https://github.com/hhhx-lab/ECNU_NYU_brats)。署名华东师大/NYU；missing T2W imputation通过MT-ED与MT-BBDM latent-space ensemble，再由RC加权CE及size-aware focal分割模型处理。

这是比赛方案身份，不是“本线程后来所有G1扩散都参加了比赛”。旧150k病灶条件diffusion、2026-10的R9 VAE自重建目标模型、后续raw-real模态训练、原E与K等必须按自身checkpoint与批次区分。本文按公开论文链定位本队，不把历史内部104成绩改叫官方179，也不把生成SSIM当分割排名。详细本队历史沿总纲已有合同读取，本次没有重复实验或追加作业。

### 常用术语

OOF（out-of-fold）指每例由训练未见该例的fold预测；纵向患者还须patient-group隔离。ResEncL/XL是nnU-Net不同容量的residual encoder配置。TTA是推理时翻转等增强后汇总。RC是术后切除腔，不是小转移瘤的同义词。LW-DSC按病灶匹配后计算，不能与体素global Dice或生成SSIM互换。hysteresis是在高置信seed基础上扩展低阈值连通邻域，并非简单整体降低阈值。本文“有权重”只表示公开服务器文件元数据已确认，未表示本机加载成功。

## 6. 用户四个起点及年度纠错

| 起点 | 深入路径 | 结论 |
|---|---|---|
| [Orchestrator segmentation](https://brats.readthedocs.io/en/latest/guides/segmentation/#pediatric) | Brain Metastases→[algorithm tables](https://brats.readthedocs.io/en/latest/reference/algorithms/)→backends | 原锚点是Pediatric；METS registry列2025 Maria Bancerek/Wes Krikorian及2023模型，非本次2026前列；其他赛道有2026不代表METS已更新 |
| [MICCAI2026 METS分类](https://papers.miccai.org/miccai-2026-sat/categories/#BraTS-METS) | 32PDF全文→作者/机构→code availability→GitHub/Bitbucket/HF | 最核心一手入口；跨分类补FLAIR，不仅搜Task1字样 |
| [Graylight2026博客](https://graylight-imaging.com/blog/brain-metastasis-segmentation-using-ai-graylight-imaging-presents-new-results-at-miccai-2026/) | 作者→013→Bitbucket源码→U-Mamba/CATMIL | GLI2026身份/方法确认；2025历史冠军非2026冠军，公开实现缺完整训练/权重 |
| [Ujjwal Baid仓库](https://github.com/ujjwalbaid0408/BraTS2026_METS) | README/tree→025→releases | 额外baseline，未绑定截图某前列；论文称权重/配方可用，当前tree无权重/Dockerfile，releases空 |

最后仓库README的**0.781/0.921/0.800/0.797为2025 GLI历史表**，不是2026METS该模型结果。论文025当前validation ET/TC/WT/RC约0.677/0.702/0.645/0.526。旧赛道高分不能挪到新方法。

## 7. 镜像与复现分级

仅查轻量metadata，没有拉镜像层/权重。

| 线索 | 实际结果 | 可以说 | 不能说 |
|---|---|---|---|
| TheRealBoys三HF | public/non-gated/API200、4二进制文件metadata | 权重获取入口已定位 | 已下载/加载/复现 |
| Bio-Intel HF | weights.zip metadata/Research-Only许可 | 公共模型包但有依赖限制 | 包内7fold已解压验收/可商用 |
| buckeye joblib | GitHub tree实际文件390527bytes | 小拒绝器公开 | 完整网络权重公开/跨模型增益 |
| MicroBT manifest | 角色/hashes，无binary URL | 原身份可审/源码可重训 | hash等于下载地址 |
| `docker.synapse.org/syn74793974/brats2026-task1:v1` | 匿名manifest401，无digest | buckeye文档镜像名 | 匿名成品可拉取已核验 |
| `docker.synapse.org/syn76138284/brats26-t1:v1` | 同上401 | FLAIR镜像线索 | 已提交/评分/取得 |
| `docker.synapse.org/syn74830137/brats-v18-task1:v18-20260725` | 同上401，Humanitas论文016 | 额外2026线索，未绑定截图 | 随意归给某前列队 |

读论文<可审源码<许可/依赖/权重齐全<严格加载<完整病例推理验收<原提交逐病例输出一致。本次没有声称到运行层。

## 8. 对我们最有价值的研究顺序

以下仅文献建议，不是已授权作业或已实测收益：

1. 先读MicroBT structured anchor/add-only ET，判断候选与父区域支持，再研究恢复，别照搬global阈值。
2. 并读buckeye和bin_JLU负结果，验证“多检小灶必更好”与“删小组件必更好”的条件及metric版本。
3. RC独立线：Mr差分/抑制、DEEPNOID专家、Vision移植、TheRealBoys区域融合；分开输入/采样/loss/解码，不一次全加。
4. 比较MIA/Latent_Minds实例均衡与我们focal；不能凭同名假定目标等价。
5. 要成品基线可先考虑TheRealBoys，权重出处最清楚，但补trainer/plans/许可/环境验收；Bio-Intel依赖与许可更多，不预设基础模型效益。
6. KCL hysteresis/FLAIR迁移门只在train-derived OOF/固定开发集设门，保留空区域、物理单位与统计合同，不读locked test调阈值。

不能把他队179、OOF、我们fixed103/内部104、截图hidden-test rank放同一列排序。真实横评需冻结患者/病例、输入模态与真假T2W、spacing、模型SHA、完整推理和官方metric版本。

## 9. 缺口与检索限制

- 完整官方最终tier未独立取回，截图是排名输入。
- Pixel仅README；本队代码本次只按论文链接定位，没有重复审核本队历史实现。
- 多数冻结weights、Docker digest、提交SHA/输出未公开，不用底座填空。
- OpenReview403/网络错误，转会议公开PDF；Zenodo异常流量403后停止；GitHub匿名API额度用尽，不借凭据/换身份绕过。
- Google普通搜索部分失败；Google学术首屏、arXiv API与精确标题检索成功。arXiv非完整挑战论文库，精确标题未匹配不证明从未发表。
- PDF抽取会断词/公式乱码，数值最终核对原PDF；不是逐图/公式排版验真。
- 所有作者成绩/算力/包内容/tests pass不是本机实测。无ECNU/VPN/训练/影像/模型或镜像下载/第三方执行。

来源与UTC/SHA见[SOURCES](SOURCES.md)与[证据入口](README.md)。保持`experimental_unvalidated / operator_approved=false / formal_gate_status=not_run_not_passed`。
