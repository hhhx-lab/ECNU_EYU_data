---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# 在同一验证集上比较本项目与外部分割模型

复核日期：2026-11-04，Asia/Shanghai。范围是 BraTS-METS 分割，不是 G1 影像生成 SSIM；本次只做可行性与历史复核，没有下载、安装、推理或提交。

## 1. 结论先读

1. **2025 可以找到真正公开的成品镜像。**首选 Orchestrator 的 `MetastasesAlgorithms.BraTS25_1`，Maria Bancerek 等，镜像 `brainles/brats25_met_gli-team`；备选第二名 `BraTS25_2`，Wes Krikorian 等。Docker Hub 的匿名元数据实际返回200、public/active、linux/amd64和完整 digest，不再只是“文档里有一个镜像名”。但未拉取或运行，ECNU容器/GPU兼容性和模型内部输出仍需 smoke。
2. **2026 首选材料较完整的 Bio-Intel，属于条件可运行。**有推理入口、专用训练器、五折清单、公开约2.303 GiB权重包；完整集成还必须取得 BrainIAC 编码器及许可。模型包内部七份权重/plans/dataset尚未解包核实，不能写即开即用。
3. **TheRealBoys 有三套真权重，但现在不是完整可直接运行包。**nnU-Net/STU-Net 的公开模型树没有 `plans.json`/`dataset.json`；其README和predict.py融合权重、RC阈值不一致，环境文件名和运行版本也不一致。需作者补充并确认冻结推理配置。单拿nnU-Net权重不等于复现该队完整模型。
4. **最重要的限制是训练暴露。**我们的fixed103来自公开2025 MET训练数据；Bio-Intel公开五折与103/103逐例匹配，每例列在四个训练折。它的完整集成不能当作这103例上的独立测试。TheRealBoys使用同1296标注数据的1011/285内部划分，但没有公开成员名单，逐例训练暴露未知。2025两镜像的训练名单未知，同源数据风险仍在。
5. **建议先做明确标注训练暴露的工程横评，再考虑独立科学比较。**现有强主对照用K1-175，比赛基座用原E-200，小病灶辅助对照用K4-175和B175；它们不是所有指标统一最好的同一个模型。

## 2. 我们当前到底比较哪套成绩

### 2.1 “K1–K4”应按实际记录解释

实际执行与完成评分的是K1/K2/K4。K1245计划没有一个已完成的K3结果；K5受科学门约束未提交，不应凑成K1至K4四套完成模型。

后续九路报告已经把原E、B175、C40-50、D50、E-SNFH175、F175、K1/K2/K4放到同一fixed103表中。此次重新核对九份原始CSV与验收SHA、病例顺序、72指标列，使用该表而非从历史名称相似的结果中挑分数。

| 路线/所选epoch | ET病灶Dice | RC病灶Dice | TC病灶Dice | WT病灶Dice | WT小病灶F1 |
|---|---:|---:|---:|---:|---:|
| 原E / 200 | 0.6117 | 0.3112 | 0.6577 | 0.6176 | 0.2947 |
| B / 175 | 0.6177 | 0.3533 | 0.6608 | 0.5874 | 0.2977 |
| C40 / 50 | 0.6342 | 0.3443 | 0.6594 | 0.5442 | 0.0905 |
| D / 50 | 0.5843 | 0.4538 | 0.6227 | 0.5284 | 0.2907 |
| E-SNFH / 175 | 0.6482 | 0.3322 | 0.6875 | 0.6235 | 0.2564 |
| F / 175 | 0.6304 | 0.3266 | 0.6737 | 0.6032 | 0.2733 |
| K1 / 175 | 0.6373 | 0.3230 | 0.6881 | **0.6303** | 0.2620 |
| K2 / 175 | 0.6109 | 0.3325 | 0.6511 | 0.6007 | 0.2171 |
| K4 / 175 | 0.6271 | 0.3272 | 0.6705 | 0.6130 | 0.2829 |

数值为各CSV原始mean，不是统一病例交集后的paired均值。ET/RC/TC/WT病灶Dice有限n=85/15/86/87，WT小F1有限n=46；所有103例都参与完整流程。病灶Dice在这套解析器中针对参考大病灶并含漏检/FP惩罚，不是所有病灶或全图Dice。

**主候选K1-175：**九路WT病灶Dice、TC病灶Dice、WT全部F1和WT FP均值较有利；K1对原E的WT Dice收益约+0.0126，但历史同病例95%CI[-0.0147,+0.0396]跨0。不能称正式证明优越。K4小F1优于K1，但九路小F1原值最高是B175；ET最高E-SNFH175，RC最高D50且仅15例有效。不给九路自创一个总分。

**身份：**K1所选epoch175 checkpoint SHA为 `28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204`，MET job3940964和batch完成0:0，103预测，合法标签0–4，72指标验收通过。原E比赛基座epoch200 SHA为 `4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`。九路完整身份见[历史验收](../../../03_experiments/13_nine_route_selected103_official_met/results/VALIDATION.json)。

原E有多份历史成绩。这里仅用九路表绑定的 `selection_evidence/E/leaderboard_metrics.csv`，SHA `294a1451a8737344a929902c8ef118c89bd9a3da971952d11b453fafccc3e167`。另一个real-only评估、旧小病灶selection JSON、官方179排行榜等不能因文件中有E就混进本表。原E不是E-SNFH175、不是E_continue_final；“后续从原E初始化”也不等于从FixV3 final接续。

### 2.2 同一份验证集的准确含义

| 队列 | 本次比较是否采用 | 边界 |
|---|---|---|
| fixed103真实T2W | **采用** | 103扫描条目、69患者组，22组有多个扫描；master val130中authentic子集 |
| master val130 | 不与103混合 | 额外27例缺真实T2W；输入补全策略不同，若研究须独立分支 |
| locked真实T2W test104 | 不采用、不读图 | 内部锁定集；也来自公开训练来源，不能仅因叫test就认为外部模型没见过 |
| master test130 | 不采用、不读图 | 不因本次调研解除锁定 |
| 官方validation179 | 非本次本地评分 | 官方不公开GT，不能用103预测冒充179；现有K1官方179提交未验收 |

fixed103冻结清单：[FIXED103_REAL_T2W_CASES.csv](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1/manifests/FIXED103_REAL_T2W_CASES.csv)，SHA `86ba853dadc64ffc45357895ba0bd543bf5b6b14c0c83eb789f1e875d4aa2e7e`。逐例与G2 master/real-only membership一致，均val/authentic，与本项目train患者组交集0。原始manifest记录44例train_top_level、59例UCSD Training，属于2025公开标注训练来源，不是官方无标签验证。

我们自己的K路线使用固定1035训练/103开发验证。训练集1035可含212个缺T2W补全病例；**这不意味着本表103输入也补全**：103全用真实T1N/T1C/T2W/T2F，推理不启用在线生成，不用VAE重建作为分割输入。相同103曾用于checkpoint选择和报告，所以本项目内部结果本来也是探索性开发成绩。

“本地验证集”需区分清单/标签和完整影像：Mac有103份历史reference labels、13,022,825字节，但指定G2 RealOnly发布目录的imagesTr当前没有NIfTI。历史label目录来自旧评估，未核它与当前参考的逐字节/几何等价。不能随便取它评分；实际统一输入/参考应复用ECNU已验收fixed103缓存，或经授权将同一515文件完整同步后核SHA。此次未扫描全机器影像、未读NIfTI内容，不能断言Mac其他位置绝无完整影像。

### 2.3 既有推理与评价环境

- K1：ECNU单A100-PCIE-40GB；nnU-Net v2专用trainer shim，`3d_fullres`/fold0，tile step0.5、Gaussian=True、mirroring=True，顺序in-process、preprocess/export workers0；没有在线增强或G1推理。
- 历史Slurm加载CUDA12.1，推理Python来自 `segmamba` 环境，这是nnU-Net运行环境名，不表示K1是SegMamba模型。该环境不能为了外部依赖被覆盖。
- 评价独立 `brats_eval` 环境：Python3.10.18、BraTS-evaluation0.0.8、panoptica2.1.0、numpy1.26.4；固定 `config=mets`，解析 `vol_threshold=27`、`overlap_threshold=0.2`。后续执行仍需实际import/源码SHA/配置SHA核验，不能沿用历史版本号即算通过。
- 区域定义：ET=3，RC=4，TC=1∪3，WT=1∪2∪3，**WT/TC不含RC**。官方小灶阈值是参考物理体积<27 mm³，不是所有原始图都恰好27体素。
- 原匹配器DSC门0.000001与解析器检出DSC≥0.2分开；不能仅统一两个数而忽略匹配/惩罚语义。原始标签/原始坐标评分，不在nnU-Net训练1mm网格直接与原生GT比较。

证据：[K1最终验收](../../02_plans/20260921_k1245_rc_aware_execution/K1245_K1_K2_K4_OFFICIAL_MET_FINAL_20261008.md)、[九路详细报告](../../../03_experiments/13_nine_route_selected103_official_met/REPORT_20261008.md)、[历史评价版本](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1/cohort/fixed103/models/K1/evaluation/evaluation_environment.txt)。

## 3. 2026公开方案逐一可用性

### 3.1 Bio-Intel：材料更完整，推荐作为2026工程候选

- [代码](https://github.com/thekeramik05/MICCAI-METS_challenge_bi-intel)，本次main提交 `83c73e0cac0b5a3c77a298547040ea3ea7f99854`。
- [权重](https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model/tree/4a8291024dafb79b4c0cc381438e976a36072201)，`weights.zip` 2,472,704,349 bytes，SHA `d308d3dd5dff4ec8fa899d64f056d8a18651b4d4e56f6618c66a2d8998e334f3`。本次公开API返回200/public/non-gated，与10月27日元数据一致。未下载ZIP、未核解压内容。
- 作者说明包有Model A五折+Model B两折，共七份checkpoint和各自plans/dataset；推理 `src/predict.py`，配置 `src/config.py`，专用trainers和Dockerfile均公开。
- Model B推理仍要求另取得 [BrainIAC](https://github.com/AIM-KannLab/BrainIAC) `BrainIAC.ckpt`；构造ViT时先读编码器，即使fold checkpoint内嵌backbone，也不能在原入口中跳过。Triad编码器只对训练Model A必要，不必为纯推理另下Triad。
- 许可：代码Apache2.0不等于权重Apache2.0。Model B与BrainIAC为非商业学术研究/教学用途；此类比较需遵守权重LICENSE，不用于临床或商业流程。
- 输入通道顺序为 **T1C/T1N/T2F/T2W**，与本项目T1N/T1C/T2W/T2F不同。必须按模态名重排，不将我们的 `_0000`直接当它的 `_0000`。缺T2W时该实现可用FLAIR代替，但本次103必须拒绝缺通道，保证这项fallback不被触发。
- 默认Model A folds0–4，B folds1/3；RC融合A/B=0.15/0.85，其他0.5/0.5；TTA默认关闭、step0.5、类置信度后处理默认开启。这些是作者原配置，不能在103上扫阈值后挑最佳。
- 作者报告部分GPU推理显存<24GiB；不是本项目A100容量验收。本次没有计时或显存实测。

**明确训练暴露：**[公开五折](https://github.com/thekeramik05/MICCAI-METS_challenge_bi-intel/blob/83c73e0cac0b5a3c77a298547040ea3ea7f99854/splits/splits_final.json)与我们的103病例ID逐例交叉：

| 折 | 我们103例出现在train | 出现在val |
|---|---:|---:|
| 0 | 74 | 29 |
| 1 | 79 | 24 |
| 2 | 89 | 14 |
| 3 | 85 | 18 |
| 4 | 85 | 18 |

103/103均在同1295成员全集，每例恰好一个val折、四个train折。Model A平均五折不是OOF；每一例的集成都会用到训练过该例的分支。以上是公开声明split交叉，不是读取权重反推实际训练日志。

仅取该例val折仍不够：按G2患者组，54/103在其val折的train中有同患者其他扫描。另49例只是通过这项直接/纵向病例训练成员筛查，不证明预训练、后处理拟合或权重选择均未曝光。Model B仅发布1/3折，不能给全部103构造完整同规格OOF。OOF子方案会改变参赛集成身份与有效队列，不能冒充原模型完整成绩。

### 3.2 TheRealBoys：权重可取得，但需补齐并澄清

- [代码](https://github.com/NicoloPecco/Segmentation_METs_BraTS2026)，main提交 `9e76175b8a13d74efca5cac6ce290fbbca6d6892`；公开API200。浏览器GitHub展示页本次跳到登录，不据此误判仓库私有。
- [论文页面](https://papers.miccai.org/miccai-2026-sat/BraTS_METS_002.html)，[PDF](https://papers.miccai.org/miccai-2026-sat/paper/BraTS_METS_002.pdf)。本次复用10月27日全文证据，未重下PDF。作者说明1296标注数据划1011train/285val，UCSD按患者划分；未公开成员名单，无法确认103的直接训练交集。不能说“全部1296都训练过”，也不能称103独立留出。

| 模型 | 固定HF仓库/文件 | bytes | 当前不足 |
|---|---|---:|---|
| nnU-Net v2 | [仓库](https://huggingface.co/NicoloPecco/nnUNet_BraTS_Metastases/tree/5852879d40e5f2cc5b5d1dd5db849dfa7fc41355)，checkpoint_best.pth | 246,962,786 | 公开树只有README/attributes/权重，无plans/dataset |
| STU-Net | [仓库](https://huggingface.co/NicoloPecco/STUNet_BraTS26_Metastases/tree/5aab6c691e922a7c773185c85d1c80bcb3da309a)，checkpoint_best.pth | 3,521,470,590 | 无plans/dataset，需STUNetTrainerLarge等专用代码 |
| MedNeXt | [仓库](https://huggingface.co/NicoloPecco/MedNeXt_BraTS_Metastases/tree/c6b49675585f695d20c1a06e9d775a6b3216197b)，model_best.model及.pkl | 757,183,284 | pkl未读取；原v1目录、trainer及计划完整性仍待核 |

合计4,525,616,660 bytes=4.214809 GiB，仅权重/配套pkl，不含环境/概率暂存。各SHA见公开审计，三个模型卡声明CC-BY4.0，代码Apache2.0。

**当前确切问题：**

1. nnU-Net v2的 `initialize_from_trained_model_folder`需要对应plans/dataset；不能拿我们自己的1mm plans顶替，不可按论文patch大小猜全套归一化/resample参数。这会改变模型预处理及输出。
2. README的NSE权重是MedNeXt/nnU-Net/STU-Net=1/0.5/0.25，RC=0.75/0.25、tau0.3；predict.py实际NSE=0.25/1/1，RC=1/0.25、tau0.1。至少两套不一致配置，未经作者确认不称复现参赛最终版。
3. README命令引用 `environment_mednext.yml`，仓库实际 `environment.yml`；导出Python3.8、nnunetv2 2.0、Torch2.1+cu118，论文描述Python3.11。导出环境没有完整安装MedNeXt/STU专用入口，不是全依赖锁定。
4. 原入口使用/input、/output、/opt/models和/tmp多个固定目录，并保留三套全体积概率；需独立挂载与新scratch，避免旧case残留混入。MedNeXt裁剪概率恢复、轴序与空间必须实测，不可仅见保存了NIfTI就验收。
5. 每类CC<50的后处理是原模型的一部分，与官方<27 mm³小灶评分门不是同一阈值，原生spacing不统一时尤其不能混用；不为改善103小F1擅自去掉它。

推荐先向作者索取 `plans.json`、`dataset.json`、专用trainer版本、实际split名单和最终融合/RC参数；不要在我们103上试两套再选择成绩高的。单nnU-Net分支若补齐可作单组件基线，但必须另命名，不能标为TheRealBoys完整集成。

### 3.3 MicroBT / COMPASS-METS

[代码](https://github.com/leonlou921/COMPASS-METS)，[资产合同](https://github.com/leonlou921/COMPASS-METS/blob/a93fa7e65c7cd66aaa2d81c44e0c4f9631c28ad0/docs/ARTIFACTS.md)仍明确15checkpoint及5个学习型gate模型“external/no URL supplied”。有SHA不等于有二进制；本次不能选作直接拿来推理的成品。源码训练复现不是本次成品横评。

## 4. 2025 Orchestrator方案能不能用

### 4.1 选对任务，不选pediatric

用户链接定位到 [Pediatric](https://brats.readthedocs.io/en/latest/guides/segmentation/#pediatric)，那是儿童脑肿瘤任务，标签含义与脑转移瘤不同。**我们应选 [Brain Metastases](https://brats.readthedocs.io/en/latest/guides/segmentation/#brain-metastases)**，不是Pediatric、GLI或Missing-MRI模块。相同四模态名称并不表示相同疾病/标签。

本次[注册表源码](https://github.com/BrainLesion/brats/blob/f587c3f7264b89a08981b6865facc46d54842e67/brats/data/meta/metastases.yml)和页面列出2025第一/第二名；未列2026 METS条目。注册表包含别的赛道2026算法，不能把它们当2026 Task1模型。

| 版本 | 作者/文档名次 | 实际镜像 | 压缩bytes/GiB | public状态 |
|---|---|---|---:|---|
| BraTS25_1 | Maria Bancerek等，1st | [brainles/brats25_met_gli-team](https://hub.docker.com/r/brainles/brats25_met_gli-team/tags) | 4,446,698,630 / 4.141 | 匿名Hub API200、public、active、linux/amd64 |
| BraTS25_2 | Wes Krikorian等，2nd | [brainles/brats25_met_weskrikorian](https://hub.docker.com/r/brainles/brats25_met_weskrikorian/tags) | 10,719,604,197 / 9.983 | 同上；注册表cpu_compatible=true |

第一名冻结digest：`sha256:f83d914d4d7a9d316f7605abaf03fa6adbcf0e86d840d93276e86ad65b2a2a82`。

第二名冻结digest：`sha256:7dcdf6f681c7148525df712a016d3a9035294cfbdbdc704bb905277f1fc54541`。

latest可变；执行前应重新核对并按上述digest固定。压缩大小不等于解压/沙箱/缓存/临时输出总占用；本次未验收服务器余量或成功pull。Docker Hub无方法描述和模型训练名单，Python封装Apache2.0也不能替代容器内权重/第三方依赖许可。

### 4.2 原接口可接我们的四模态

以下仅是后续实现骨架，**本次没有执行**。参数必须由冻结103 manifest逐例提供，不能让glob误收val27或test；不要把GT放/input。

```python
from brats import MetastasesSegmenter
from brats.constants import Backends, MetastasesAlgorithms

segmenter = MetastasesSegmenter(
    algorithm=MetastasesAlgorithms.BraTS25_1,
    cuda_devices=allocated_host_gpu_id,
)
segmenter.infer_single(
    t1c=case_paths["t1c"], t1n=case_paths["t1n"],
    t2f=case_paths["t2f"], t2w=case_paths["t2w"],
    output_file=exclusive_output_file,
    backend=Backends.SINGULARITY,
)
```

Python接口可显式按模态名接入；batch模式要求每例文件夹和同名 `-t1c/-t1n/-t2f/-t2w.nii.gz`。原封装把输入复制到临时目录、生成内部标准病例名并维护回映射。2025 MET注册名模式统一 `BraTS-MET-{id:05d}-000`，不同原时间点会被改成-000：需smoke核是否改变模型行为，并把内部ID到原始扫描/患者/时间点映射存证，不把命名改变当删除纵向病例。

**输出标签与治疗时间点尚待实测。**2025 MET框架覆盖治疗前后，不应未经检查就认定无RC；也不能仅凭注册表认定每个镜像都正确输出0–4。需确认label1=NETC、2=SNFH、3=ET、4=RC、native网格正确；若某模型确实不支持RC，报告RC不支持/NA，另立共同区域分析，禁止把它伪装成完整四区域竞争者。

### 4.3 ECNU/HPC与本机兼容性

- 两镜像都是Linux amd64。我们Mac不是NVIDIA CUDA工作站；第一名无CPU兼容声明，不适合许诺本机直接GPU运行。第二名有CPU声明，仍未验证Mac架构模拟/性能，不建议用作高效103例主路线。
- [官方后端文档](https://brats.readthedocs.io/en/latest/guides/backends/)支持Docker和Singularity。ECNU不能默认用户拥有Docker daemon或root权限；容器内requires_root不等于可向服务器取得sudo权限。
- 当前Singularity实现实际执行 `singularity build --sandbox --fakeroot`，运行时也涉及fakeroot/可写沙箱；文档写支持HPC不等于ECNU管理员已配置subuid/subgid、user namespace或可用singularity命令。Apptainer替代路径亦需独立兼容核验，不绕权限。
- Singularity缓存默认落 `tempfile.gettempdir()/brats_singularity_images`，要把TMPDIR和镜像/scratch预算明确配置到获配持久空间，不让登录/tmp成为未登记长期存储。
- 原封装2025分支将临时/input挂载rw，不应把原始数据直接以rw交给未知容器。复用独立只含四模态的暂存副本；源数据只读、执行前后SHA一致。是否可将临时/input进一步设ro须smoke，不能擅改使镜像失败又称模型不行。
- 官方Singularity GPU参数是**宿主机GPU ID**，可覆盖已有CUDA_VISIBLE_DEVICES。需按Slurm实际分配映射，不能固定0而误用未分配GPU；记录宿主/容器可见GPU一致，单GPU隔离。
- 运行隔离网络、GT不可挂载；不执行自动安装，不修改原segmamba/brats_eval环境。容器entrypoint/workdir、输出权限、finite、形状/spacing/affine、类别、完整103覆盖均须smoke/全量独立验收。

## 5. 怎样做到真正同口径横评

### 5.1 先明确比较目的

**A. 成品工程横评：可行。**冻结fixed103和我们的K1/原E模型，跑2025第一名或Bio-Intel完整集成，按同一个MET流程评分。所有列加训练暴露状态：本项目留出但选模使用；Bio-Intel完整集成公开split训练暴露；TheRealBoys/2025名单未知。报告标题用“同fixed103公开成品描述性横评”，不称独立测试或最权威冠军排名。

**B. 独立泛化比较：当前未具备。**需要所有模型均未用于梯度训练、选择、阈值拟合及同患者纵向曝光的共同病例，并考虑预训练来源。我们的内部val/test是从公开标注训练集划出来，不能简单改用locked104解决外部训练暴露；179不公开GT不能本地评分。可向作者索取split/训练与拟合范围，或另外获得有授权GT的新患者外部队列。若要用各方法在本项目相同patient split重新训练，则是另一项同预算方法比较，不是拿成品直接跑。

**C. OOF补充分析：只在身份明确时单列。**Bio-Intel的单val折可避免直接该扫描训练，但54例还有纵向同患者交集，七分支原集成无法覆盖全部OOF。筛出49例或单分支均需新的冻结子合同，并在该子集重新评价我们的模型，不拿原103mean相减。预训练/阈值选择泄漏未知时仍不得称完全独立。

### 5.2 同输入与原生输出，不强行同预处理

公平要求同病例、同四份原始真实图、同GT、同空间与同评分程序；**不要求将所有网络塞到我们的nnU-Net plans/归一化环境里**。各自用训练时对应的原始预处理/target spacing与作者冻结后处理，把结果正确恢复至同一原生参考网格，再统一评分。修改外部输入归一化、用我们的VAE图或替换plans，会变成改模型实验。

本项目和TheRealBoys通道序T1N/T1C/T2W/T2F；Bio-Intel通道序T1C/T1N/T2F/T2W；Orchestrator按参数名接入。要显式字典映射并校验dataset.json，不能凭文件排序接通道。GT只交评估器，不交外部推理；非1mm体积资格、crop/transpose/native恢复不能由文件名推出。

### 5.3 推荐的完整执行门

1. **冻结合同：**只103，无新划分、无合成T2W；冻结515源输入/参考SHA、正确标签overlay、患者组、模型权重/镜像digest、代码/依赖、每模型TTA/后处理/seed。采用新唯一实验目录，旧结果只读。
2. **轻量远端资源预检：**经用户授权后核ECNU获配目录、当前额度/余量、容器能力、NVIDIA驱动、单GPU与环境。镜像压缩15.166GB加解压/沙箱/转换副本不是可直接从历史3TiB余量放行；权重包压缩也不代表解压预算。先实际manifest再估容量。
3. **完整资产验收：**授权后下载至独立目录，校验大小/SHA/digest、plans/dataset/encoder/许可；拒绝猜参数或使用未知checkpoint。Torch/pickle不安全对象仅在隔离环境加载，不在base自动反序列化未知模型。
4. **小规模smoke：**冻结选择的典型真实T2W病例涵盖RC、无RC、纵向时间点、非1mm几何；只检运行与空间身份，不用于调阈值。核退出码、源SHA前后、输出合法labels/finite/shape/spacing/affine、CPU/GPU内存和单例计时。需要适配器时只改接口路径/隔离，不改网络、处理算法或冻结门；保留失败。
5. **唯一全量：**smoke通过后按当次资源预算写Slurm，先held审计再释放，用afterok衔接CPU评价；不在登录节点推理、不自动retry/降batch/跳例。可复用本项目既有103预测，但必须核原权重SHA、输入/参考SHA、模型配置、输出空间和评价合同一致；无法核实则同合同重推，不静默混入旧结果。
6. **统一MET：**所有输出在同native网格交同版brats_eval，原始匹配/物理体积/解析器固定；每模型103完整，缺图/程序失败记FAILED而不是NA。定义上空区域保持官方语义，真实漏检惩罚0不当缺失剔除。
7. **统计与结果：**列ET/RC/TC/WT的病灶Dice、NSD、HD95、全部/大小F1、FP/FN、finite n、逐例覆盖；每指标paired共同有限病例，报告排除理由，不零填。新CI建议以69患者组cluster bootstrap、20,000重采样、seed20260831，保留病例等权主mean；长纵向组相关性由重采样处理，可另列患者等权敏感性。历史九路bootstrap单位是病例，不可把旧CI改叫患者组CI；需新算，主终点和多重比较校正预先冻结。
8. **验收/回收：**训练暴露明确标注；COMPLETED0:0+103产物/评分COMPLETE+报告/传输SHA联合验收。默认回收轻量CSV/JSON/日志/计时/manifest；不在本次研究中自动回收NIfTI、权重或镜像。更新本项目文档，仍不称比赛平台评分、临床验证或正式优越性。

建议主模型组合：**K1-175 + 原E-200 + 2025 METS第一名**先建三路工程横评；2026 Bio-Intel作为第二阶段完整成品列（训练暴露），K4/B175作为小病灶辅助对照。若要求独立科学结论，先解决共同未曝光队列，不要先跑一整批再解释训练泄漏。

## 6. 本次核查与未做事项

- 重核固定103身份、69组、源数据出处、九份成绩CSV/SHA、K1推理/评价配置；公开Bio-Intel五折作聚合交叉计数。
- Chrome + Playwright Bridge读取Orchestrator/HF/Hub页面和公开API；GitHub官方API只读源码，HF/Hub轻量元数据返回200。Node直连HF/Hub超时保留；浏览器正常官方请求成功，不绕登录/反爬。
- 复用10月27日TheRealBoys论文全文证据；此次未重下PDF，不将旧PDF检索UTC冒充本次下载时间。
- 未读影像体素、未读test图、未访问ECNU/VPN、未下载权重/ZIP/镜像、未安装/执行第三方代码、未加载模型、未计时/评分、未创建/释放/取消作业。
- 因此“能够接入”是条件可行性，不是本项目已成功运行；正式科学状态不变。

详细URL、固定提交/blob、大小/SHA、访问UTC、聚合成员检查与文档QA分别见[公开审计](PUBLIC_SOURCE_AUDIT.json)、[本地历史审计](LOCAL_HISTORY_AUDIT.json)、[验证记录](VERIFICATION.json)。
