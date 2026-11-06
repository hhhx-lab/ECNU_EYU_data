---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# 2025 METS第一名与2026 Bio-Intel：ECNU同fixed103横评执行计划

## 2026-11-06 用户追加执行范围

用户本次明确授权：CPU机时可能恢复后，独立提交Bio-Intel完整103推理一次；回传已完成2025的103份预测，先与K1 selected175及原比赛E epoch200比较，不等待Bio结束。原两次拒绝、2025完成作业及原计划保留。R3复用冻结推理程序/环境/模型/资源，仅绑定新attempt和授权身份；实时队列使用展开数组逐任务CPU计数，无自动重试、重装或GPU smoke。

本次允许103份2025预测NIfTI回传Mac，撤销仅对此批预测的原不回传限制，不下载原始MRI或外部模型。评分继续严格使用第6节MET协议；原E只有可信现存预测及参考身份核验通过才能作为本次统一重评列，否则明确为历史同103结果。不会自动提交额外原E GPU重推。fixed103是验证集，不是锁定测试集；本次2025/K1/E先行比较不代表原J07三模型联合门已完成。

制定日期：2026-11-04，Asia/Shanghai。用户选定两套模型并要求拟定计划；本次仅文档设计与本地核验，未启动服务器接入、资产下载、环境安装或作业提交。本计划不授权更改已有训练、G1评价、K5、监控或比赛提交。

## 1. 目标与结论边界

在ECNU服务器，对同103个真实四模态扫描分别运行两套公开成品，统一评价并与本项目K1 selected175比较。不是重新训练，不做T2W补全，不用VAE重建，不在这103例上调参/选外部权重。

主表三个模型：`K1_175`、`EXT2025_MET_GLI`、`EXT2026_BIO_INTEL_FULL`。原E-200仅列已绑定历史结果作为辅助参考；只有其现存预测/输入/评价身份完整核验后，才增设第四个统一重评列，不因名称相同混入旧E成绩。

**比较名称必须是“同fixed103公开成品描述性横评”。**fixed103来自公开2025 MET标注训练数据；Bio-Intel声明的五折中103/103各在四个训练折，完整集成明确有直接训练成员暴露。即使只取val折，54/103仍有同患者其他扫描在train；本计划不改成OOF或筛49例。2025镜像训练名单未知。本项目K1的103也是开发/选模集。因此同输入同评分不能升级为独立泛化、官方冠军排名或临床有效性证明。

完整模型的配置及所有后处理必须预先锁定。某模型某区域不支持、产物缺失或失败如实报告；禁止事后去掉弱区域、病例或调整有效分母以制造优势。原三项科学状态保持。

## 2. 比较对象与版本身份

| 对象 | 冻结选择 | 资产/配置边界 |
|---|---|---|
| 2025 METS第一名 | `MetastasesAlgorithms.BraTS25_1`，库列Maria Bancerek等第一名 | `brainles/brats25_met_gli-team`；只该镜像，不下载2025第二名 |
| 2026 Bio-Intel | 公开完整集成，Model A folds0–4、Model B folds1/3 | 七fold全部使用，不换单折、best checkpoint或本项目plans |
| 本项目主基线 | K1 selected epoch175 | 固定原权重、原推理及原MET身份，现存预测优先复用 |

2025镜像digest：`sha256:f83d914d4d7a9d316f7605abaf03fa6adbcf0e86d840d93276e86ad65b2a2a82`，匿名Hub API已核linux/amd64/public/active，压缩4,446,698,630 bytes。执行按digest，不跟随latest静默漂移。[镜像入口](https://hub.docker.com/r/brainles/brats25_met_gli-team/tags)。

Orchestrator封装按已调查提交 `f587c3f7264b89a08981b6865facc46d54842e67` 核源码/接口，不能自动升级。只用[Brain Metastases](https://brats.readthedocs.io/en/latest/guides/segmentation/#brain-metastases)，不是Pediatric/GLI/Missing-MRI。

Bio-Intel[源码](https://github.com/thekeramik05/MICCAI-METS_challenge_bi-intel/tree/83c73e0cac0b5a3c77a298547040ea3ea7f99854)提交 `83c73e0cac0b5a3c77a298547040ea3ea7f99854`；[HF权重](https://huggingface.co/keramik05/MICCAI-METS_challenge_bi-intel_model/tree/4a8291024dafb79b4c0cc381438e976a36072201)revision `4a8291024dafb79b4c0cc381438e976a36072201`，weights.zip 2,472,704,349 bytes，SHA `d308d3dd5dff4ec8fa899d64f056d8a18651b4d4e56f6618c66a2d8998e334f3`。解包后七checkpoint、两套plans/dataset均逐文件登记SHA，并核checkpoint名为作者默认`checkpoint_final.pth`。

Model B还需[BrainIAC](https://github.com/AIM-KannLab/BrainIAC)的`BrainIAC.ckpt`及配套专用trainer/backbone；公开权重包不能代替该额外构造依赖。编码器准确下载URL/revision/bytes/SHA和LICENSE仍待资产阶段核验，缺一项不得进GPU smoke。Triad初始化权重不为纯Model A推理默认另下；若原入口检查要求它，先查清推理真实依赖，不按作者训练机prepare脚本盲下两编码器。

Bio-Intel冻结作者当前默认配置：step0.5、TTA关闭；除RC外A/B各0.5，RC A/B=0.15/0.85；postprocessing开启，小组件门50个体素、NETC/SNFH/ET置信度0.6、RC0.5。完整配置、源码及后处理空间要存证，不把50体素与评价27mm³混同，不用本项目103调阈值。外层可控制的随机源预定seed20261104，作者自带随机策略如有不同必须先核清并登记，不覆盖算法语义或多seed挑结果。代码Apache2.0不自动覆盖权重；BrainIAC/Model B按非商业学术研究许可执行。

K1权重SHA：`28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204`。历史官方MET CSV SHA：`62e25bd3c6402bc8ad2d25ab9a5c7c0d71ba4b06a80a20319dafa76c7eaf65a8`。现存预测能否完整复用须实测，不能把历史成功日志当当前文件可用性。

## 3. 唯一数据合同

固定清单：[FIXED103_REAL_T2W_CASES.csv](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1/manifests/FIXED103_REAL_T2W_CASES.csv)，SHA `86ba853dadc64ffc45357895ba0bd543bf5b6b14c0c83eb789f1e875d4aa2e7e`。103扫描、69患者组、22组多时间点，均master val/authentic；本项目train患者组交集0。

输入/参考清单：[INPUT_REFERENCE_SHA256.json](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1_selected103_official_met_20261008_r1/manifests/INPUT_REFERENCE_SHA256.json)，文件SHA `5f52bfdc3a957d848063850460226a7c3d9c25f0926e827f728f21e525fbd9df`。实际515项=103×(真实四模态+正确参考seg)；在ECNU逐文件复核存在、bytes/SHA、合法标签、finite和原生几何，绑定原纠正label overlay。不得从其他目录用同名文件顶替或读manifest的合成T2W备用路径。

只访问本103输入/参考及所选模型，master val27、locked104/master test130、官方179均不读图不评分。无需重建全训练数据，不扫全账号原始MRI，不下载MRI到Mac。

### 输入与输出适配

- 本项目K1顺序T1N/T1C/T2W/T2F；Bio-Intel顺序T1C/T1N/T2F/T2W。按模态名字典显式重排，核dataset.json；不能按glob或数字通道名称猜测。
- 2025 Orchestrator用参数名传t1c/t1n/t2f/t2w。它会内部重命名到`BraTS-MET-{id:05d}-000`；必须保留原ID/患者/时间点到内部ID映射，并通过纵向病例smoke。确认该改名是否只为路径，不能假设时间点信息无关；如果影响行为，应先核原镜像接口，不偷偷改模型逻辑。
- 103全四模态存在；Bio-Intel缺T2W时用FLAIR替代的fallback必须由外层输入门拒绝触发。禁止补全、生成、图像强度人为匹配或GT派生crop。
- 各模型保留训练时plans、归一化/resample、融合与后处理，再将结果恢复原生输入/参考网格。允许必要的接口路径/通道适配，不替换算法；需要插值/重定向等未明确支持的空间修正时停下核查，不事后把结果随意resize至GT。
- 所有输出须为合法0–4整数标签，NETC1/SNFH2/ET3/RC4与本项目一致；shape/spacing/affine/轴方向及物理覆盖匹配。允许某个病例没预测RC，不等于整个模型不支持RC；若确证镜像任务不支持RC，完整四区比较hold，另立共同区域协议，不把RC填零伪装支持。
- 原始MRI只读；未知模型仅得到四模态输入，不挂GT、用户home、env或SSH key。封装需要rw/input时给独立临时副本，源数据前后SHA一致，GT仅交评估器。

## 4. 服务器布局、环境与容量

拟议独立远端根（本次未创建）：

`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_external_mets_fixed103_20261104_r1`

预期子目录：`contract/`、`assets/`、`input_stage/`、`scratch/`、`predictions/{EXT2025_MET_GLI,EXT2026_BIO_INTEL_FULL}/`、`evaluation/`、`reports/`、`logs/`、`operator/`。旧K1/E目录只读，资产只存服务器，不把外部权重或镜像回传Mac。

运行参数由独立未入Git的`.env`提供：执行根、模型/编码器路径、容器后端、TMPDIR/scratch、Conda路径、分区/QoS等；提交时导出一份无凭据配置快照和哈希。后续`.env.example`逐项中文说明获取与必要性，不复制本机服务器密码；Bio-Intel算法配置使用上述固定值，不开放事后自由调参。

### 环境门

1. 后续按本机aTrust README，经精确`ECNU_SERVER_ENV_FILE`读取原凭据；只关闭本次创建且有`VPN_SESSION_OWNED=1`的VPN，不碰用户会话。当前计划不启动VPN。
2. 一次轻量服务器预检记录UTC、获配路径实际额度/使用/余量、NVIDIA驱动、A100单GPU能力、分区/QoS/最大作业限制、容器命令与权限。历史3TiB扩容和旧余量不当本次放行证据。
3. 2025优先核现有Singularity/Apptainer后端可用性。当前封装使用sandbox/fakeroot，必须有管理员允许的namespace/subuid等能力；不装Docker daemon、不申请sudo、不关闭保护、不绕集群限制。可用Docker也须受集群许可、仅获配GPU与非特权运行；后端替代与封装调用适配另留证。
4. Bio-Intel优先独立Conda环境和冻结源码；依赖按真实推理入口核完整清单/版本，新增安装先获用户授权。不改原`segmamba`/`brats_eval`/base，不在作业里联网安装。构建独立容器只能在后端许可及新增资产预算通过后选择，不假设作者Dockerfile开箱可用。
5. 保留/复核既有评价环境Python3.10.18、BraTS-evaluation0.0.8、panoptica2.1.0、numpy1.26.4及原配置/源码SHA。若已漂移，独立恢复精确评价环境并先审批依赖；不自动升级库。
6. 未知checkpoint/pickle只在隔离作业环境加载。推理禁外网及GT/凭据访问；GPU必须核Slurm宿主/容器映射，不能固定宿主GPU0或覆盖获配CUDA_VISIBLE_DEVICES。不在登录节点做GPU前向。

### 容量门与临时产物

已知需取得的压缩资产合计6,919,402,979 bytes=6.444196GiB，仅2025镜像+Bio-IntelZIP，尚不含BrainIAC/环境/展开镜像。先以**额外150GiB作为规划预留**，不是已获配、实测峰值或固定可通过预算。

真实放行公式：新增资产所有持久副本+下载/转换同时存在副本+独立环境+四模态staging（如需）+两路103最终预测+日志/评价产物+并发运行scratch峰值+安全余量。不把已有文件算成全新，也不把旧已分配空间重复拿来用；与账号其他获授权作业的未物化预留合并核算。

先核镜像layer总展开尺寸、sandbox/SIF转换是否双存、ZIP目录展开大小/数量/路径越界、编码器体积。smoke测实际峰值再冻结`CAPACITY_BUDGET.json`；即使余量>150GiB也不代表通过，实际预算若更大需按新数字判断。

默认逐病例调用作者完整推理入口，网络fold可顺序加载，每病例完成融合/后处理和最终输出验收后只处理本次生成的可再生scratch；不保存全103×七fold概率图。该调度只改输入批次大小，不改单例模型逻辑；需要按作者实现核逐例与小batch输出等价。若入口无法按单例隔离，则按原概率生命周期重新估预算，不擅删尚需融合的分支输出。scratch清理仅限计划范围、新建清单且验收后；源数据、资产、失败及最终预测不删除。

两路GPU全量可并行，最多各单A100。若容量只够一路峰值，预算审核时改为顺序执行并记录依赖；不靠运行中抢占scratch。若连单路也不够则hold，不删旧数据、降精度或换未授权共享目录。

## 5. 分阶段执行与拟议Slurm图

### S0：资产/环境准备，尚不提交GPU

取得执行确认后，先做轻量远端接入/容量/容器预检，再按固定revision/digest与许可取得必需资产；断点下载与原完整产物分开，校验大小/hash，不执行自动“最新模型”搜索。BrainIAC链接/许可不明确或需额外安装时先通知，不用别的编码器顶替。

完成适配器、正式Slurm、环境清单和CPU边界测试：病例/通道/标签、正负空区域、27mm³边界、非1mm与非零crop、missing输入拒绝、SHA/重复/旧输出拒绝、内部ID可逆映射、GPU隔离、失败标记。源码/配置/manifest/模型/依赖和脚本形成独立运行包SHA。未实现这些脚本前，PLAN.json本身不可提交。

### S1：完整输入及模型门，两个小smoke

CPU输入/资产门逐515验收，核七fold及BrainIAC/plans/metadata；模型严格加载的CPU/GPU检查按架构放在获配作业，不用登录节点。K1已有103预测若存在，检查模型/输入/参考SHA、输出manifest、native几何及标签；丢失或不等价则本轮统一K1重评hold，先报告是否需要额外单GPU基线重推，不自动加第八个GPU作业。

smoke确定性选6–8例并冻结清单：至少有RC、无RC/空相关区域、WT小病灶、两次同患者扫描、非1mm网格（若103中有）和估计内存最大的图；某特征不存在则报告无例，不借test补齐。只用标签/几何选工程覆盖，不按模型效果挑病例、不调参数。

每路均用完整参赛/公开集成，检查源SHA、输出label/native、内部时间点映射、完整模型消费、缺T2W fallback未触发、GPU独占、网络/GT隔离、退出码、峰值显存/内存/scratch、冷启动及每例时间。固定一例重复运行检查输出稳定性；差异须定量记录并解释，不能事后挑seed。

CPU smoke门须同时消费两路`COMPLETED 0:0`、`SMOKE_VALIDATION.json`、产物SHA和正确合同。小smoke证明工程可运行，不证明全103质量或科学优越性。

### S2：全量及统一评分

两个smoke通过后，以实测最慢单例和图体积/patch数外推103耗时，另计冷启动、I/O和验收，采用至少1.5倍保守系数；估计总时长不超过获配时限85%才放行。若超过48h候选上限，不直接提交或缩减病例/fold/TTA：另提保持单例算法不变的case分片/时限调整方案并冻结新预算。

| 逻辑作业 | 拟议资源上限 | 依赖 | 当前状态 |
|---|---|---|---|
| J01 数据/资产/基线审计 | CPU8、64GiB、4h，无GPU | S0资产/环境/静态预检通过 | 未创建 |
| J02 2025完整模型smoke | 单A100、CPU8、64GiB、4h | afterok J01 | 未创建 |
| J03 Bio-Intel七分支smoke | 单A100、CPU8、64GiB、4h | afterok J01 | 未创建 |
| J04 联合smoke/容量/估时门 | CPU4、16GiB、2h，无GPU | afterok J02和J03 | 未创建 |
| J05 2025完整103推理 | 单A100、CPU8、64GiB、候选48h | afterok J04 | 未创建 |
| J06 Bio-Intel完整103推理 | 单A100、CPU8、64GiB、候选48h | afterok J04；容量不足双路则追加J05 | 未创建 |
| J07 三模型统一MET与paired/报告 | CPU8、64GiB、候选24h，无GPU | afterok J05和J06；K1预测审计pass | 未创建 |

总计拟议7个作业，其中4个单GPU作业（两smoke、两全量），不需要七张GPU来跑七fold。资源是计划上限建议，最终按实时分区/QoS、CPU内存和smoke冻结；当前没有JobID、提交成功、RUNNING或估计完成日。

**分两批创建：**S1只创建J01–J04；全部`sbatch --hold`，完整资源/脚本/路径/依赖审计后释放同一ID。S1成功后才创建J05–J07、held核验再释放；J05/06运行入口必须重读J04 pass及同SHA合同，不能只依赖调度状态。新的PATH/资源预算/源码修复需独立attempt，不原位改包。绝不把逻辑名当真实依赖JobID提交。

不在登录节点推理，不DDP、不训练、不动态装包；不沿用旧提交器的状态词解析假设。提交返回不明或已创建held先查审计，不重复sbatch。一个新失败阻止联合全量/汇总门；不为了“先出一列”自动取消其他任务或改DAG。

## 6. 统一评分与横评表

J07只在两路各103输出/manifest/COMPLETE及K1复用门通过后，用**同一个精确评价环境与源码配置**重评三路。保留原K1 CSV，生成新三路CSV；检查重评K1与历史每例72指标一致（要求相同schema/NA位置，finite值绝对容差1e-12、相对容差1e-9，不按结果放宽），不一致则比较hold查数据/空间/版本，不拿新外部数与旧K1数直接相减。并发评价worker数按最大病例内存门冻结，不默认8核即8个大体积同时评分。

冻结`config=mets`、物理体积27mm³、解析overlap_threshold0.2；原matcher DSC门1e-6与parser检出门分别保留。ET=3、RC=4、TC=1∪3、WT=1∪2∪3，WT/TC不含RC。空区域按原官方语义，完整流程空区域若有真实错误须修独立适配器与真实边界对照，不吞异常/跳例。

- 主观察指标：WT `lesionwise_dsc_mean`，是参考大病灶资格且含FP/FN惩罚的病灶Dice，不是全图Dice。按点估计可标“本开发集该指标最高”，不授予万能最佳。
- 重要次要/保护项：ET/TC/RC病灶Dice、NSD/HD95，WT全部/大小F1、FP/FN/例及小灶TP/FN；完整72列保留。RC仅15例、WT病灶Dice87例、WT小F1 46例是既有定义分母，不把NaN补0，不把真实漏检0当NA删除；全体103完成状态另核。
- 所有两两比较含原始mean、双方共同finite均值、方向化差值、n、W/T/L；高好Dice/F1/NSD，低好FP/FN/HD95，不混病例内std和组间误差。
- 新paired CI以69患者组为队列分组框架，20,000次、seed20260831。每指标先取双方共同finite病例及其G个非空患者组，重采样G组且保留抽中组的全部有效扫描，计算病例等权差值；记录G（可能小于69），不将无finite的组/病例补零，也不会制造全空重采样。另列患者等权敏感性，不用旧病例级九路CI冒充新CI。
- 两个外部模型相对K1为预定主对照，外部互比为次要；不逐72指标挑未校正显著性作胜出结论。本轮默认报告描述性CI，不以训练暴露的开发数据作正式多重检验优越性判定。
- 每列永久带训练暴露标记：K1=own_holdout_selection_used；2025=public_training_overlap_unknown；Bio-Intel=declared_full_ensemble_train_exposed_103_of_103，并说明声明split非完整训练日志法证。
- 记录A100型号/驱动、计时边界、冷启动、逐例时间、最大显存/CPU/scratch、完整fold数。各自原TTA/预处理不同是成品身份的一部分，不以“同硬件”冒称同算力预算消融。

最终一页摘要分开写“整体重叠”“小病灶检出”“RC”“资源开销”，不把G1影像SSIM混进分割排名。允许指出点估计最高者及其代价，但不选外部超参数或替换比赛基座。

## 7. 失败纪律、进度与回收

缺资产/许可、容器权限不足、配额不足、SHA/病例/通道/几何漂移、非法label/nonfinite、GPU越界、OOM/traceback/非0、错误退出或作业消失无完成证据均记hold/FAILED并通知；保留所有原日志/held/失败，不自动retry、降batch、删fold、补模态或跳病例。推理TIMEOUT不像训练checkpoint续跑；可以记录已完成病例，但未经新attempt/逐例等价/完整覆盖与用户确认不能自动接续或混入旧输出。

原包/参考/模型/既有任务只读。任何清理只针对本计划明确创建并完成消费的临时scratch；原失败、旧数据、预测、模型、环境、账号其他作业均不删不改。

本次未恢复30分钟heartbeat。后续默认用户手动查询，每次一次即时轻量快照核作业squeue/sacct、STARTED/progress/FAILED/COMPLETE、静态SHA和磁盘；不持续SSH/sleep循环轮询，不把PENDING当RUNNING。

完成验收：所有必需allocation及batch `COMPLETED 0:0`、J05/J06各103合法预测且manifest与COMPLETE匹配、J07三路309预测消费/72指标/paired合同完整、K1历史一致、没有FAILED与漂移。随后只回收轻量报告/CSV/JSON、配置/版本/hash、日志/计时/容量、提交和UTC验收；传输前后bytes/SHA核对，不回传NIfTI、权重、ZIP/镜像、全概率或大events。

预期交付：`MODEL_IDENTITY.json`、`INPUT_OUTPUT_SHA256.json`、`INFERENCE_VALIDATION.json`、各模型`leaderboard_metrics.csv`、`MEANS_216.csv`、`PAIRED_VS_K1.csv`、`PAIRED_ALL.csv`、`RUNTIME_STORAGE.csv`、`TRAINING_EXPOSURE.md`、`COMPARISON_REPORT.md`、`FINAL_ACCEPTANCE.json`。记录服务器UTC、本地接收UTC及北京时间，不修改服务器时钟或用本机时间冒充远端时间。权重/镜像及预测继续服务器登记，文档/总纲/知识与索引按阶段同步。

## 8. 当前计划状态与下一步

| 阶段 | 当前状态 |
|---|---|
| 选定两套模型、固定103及K1身份 | 本地证据已核，计划确定 |
| BrainIAC完整资产/许可、ECNU容量/容器 | 待执行阶段实测 |
| 新依赖安装/模型下载 | 未实施；安装需清单授权 |
| 适配器/运行包/正式Slurm | 未实现/冻结，不可提交 |
| 七作业held/释放、smoke、全量、评价 | 全部未开始，无JobID |
| 两外部模型新分数 | 未产生 |

下一步是用户确认计划的执行范围，然后只做S0预检与资产/环境准备；发现新增安装、编码器访问门、容量或容器阻塞先报告。S1真实smoke通过且最终预算冻结后才推进S2，不提前给全量“运行中”状态或完成时间。

前置证据：[可行性报告](../../07_references/20261104_external_models_fixed103_feasibility/REPORT.md)、[本地历史审计](../../07_references/20261104_external_models_fixed103_feasibility/LOCAL_HISTORY_AUDIT.json)、[公开来源审计](../../07_references/20261104_external_models_fixed103_feasibility/PUBLIC_SOURCE_AUDIT.json)、[九路历史报告](../../../03_experiments/13_nine_route_selected103_official_met/REPORT_20261008.md)。
