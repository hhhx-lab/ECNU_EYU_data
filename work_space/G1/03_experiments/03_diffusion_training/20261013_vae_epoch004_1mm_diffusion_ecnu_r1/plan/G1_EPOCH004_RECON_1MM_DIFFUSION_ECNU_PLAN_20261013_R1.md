# G1 epoch004重建影像训练1mm病灶Diffusion：ECNU实施方案 R1

编制：2026-10-13，北京时间。方案状态：`draft_for_user_review`，尚未执行。
研究状态：`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 1. 结论与本次审批范围

**epoch004全量四模态原生空间重建作业已完成。建议在ECNU直接使用远端重建数据，经过独立验收、严格物理1mm预处理后，训练四个独立标签条件EDM，各200,000个成功optimizer更新。** 不需要先把5150个NIfTI拉回本机，也不需要Google Drive中转。

本文件只供用户审核；编写方案不代表已经修源码、登录ECNU、预处理、提交smoke或训练。后续须得到明确执行授权；失败恢复规则需一并确认。现有重建数据、原始数据、旧150k模型、失败记录与历史运行包均只读保留。

| 建议选择 | 本方案默认 |
|---|---|
| 训练数据 | epoch004重建T1C/T1N/T2W/T2F + 原样真实seg；不使用VAE latent |
| 数据划分 | train823 / development val103 / locked test104 |
| 模型 | 当前日期改进版；四个独立单模态标签条件EDM，不是联合四模态 |
| 初始化 | 各模态随机初始化，不续用旧150k或旧错轴checkpoint |
| 预算 | 每模态200,000成功更新；合计800,000，非175 epochs |
| 资源 | 四个独立Slurm训练job，每个1张A100，能并发时并发 |
| 选点 | 20k、40k、…、200k；固定val103，按病例等权的病灶质量指标选点 |
| 科学口径 | 新多因素探索分支；不能宣称VAE、1mm或beta1单因素带来提升 |

建议200k是独立新预算，保留150k和所有预注册候选便于看学习曲线；200k不是已证明最优，也不承诺优于历史150k。若用户选择150k，应在首个正式训练job前改版冻结，不在看到val结果后改变预算。

## 2. 已有证据：哪些完成，哪些待验收

[2026-10-09终态证据](../../../03_experiments/01_vae_finetuning/20261008_epoch4_full_reconstruction_r2/operator/readonly_full_r2_20261009T085622Z/FULL_STATUS.json)记录：job`3941042`和batch均`COMPLETED 0:0`，运行19:40:15，无FAILED，运行包`COMPLETE.ok`、`DATASET_VALIDATION.json`均pass。完成marker UTC为2026-10-09T01:34:55，远端只读快照UTC为08:45:34；Slurm显示时间未自带时区，不作为UTC转换依据。

| 项目 | 已有值 |
|---|---|
| 病例 / 四模态图像 / 标签 | 1030 / 4120 / 1030 |
| 原生数据划分 | 823 / 103 / 104 |
| epoch004 VAE checkpoint SHA256 | `5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b` |
| 重建源CSV SHA256 | `0d1765e12cdad7d8a5df57a5d2c46c79414ed99a2015a4d8fed57cd4c4fc836b` |
| 重建r2计划SHA256 | `545a8d6b2fa9fe155bd1da17914734d5da9762e72e321253e82098bf8a25d83e` |
| 日期改进版ZIP SHA256 | `742a4b6a31d7d775d12e6deb0ba0cef47aed7b8e1e149ecccd58a6bf699f4b2b` |
| G2权威real-only membership SHA256 | `0a533791617d44d1ffe1afefe04b1154db2e7f8cf6db9f3421a8c1ff1aec7f9a` |

本次本地交叉核对：完成marker中的1030病例ID与[G2 real-only成员表](../../../../G2/02_dataset_release/20260919_archived_local_release/splits/splits_final_train_val_test_membership.csv)集合完全一致；成员表无重复case、无跨split患者组，组数train511/val69/test61。历史Diffusion的926成员恰等于其train+val集合；VAE quick-val20全部属于val103。**这些是本地ID级核对，不是当前远端文件逐SHA/逐体素验收。** 重建marker的split总数也不代替逐case的split一致性核验。

还未完成：独立复读全部5150文件SHA、finite、源/输出几何、原标签字节一致性、重建后病灶外观、严格1mm训练输入、A100训练/恢复及四模态生成QC。用户此前要求“先别拉回”继续有效；本方案仅轻量证据回传。

## 3. 独立目录和数据流

远端只读重建根：
`/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_vae_epoch004_full_reconstruction_20261008_r2`

真正的NIfTI输入为该根的`dataset/cases/<case_id>/<case_id>-{t1c,t1n,t2w,t2f,seg}.nii.gz`，每例还有`case_manifest.json`。不能把`dataset/`当成直接包含病例目录，也不能使用r1失败目录。

建议新的远端训练批次（执行时首次独占创建，不覆盖）：
`/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261013_r1`

对应本地证据根（执行授权后创建）：
`work_space/G1/03_experiments/03_diffusion_training/20261013_vae_epoch004_1mm_diffusion_ecnu_r1`

```text
新批次/
  plan/                 冻结MD/JSON、明确执行授权及SHA
  runtime/repository/   原始37源码快照 + 单独登记的工程补丁
  config/               四模态配置、环境清单、空间/强度/恢复合同
  manifests/            原生与1mm逐文件SHA、病例/患者组/病灶/标签清单
  evidence/             各阶段门控、独占reservation、held提交和launch审计
  preprocessed_1mm/     独立train/val数据及逐例映射，原生重建原件不动
  smoke/                合成回归、真实四模态smoke、恢复测试、可视审查
  training/<modality>/  独立Checkpoint、日志、成功/失败/受控恢复证据
  evaluation/           按checkpoint/seed/target区分的逐例结果
  selection/            四个选点摘要与模型SHA
  reports/              中文工程验收与探索性研究报告
```

只创建实际有内容的目录；不是现在建立空骨架。配置路径通过批次`.env`加载，环境实例化后冻结；env仅写非敏感运行路径/参数，服务器凭据仍只从约定登录env加载，绝不复制进实验或总纲。

数据流为：**只读原生重建 -> 独立文件/标签/背景/身份验收 -> 新1mm预处理 -> 新CSV -> 四模态训练 -> 固定开发集选点 -> 图像QC -> 独立S2接入验收**。

## 4. 开训前必须补齐的工程门

当前源码[审查证据](../../../03_experiments/03_diffusion_training/20261013_tumour_diffusion_1mm_dated_revision_readiness/README.md)仍为`not_ready_for_full_training`。工程补丁在新冻结runtime中实施并记录diff、每文件SHA和测试，不暗改用户ZIP原件，不更换EDM核心架构。

| 必做项 | 验收要求 |
|---|---|
| 消费端数据身份 | train/resume/eval/cache入口重算当前文件SHA、marker状态及空间/强度版本；marker缺失或漂移非0退出 |
| 四模态评价身份 | 每个checkpoint分别检查训练成员/CSV/data manifest/schedule/架构；不符时拒绝，不只warning首模态 |
| 患者与test隔离 | 新CSV从权威membership显式映射；任何跨组/额外病例/缺病例立即停止，test不能默认落入train |
| 精确恢复 | 保存网络、optimizer、scaler、成功更新计数、RNG、MONAI随机状态、sampler/permutation/batch位置及完整训练合同 |
| 保存与终止 | 每5000成功更新、每20000选点、150000及200000原子保存；SIGTERM仅在安全更新边界另存受控恢复checkpoint |
| 独立运行与证据 | 去掉旧nohup固定日志入口，Slurm直接调用Python；日志含attempt和job ID，存在则拒绝覆盖 |
| 目标环境回归 | Linux Bash实际dry-run/import、CUDA AMP/compile包装兼容、非finite和保存中断/恢复测试均pass |
| 评价坐标专项 | 非对称XYZ/RC、小病灶/大病灶tile、train/eval crop及per-lesion SSIM bbox一致；不能仅凭训练crop修好认为评价都正确 |

当前`compute_per_lesion_ssim`仍按Z/Y/X切窗，而CSV组件生产链标记XYZ，必须专项核对并必要时修正后才用于选点。本文件不把该未完成专项冒充新的已复现验收结果。

训练初期使用`loader_workers=0`，便于明确sampler和MONAI随机状态恢复；CacheDataset初始化线程可设4。若后续改变workers、batch、compile、loss或data，须新合同和smoke，不偷偷沿用旧attempt。不新增未经讨论的EMA、DDP或随机增强机制。

## 5. P0/P1：原生验收、mask与1mm预处理

### P0 独立原生数据验收

在CPU计算作业中逐例核验1030个manifest和5150个NIfTI：SHA、大小、finite、3D形状、四模态与seg affine/spacing一致、标签整数且在0–4；源seg与重建seg逐字节相同，并与G2纠正标签来源一致。若重建采用的seg与权威overlay不同，停止并登记，不能静默替换标签。

重算case集合、split和患者组；manifest逐例split必须与G2一致。保留实际来源路径映射，不把历史`/root/brats2026/...`直接当作ECNU可用路径；远端真实源由重建运行合同和已冻结CSV解析、查存在后绑定SHA。不得扫描无关目录寻找“差不多”的替代数据。

本阶段只做完整性检查，可核验locked104文件是否齐全，但不运行测试质量评分、不看测试曲线、不作训练调参。统计受控保留，训练和选点只获得train/val身份与文件。

### 背景和强度专项：不能把VAE非零背景当作完整脑区

epoch004重建经过`robust_normalize -> model-space VAE posterior_mean -> native恢复 -> modality postprocess`，通常为[0,1]量级，不是原扫描强度，也不是已做过1mm整例zscore。部分脑外可能残留正值；当前预处理依赖多模态非零mask，必须实测是否将脑外重建伪影纳入归一化。

建议为train/val每例生成**固定共同原始脑支持mask**：按已冻结的真实源T1N/T1C/T2F robust-normalize非零联合及hole-fill，检查覆盖seg正标签；保存mask SHA、来源SHA和算法版本，不依赖每模态重建的噪声阈值，不用val指标调阈值。重建图mask外置零写入新中间输出，四模态统一支持；不修改原生重建原件。若真实源mask无法取得或覆盖异常，P1阻塞，不能直接退回“重建图非零即脑”。

对训练集选择24个独立患者代表病例（尽量8 RC、8小病灶、8大/tiling；类别不足则停止核实，不混用val或伪造配额），逐例显示真实影像、原生重建、1mm重建、seg与mask。人工检查小灶消失/模糊、RC边界、截断、四模态支持区和明显脑外伪影；全部决定写入逐例CSV，自动finite不替代人工。

### P1 严格1mm处理

只将train823+val103共926例进入训练/开发1mm数据。locked104保留原生，待选点冻结后另行测试授权才处理用于评价。

统一四模态crop bbox、目标affine与`physical_spacing_v2`映射，spacing=1×1×1mm，保持低侧体素cell边界；整例脑mask内zscore后crop，不对每个训练crop再归一化。输出四模态float32、标签0–4；原生标签原件不变，1mm seg作为明确派生结果单独SHA。

逐例保存orig affine/shape/spacing、bbox、new affine/shape/spacing、output->input及逆映射、mask/source/output SHA、归一化统计和插值版本。不得只写一个1mm header。标签插值使用当前合法类别算法并验收，不能将整数类ID直接线性混合；记录各标签连通域/体积和小灶在重采样前后的变化。意外丢RC/小灶必须逐例判定并冻结策略，不静默减少训练分母或强行更改原seg。

重新创建CSV与实际lesion/tile计数：不能套用旧7403行/6669训练crop数量。tile全部覆盖病灶、每行ROI与label合法一致；patient-group与case/visit分别存储。采用同一四模态有效train823，valid T2W以G2 real-only合同为准；对包内266名单与权威265做差异清单，不机械多排1例，不把缺失T2W补全或重建当作真实完整病例新增。

## 6. 训练参数建议

| 参数 | 冻结建议 |
|---|---|
| 模态与输入 | t1c/t1n/t2w/t2f四模型；1扫描通道+4标签条件，BRATS_2024 |
| Backbone | Unet_NnU，time_ch_count=8，continuous embedding |
| planner / crop | planner_gpu_gb=48固定宽度；64³，1mm下64mm视野；大病灶tile |
| 训练预算 | 每模态200000成功更新；恢复累计，不重新计数 |
| 初始化 / seed | 随机初始化；共同seed=20261013，记录各模态独立RNG状态 |
| 优化器 | AdamW LR=2e-4、weight_decay=1e-5、betas=(0.9,0.999) |
| LR调度 | 沿当前代码固定LR；不额外引入175轮nnU-Net衰减 |
| EDM | sigma_data=1.0、sigma_min=0.002、sigma_max=50、rho=7、snr_shift=0；其余sigma抽样参数随源码冻结 |
| 条件 | p_uncond=0，noise_type=gaussian_tumour，n_steps=1000元数据 |
| 小病灶loss | weight=3.0、threshold=27、clamp=1.0；实际effective voxel定义随合同冻结 |
| 患者loss balance | none，不额外引入未验证权重；报告每组crop数量 |
| batch / AMP | 首选batch16、CUDA FP16 autocast+GradScaler；目标完整smoke通过后才冻结 |
| compile | 首选关闭，固定eager；不是删功能或降精度，先保障新数据与恢复合同可靠 |
| loader / cache | loader_workers=0、cache初始化4线程、cache_rate=1；单作业建议CPU8/96GB |
| 保存 | 每5k + 候选20k倍数 + 150k + final200k + 安全TERM；不删除旧checkpoint |

Beta1=0.9遵循当前改进版。若目的是严格与旧优化器比较，应先审批改为0.5或另列对照；不把0.9的科学效益当成确定事实。batch16是目标而非已验证容量；若显存/吞吐不满足，在正式训练前反馈并审批统一batch8或梯度累积等新合同，再smoke，不自动OOM降batch后继续。

建议候选日后若需要compile提速，另做目标环境完整编译、同形状反向、保存恢复验证并独立修订配置；本首批不混用eager/compiled训练路径。

## 7. P2：单A100真实四模态smoke与恢复

每模态1张A100，以同一冻结train病例子集执行500成功更新，再从完整checkpoint恢复至600；四个任务可并发，失败各自保留。不是当前包里只跑t1c/t2w的两模态smoke。

GPU smoke必须包含实际64³/batch16前后向、AMP/scaler、finite梯度、保存原子性、峰值显存、CPU内存、稳定吞吐和真实数据loader。合成中断恢复测试验证保存后的网络/optimizer/scaler/RNG与sampler下一batch一致；若要求完全bitwise，还须显式确定性算子并实测，不能由恢复字段存在推断。针对SIGTERM控制退出、OOM/nonfinite异常和不完整tmp保存分别验收。

另在24个train代表病例上做四模态生成：合法label含RC、finite、相同几何/脑支持、support外逐体素不变、无label时strict no-op；逐例保存montage和独立人工复核决定。RC边界与跨模态质量阈值在看val/test前预注册，若尚未校准标`not_formally_calibrated`，不得编造量化门pass。

四模态smoke、恢复及人工门全部通过后才能放行P3正式训练；提交成功或`COMPLETED 0:0`不足以自动科学放行。

## 8. ECNU资源、提交和受控续训

| 阶段 | 建议资源 | 时间策略 |
|---|---|---|
| P0独立原生验收 | CPU partition，经当次核验；8CPU/32GB，无GPU | 初始4h，按实际I/O估算，不在login节点处理全量 |
| P1预处理 | CPU，4病例workers/8CPU/96GB，无GPU | 初始24h，实际mask/插值压力后确认；受上限约束 |
| P2四模态smoke | 4个独立a100 job，各1GPU/8CPU/96GB | 初始各8h，实际资源不足先修合同 |
| P3正式训练 | 4个独立a100 job，各1GPU/8CPU/96GB | 每作业首选48h，但必须当次核实partition/QOS上限 |
| P4候选评价 | 各单A100，按已完成模型/候选可独立任务 | 与训练不抢同一GPU，依据候选生成量估算 |

48h不代表200k一定能完成。由P2稳定吞吐`r`成功updates/s估算每模态200000/r加cache/I/O余量，附预计GPU-hours及实际QOS可提交额度；资源批准后锁定每作业walltime。不能通过数组隐藏提交数量或绕过QOS，不能取消无关作业。

每个阶段/模态提交前：逐文件SHA、真实import、torch/CUDA版本、GPU型号、bash -n与真实dry-run、所有路径/环境变量、当前用户squeue/sacct、/public空间、上游门控均检查。显式使用`/public/home/zqchen/.conda/envs/segmamba/bin/python`作为候选已有解释器，先验证当前runtime所需API/包，不能因为历史VAE成功就认定Diffusion兼容；不足时报告并请求环境处理授权，不自动安装。

采用独占reservation -> 唯一`sbatch --hold` -> 原子记录canonical job ID/脚本/参数/SHA/资源/日志路径 -> 核对上游门 -> release。P3只在上游验收后提交，不提前创建等待的正式训练jobs。QOS拒绝保留回执、结束本次操作，不循环重提。每job编号唯一登记；无canonical ID不得靠空旧receipt推断。

Slurm直接在新模态工作目录调用冻结Python，使用调度器给的CUDA_VISIBLE_DEVICES，不硬写宿主GPU编号；每job单GPU、非DDP，不调用旧四GPUnohup launcher。提前创建真实日志目录，绝对stdout/stderr路径必须指向本attempt，禁止指向旧r1/r2或其他实验。

### 超时与失败区别

在用户批准本方案所列**受控超时恢复策略**后，可为计划内TIMEOUT/已记录安全TERM创建独立续训job：先冻结原job终态和完整checkpoint审计，以同预算恢复至累计200k，不重投旧脚本、不覆盖旧attempt；runtime/data/优化器/增强合同不变，新launch绑定原job与resume SHA。

网络/optimizer/scaler/current successful_step/sampler/RNG、LR、数据SHA、loss/空间合同任何缺项或漂移，禁止续训。任一OOM、nonfinite、CUDA异常、ImportError、非预期非0退出或作业消失无证据，立即停该分支并通知，不自动建“下一个attempt”。合法其他模态继续，不动无关作业。若用户只批准首次提交而未批准受控恢复，则TIMEOUT后也先报告等待授权。

## 9. P4/P5：选点、双目标报告、图像QC

开发集只用val103，test104绝不参与选点。将VAE选点用过的quick-val20标注在报告中；共同开发集二次选择及报告只能exploratory。

固定候选每模态20k至200k的10点；每点EDM-Heun18、CFG=1、churn=0，预注册4个生成seed（20261013–20261016），按case/lesion/modality/seed稳定派生随机流。冻结数据/代码/评价版本，每点保持完全相同病例和噪声序列，不用一条随机样本波动选“最佳”。

**选点目标**：对VAE-reconstructed val图的per-lesion bbox SSIM（在专项坐标通过后使用），病例内体积加权、病例间等权，再对4seed取均值；每模态单独最大化，容差1e-6内取最早step。主选择集合为预注册阳性病例，no-lesion病例记NA及no-op，不补0混入均值。没有完整合法分母或任一hard QC failure时不能选点，不能静默删难例。辅助报告tumour masked MSE/MAE/PSNR、大小病灶和RC/ET/NETC/SNFH分层，不能仅凭全脑背景SSIM选点。

同时报告**authentic real val影像目标**：使用相同mask/物理grid、分别冻结归一化，以同位置/label比较生成与真实图像，显示VAE平滑/细节丢失与真实目标偏差。真实目标是预注册次要诊断，不看过结果后换主选点规则；冻结选点后不得再根据真实目标偷偷挑另一checkpoint。仅匹配VAE目标更好不等于更接近真实影像。

SSIM等对应条件随机生成与单一真实图像不是分布质量的完整度量，另做可视病灶形态、边界、跨模态一致性与support事务QC。最终四checkpoint集合需一起通过train代表病例和固定val质量审查，原G2旧150k gate不转移到新权重；未经人工复核不填写operator approve。

若做同病例模型比较，输出原始均值、共同患者组paired均值/差、95%CI、W/T/L、未校正p及分母；patient_group为bootstrap独立单位，保留组内访视，20000 bootstrap、seed=20261013，SSIM高优/MSE低优，不把多个生成seed当独立患者。缺值须写原因和配对集合，不单列差值。历史旧空间、旧预算、旧beta和真实图训练共同改变，比较只能描述整体分支差异，不能归因VAE。

本轮不自动加训四个真实图1mm对照模型。若要回答“VAE重建数据本身是否有优势”，另预注册同代码/1mm/优化器/budget/seed的real-only匹配对照，并明确追加四模型成本；旧150k模型不构成纯数据因素对照。

locked104的数据虽已重建，但直到checkpoint/选择和评价合同冻结、且另获测试授权后才做一次受控评价；不把该锁定集合或原VAE全量重建完成当正式优越性证明。

## 10. 交付、监控与后续S2边界

每阶段完成拉回轻量证据：计划/config/env.example非敏感模板、source/数据/空间manifest、审计、stdout/stderr、smoke montages/逐例审查、模型SHA/选择摘要、逐例及汇总JSON/CSV/MD/ok。默认不拉NIfTI、checkpoint或巨量events；无Google Drive上传。需要权重或数据后另行确认。

执行授权后才建立30分钟heartbeat；每次只做一次即时快照，不sleep/循环轮询/持续SSH；PENDING/RUNNING无异常保持安静，阶段变化、失败、需要决策或完成才通知。凭据只从`/Users/hwaigc/比赛+课题/服务器登陆/服务器登陆的env`加载；每次显式启用`ecnu-vpn-split.sh`，结束只关闭本次VPN，不暴露secret。不安装依赖、不Git、不ZIP/Docker、不平台提交、不动旧失败。

工程最终验收：4个模态累计200000成功updates、唯一历史链、完整网络/optimizer/scaler/合同SHA、合法选点、四模态生成QC、轻量证据回传逐SHA通过；记录本机和远端UTC并给中文总结。完成后关闭本方案heartbeat。每次状态或研究语义变化同步G1专题、知识登记、更新记录并运行Conda base catalog build/check。

新模型**不自动替换S2现有增强器或原E**。另行验证1mm与S2 `[1,0.8593999743,0.8593999743]`的轴序/物理尺度、强度及原空间映射，四模态/RC质量与support外不变、推理延迟；然后另立分割训练计划和同预算比较。G1图像质量、offline补全、online增强、正式比赛结果不能混成一条“提升”。

## 11. 执行清单与待用户确认

| 阶段 | 进入条件 | 当前状态 |
|---|---|---|
| 本方案 | 本地历史证据、源码审查、用户需求明确 | 已写，待审核 |
| P0 | 明确执行授权、候选环境兼容、CPU资源 | 未执行 |
| P1 | 独立原生验收+源mask/成员身份pass | 未执行 |
| P2 | 工程补丁/1mm/CSV静态门与SHA冻结 | 未执行 |
| P3 | 四模态GPU/恢复/自动/人工门全部pass | 未提交 |
| P4/P5 | 候选完整、评价身份与坐标门pass | 未执行 |
| S2/locked104 | 单独冻结方案与授权 | 不在本轮自动执行范围 |

建议用户确认：按本方案200k/四独立A100/eager/batch16目标/beta1=0.9执行；授权在新runtime补齐工程门和独立CPU/GPU预检；是否同时授权在安全完整checkpoint与合同不变时对计划内TIMEOUT作独立恢复。审批若调整任何项，先修订R2并冻结SHA，再提交，不在已发出的作业上改参数。
