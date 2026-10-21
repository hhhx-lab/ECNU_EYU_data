---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# K1小病灶改进：修正后的分阶段执行版

日期：2026-10-13，Asia/Shanghai。用户已在本会话同意“把这三处完善之后执行这个方案”。本文件承接10月13日用户修订，授权执行P0及符合后续门控的独立路线；不跳过诊断直接训练，不授权K5，不更改旧实验。工程执行授权与上方科学状态字段不是一回事。

## 1. 版本与目标

- 10月9日原稿SHA：`0adb1b2aaf69acd41b0a747953f46454100782c05cc3e36fe9293f87ed0e2df6`。
- 10月13日用户修订原件SHA：`0d9ac0379062f82d1c39533799796a2235ebf2e7aff4aa6f4bd94cc1449fd653`。
- 两份原件与`REVIEW_20261013.md`永久保留；发生冲突时以本执行版的方法合同为准，旧稿中的理论上限/共同漏检推断不用于执行。
- 实验入口：`S2/03_experiments/14_k1_small_lesion_refinement/s2_k1_small_lesion_p0_20261013_r1`；远端同名批次放在`S2/experimental_unvalidated`。各远端阶段独立审计、排他reservation、held提交；失败后不覆盖或自动重提。

目标是保持K1整体病灶表现与FP控制，同时定位并改善WT小灶召回。不是证明生成器或nnU-Net某一环节有罪，也不把九路多因素/异预算比较视为单变量因果实验。

## 2. 三项必须贯彻的修正

### 2.1 标签往返只是固定管线表示保真诊断

原始硬标签按冻结nnU-Net预处理重采样，再转one-hot并按冻结概率导出路径返回原网格，用官方MET评价。它衡量该确定性标签管线的保真度，不是所有1mm网络的理论召回上限。每个原始参考实例归类为：

1. `disappeared_in_label_pipeline`：在该硬标签管线中消失。
2. `survived_roundtrip_match_failed`：保留，但往返官方匹配不通过。
3. `survived_roundtrip_match_passed`：保留且往返匹配通过。

另记映射不确定/合并/分裂等状态，不硬塞入上述三类。各类比例只用于定位脆弱环节；不称“绝对不可达”，不据此断言S/L/P无效，不用第三类数量限制一切训练的提升空间。插值、软标签、输出概率和真实模型可能与one-hot硬标签行为不同。

### 2.2 共同漏检与新/失检必须有实例身份

46例当前共有286个WT小参考灶；K1/原E/K4/B175/D50分别TP=46/52/52/53/58，各自漏检约80%。这些计数不证明约80%是共同漏检。K4净多6个TP也不等于明确新检出6个、无失检。

逐路线官方JSON若只有体积和列表位置，没有稳定ID，先只做独立计数核对；不得按等体积、列表序号配对。由原始完整参考标签建立稳定实例ID，记录region、原始geometry、标签SHA、组件掩码SHA及确定性扫描键，之后为每一路用同一官方匹配器生成reference-ID检测矩阵。矩阵完备后才能统计交集、新检出/失检/共同检出及其尺寸、spacing和往返分类。

K1召回-FP曲线覆盖其他路线工作点只说明决策差异与观测相容，不证明表征相同或收益完全来自阈值。

### 2.3 小灶资格固定在原始完整参考标签

以冻结官方提取器的原始完整region组件及官方volume<27判定小灶。首先核实volume单位、spacing处理、连通性，不未经核验就把27当作所有空间的体素数或mm³。WT=1/2/3，TC/ET/RC以官方config为准，RC=4不能并入WT。

每个原生实例保留原始体素数、物理体积、spacing/affine、region、patient-group、稳定ID。对实例掩码及metadata同步施加预处理、裁剪和空间增强，记录消失、裁掉、截断、合并、分裂。不能重算局部crop/下采样组件体积后，把大灶碎片变成小灶。生成实例单独记录供体ID、原始资格、合成ID和生成/插入变换链；资格未知的生成灶不能默默纳入小灶loss或配额。

## 3. 冻结输入与比较边界

锚点K1 selected175 SHA：`28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204`；原E SHA：`4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`。

固定train=1035/development=103、patient-group split及seed20260724。父plans SHA `c20ac311f0b3db0f0710e98b0b56e65e8bb38c13b95094b6d6f9966ac529ffa5`，3d_fullres spacing1mm、patch128³、batch2；原始影像与G1空间不能由此推定。每阶段绑定515份fixed103输入/标签SHA、split/mapping、dataset.json、checkpoint、运行代码、官方MET及nnU-Net安装版本和函数源SHA。

训练候选S/L各自从原E网络权重独立175轮，保持原K1在线增强/G1/QC/学习率/batch/patch，不从K1-175继续加预算。每25轮留点，25/50/75/100/125/150/175按fixed103 ema_fg_dice最高、并列取较早epoch。若新实现不能复现冻结K1行为，先建立匹配无干预控制路线，不以旧K1直接冒充新框架控制。

## 4. P0执行与逐阶段验收

P0不训练、不写旧目录。先完成本地官方明细核对和一次远端只读输入/软件/预测/概率可用性核验；真实资产核验后才能部署CPU/GPU诊断。不得因为代码能编译就宣称重采样和官方匹配合同通过。

| 阶段 | 动作 | 必须输出与门控 |
|---|---|---|
| P0-A | 复核既有K1/K4/原E/B175/D50明细 | 103同病例、region参考组件计数/体积分布一致、官方27/.2边界、TP/FN/FP与CSV一致；没有ID时明确identity unresolved |
| P0-B | train-only实例清单及fixed103身份 | 原始完整标签SHA/geometry、提取器/region/连通性、稳定ID，缺失/重复/身份碰撞拒绝；不能读取locked数据调参 |
| P0-C | 冻结硬标签管线往返 | 使用真实nnU-Net reader、transpose/crop、resample_seg/probabilities、label manager/export；原图geometry与515 SHA一致，one-hot合法，官方MET和逐实例状态；不是理论上限 |
| P0-D | K1/K4/原E等逐实例错误 | 同一reference-ID检测矩阵，官方匹配>=.2，TP/FN/FP与既有MET一致，报告intersection/new/lost及不可判定数 |
| P0-E | K1-175/150概率诊断 | 各checkpoint SHA/CPU加载，单A100非exclusive、sequential_in_process、workers0；冻结工作点扫描、空间与ROI，在同fixed103评价，不替换原预测 |
| P0-F | 曝光、深监督、梯度与G1谱系 | 旧events不足就标historical exposure unavailable；新短程诊断不冒充旧175轮曝光。标签体素占比只是proxy；loss机制需真实逐实例梯度测量。G1代码/CSV/checkpoint谱系单独核查 |

每阶段结束拉回轻量manifest/evidence/JSON/CSV/MD/ok/log；默认不拉checkpoint/NIfTI/events JSONL。概率与中间标签留远端独立诊断根。记录本地/远端UTC、hostname、磁盘、squeue/sacct、stdout/stderr、STARTED/FAILED/VALIDATION/COMPLETE与关键错误；空squeue的已完成作业由sacct和成功证据确认。只做即时快照，不sleep/循环轮询/持续SSH。

## 5. 预注册诊断规则

下列数值在查看新P0结果前冻结，只是探索性诊断描述，不是科学门或无效性证明。软件/空间测量语义若与实际不符，记录blocked contract并新建修订，不原位改阈值。

- 往返匹配通过比例：<0.35记低，>=0.55记较高，中间灰区；不限制训练收益上界。前两类中某类>=0.65记占主导，但不称“绝对不可达”。
- WT概率ROI：原始完整实例掩码随真实空间变换后在训练空间外扩1体素；分别报告原掩码内和扩张ROI内max，避免附近大灶混淆。消失实例需几何映射ROI，不能当空数组概率0。
- 弱响应：FN中max WT概率处于[0.25,0.5)的比例>=0.40，且175/150均成立；低响应max<0.10比例>=0.55。两种亚群可并存，不能当互斥机制。
- 概率扫描global WT阈值0.10/0.15/0.20/0.25/0.30/0.35/0.40/0.45/0.50，额外保留原多类argmax预测基线；新WT体素按1/2/3中最大概率分配，label4与WT竞争规则须在GPU合同中先锁定，不静默覆盖RC。体积分档变体在读取曲线前另行预注册，不能边看结果边搜索。
- 同召回FP诊断：B175/D50/K4至少2个工作点与K1扫描相容，或FP增加<=0.10/例、小灶micro recall增加>=0.04；仅解释为可进一步检验P。
- 曝光proxy：完整含资格小灶patch比例<0.15，或诊断采样推算每实例175轮次数中位<20，提示S；推算必须给采样次数/置信范围，不能写成真实历史曝光。
- 大灶主导proxy：小灶实例数占前景>=0.30而体素占比<0.03，只提示测梯度；L优先级要结合实际loss/梯度与保护项，不能凭proxy证明梯度稀释。

1,296例物理统计和“G1约80%错crop”仅是来稿假设来源，未本批验证，也不能绑定K1实际runtime。若G1错轴被冻结训练谱系证实，另立G诊断，不与S/L/P混改。

## 6. P0后的候选与门控

P0先定位，允许不同亚群不同机制。P成本低但不训练；需要训练的S/L同一时间先启动一个，不能通过往返类别直接排除任何候选。没有定位证据则hold，记录原因；不为了“执行”而强制长训。

- S：只改train-only实例采样，初始候选25%小灶patch配额；原随机/前景采样与K1其他参数不变。提交前锁定实例资格/变换链/实际曝光，不按fragment重分类。
- L-a：最高分辨率局部WT BCE，额外权重上限2、系数0.2。
- L-b：小灶实例+适量背景的Dice/BCE逐实例归一平均。与L-a择一，先验证finite、梯度、背景/大灶/RC保护；原实例身份同步变换，不由变换后小组件重建资格。
- P：固定175，仅改变一项推理决策或概率导出；开发集扫描不能当独立验证。另冻结未用于此轮调参的独立病例验证合同，未获独立数据时只能exploratory。
- 标签保留A/更细spacing B/高分辨率分支C：先做表示保真及影像信号诊断，不能称理论上限。A不得把大灶fragment当微小灶；B须实测网络形状/权重兼容性，不武断否定原E初始化；改变plans/预算后另立计划。D为阶段停止、保留证据，不宣称小F1上限已确定。
- S+L或结构/G1组合必须在单变量结果明确后独立设计，不归因给单机制。

## 7. 报表、筛选及完成边界

同时列双方官方原始mean、complete-case paired mean、方向化差值、有效n、W/T/L、95%CI和未校正p。病例级paired bootstrap20000、seed20260831，不能把同一病例多个病灶当独立病例；稳定ID病灶比较另列身份矩阵和病例聚类不确定性。

WT小F1及TP/FN主读数；全部WT FP、预测组件尺寸、WT病灶Dice/全部F1/大F1、ET/TC/RC病灶Dice和RC有效n保护。训练前候选初筛合同需冻结：小F1点差>=+.02、WT Dice/大F1>=-.01、WT FP<=+.10/例；全部F1及ET/TC/RC保护值须在训练合同预注册，未冻结不得提交。+.02不等于显著性，不能修改阈值追通过。

阶段工程验收pass不等于正式科学优越性；同fixed103用于选checkpoint和报告，仅exploratory。不能因K1均值最大就宣称全面优越。P0完成后形成选路记录；175长训需通过对应合同/技术预检及诊断门，不提前伪报开始或完成。

ECNU凭据只从约定env加载，单次显式start分流VPN、finally仅stop本次VPN。不得安装依赖、Git、Synapse上传、改写旧数据/失败/runtime。每次状态/语义变化更新总纲专题、知识登记、更新记录并执行catalog build/check。
