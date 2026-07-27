# BraTS 2026 Task 1 最终全流程执行与验收总控手册

> 文档版本：v2.7
> 状态快照：2026-07-26 CST
> 项目根目录：/Users/hwaigc/比赛+课题/ECNU_EYU_data
> 文档性质：Task 1 的唯一总控入口。模块文档负责实现细节；本文冻结数据、模型选择、阶段依赖、放行门和发布边界。
> 本次更新：同步 2026-07-26 的双 H20 实际验收与 Dataset264 搬运状态。以 E 为唯一 Route A 父 checkpoint，结合 BraTS 2025 的 train-only 在线调度原则与已冻结 G1 四模态 Diffusion，完成 Route A 原子事务、nnU-Net 2.8 接口、Gate 1/2、配对 control、真实训练路径 smoke 和训练/推理隔离；双 H20 UHost 已连通，运行时资产 SHA256、独立 Conda、Torch/CUDA、`pip check`、`rsync` 和远端 73/73 测试已验收。ECNU 分流 VPN 已打通，Dataset264 正由 ECNU 后台直传 H20；真实 Gate、smoke 和配对训练仍未启动。

---

## 1. 先读结论

### 1.1 当前已冻结的事实

1. Dataset264 已通过缓存和运行时验收，固定 split 为 train/validation/locked test = 1035/103/104。
2. G1 missing-T2W completion 与 G2 completion QC 已完成。265 个 completion 病例中，212 个仅进入训练，53 个仅作 evaluation（27 validation + 26 locked test）。
3. S2 的 B、A-1、E、A-1+E 均已训练至 200 epoch，并在同一固定 103 例上完成 BraTS-evaluation mets 官方兼容病灶级评估。
4. E（Focal CE）是后续小病灶研究的唯一冻结基座；B 是整体分割的保守基线。A-1 与 A-1+E 只保留为审计证据，不再派生后续实验。
5. E 仍存在明确的小病灶瓶颈。原始 E 只作为部署锚点；增强因果效应必须比较同一 E checkpoint 派生的 `E-continue (p=0)` 与 `E+MET-AUG-A (p=0.20)`。S2-DS-D 是后备路线，不再阻塞 G1->S2 样本增强。
6. G1 四模态 150k checkpoint 和 G2 full94+9 parent gate 已完成。Route A 的 train-only 组件合同、原子事务、nnU-Net 2.8 adapter、Gate 1/2/approval、配对训练合同和联合显存/吞吐 smoke 已实现；本地项目环境 73/73 测试通过。
7. 新 UHost 已验收 2 张 95.1 GiB H20、44 CPU 线程、472 GiB 内存与约 915 GB 可用根盘；容器无 Slurm，因此 `.slurm` 文件保留 SBATCH 头以便迁移，当前在 UHost 上由 Bash/PID/日志锁直接运行。运行时代码、四个 G1 checkpoint、G2 gate/mapping 和 E checkpoint 已通过 SHA256；项目独立环境固定 `torch 2.7.1+cu128`、`nnunetv2 2.8.0`、`numpy 1.26.4` 和 `monai 1.5.1`，远端 73/73 测试、双卡 CUDA 实算与 `pip check` 通过。Dataset264 的 raw 32GB 与 preprocessed 68GB 已从 ECNU 启动两路可续传直推，目标为 `/root/brats2026/data/s2_dataset264`；完整数据门与真实 Gate/训练尚未通过。

### 1.2 当前不允许执行的动作

除非用户在当时明确授权，下列动作均不得启动：

- 在线 Diffusion 或任何 MET-AUG 路线的训练、推理调用、写入真实训练 patch。
- 官方 179 例推理、ZIP 打包或 Synapse 提交。
- 重训已经完成的 B、A-1、E、A-1+E。
- 删除 Dataset264 cache、checkpoint、失败作业日志、103 例预测或现有评估产物。
- 以旧 G1 代码、旧 label_pool.csv 或历史 Diffusion 结果绕过新的 train-only manifest 与 Gate。

本文件描述后续允许怎样设计、验证和放行；它本身不是任何训练、扩散、179 例推理或提交授权。

### 1.3 现行模型职责

| 用途 | 冻结选择 | 说明 |
|---|---|---|
| 小病灶后续基座 | E / Focal CE | 当前固定用于 MET-AUG-A；不得被未选择的实验替换。 |
| 整体分割保守对照 | B / 原版 RC-aware Dice + CE | 用于整体 DSC/NSD、FP 风险和发布前回退比较。 |
| 增强因果对照 | E-continue / p=0 | 与 Route A 同父 checkpoint、训练预算、seed 和标准增强；尚未训练。 |
| 当前主候选 | E+MET-AUG-A / p=0.20 | 代码和 Gate 工具已完成；真实 Gate、训练和 103 例结果尚未产生。 |
| 二阶段后备 | S2-DS-D | 只有 Route A 结论后仍有明确小病灶瓶颈且时间允许时再设计/运行。 |
| 后续增强路线 | MET-AUG-B 至 MET-AUG-F | 仅保留分期规范；A 未通过前不得提前实施。 |
| 官方发布候选 | 未冻结 | 只有完成后续选择并获得用户授权后才可定义。E 不是自动的 179 例提交模型。 |

---

## 2. 命名、范围与边界

### 2.1 名称去冲突规则

旧总计划曾将 completion + online Diffusion 与当前 Deep Supervision 使用同一个单字母代号。该冲突名称现已禁用。

| 名称 | 含义 | 当前状态 |
|---|---|---|
| B | Dataset264 completion-only 原版 RC-aware Dice + CE | 完成，整体保守基线。 |
| A-1 | 缩减网络 stage 的架构消融 | 完成，拒绝为后续基座。 |
| E | 原版 6-stage 架构上的 Dice + Focal CE | 完成，冻结为小病灶基座。 |
| E-continue | 从冻结 E 继续 200 epoch、无合成增强的配对 control | 已实现，未训练；只用于 Route A 因果归因。 |
| A-1+E | A-1 与 E 的组合消融 | 完成，拒绝为后续基座。 |
| S2-DS-D | E + 正确的 Deep Supervision loss 权重 | 后备设计，未启动，不阻塞 Route A。 |
| MET-AUG-A | 真实单病灶组件 + G1 四模态 Diffusion + E | 桥接代码完成；真实 Gate/训练未启动。 |
| MET-AUG-B 至 MET-AUG-F | 转移瘤合成增强的后续受控路线 | 仅方案审校，A 通过后才按序考虑。 |
| MET-AUG-EF | 仅在 E、F 各自通过后才可单独立项的交互实验 | 未设计为默认路线。 |

任何未来 run、目录、trainer、报告和提交包都必须使用上述命名。不得再次出现含义不明确的单字母旧代号。

### 2.2 数据边界

- Dataset264 的训练、validation、locked test 身份和病例组隔离不可改写。
- 固定内部 validation 103 例有真值，只用于 checkpoint 选择、正式内部评估和风险复核。
- locked test 104 例不用于训练、调参、生成供体或模型选择。
- 官方 179 例无公开 seg，不进入训练、G2 synthetic source、checkpoint 选择或任何增强 Gate。
- validation、locked test、官方 179 推理和提交阶段均不得调用在线 Diffusion。

### 2.3 执行环境边界

- 训练、评估和文档脚本使用项目约定的独立 Conda 环境；禁止 sudo pip，禁止混用系统/Homebrew Python 与项目 Conda/uv 环境。
- 已验收的独立评估环境为 /public/home/zqchen/.conda/envs/brats_eval，固定 BraTS-evaluation 0.0.8、panoptica 2.1.0、numpy 1.26.4。
- UHost 上的 S2 训练环境固定为 `/root/brats2026/envs/s2_met_aug_h20`，从镜像 `/usr/local/miniconda3/envs/py312` 克隆。因 nnU-Net 2.8 明确排除镜像的 torch 2.9.*，项目环境单独固定 `torch 2.7.1+cu128 / torchvision 0.22.1 / torchaudio 2.7.1`；不修改基础 `py312`，不使用 `sudo pip`。
- UHost 项目根为 `/root/brats2026/ECNU_EYU_data`，数据根为 `/root/brats2026/data/s2_dataset264`，运行根为 `/root/brats2026/runs/s2_met_aug_route_a_20260725`。每个 trainer 只可见一张 H20；两臂可各占一张卡，不使用 DDP。
- 本机记录 Git 时使用 /opt/homebrew/bin/git。源码快照必须同时记录 HEAD、worktree 状态、实际部署文件 SHA256 与环境版本；仅记录 Git HEAD 不足以标识 dirty worktree。
- 不在文档、脚本、Slurm stdout/stderr 或归档中写入密码、token、私钥或明文连接凭据。

---

## 3. 冻结数据、标签与评估合同

### 3.1 病例身份与 split

| 数据层 | train | validation | locked test | 合计 | 使用规则 |
|---|---:|---:|---:|---:|---|
| master | 1035 | 130 | 130 | 1295 | 身份与 patient-group 总口径。 |
| real-only | 823 | 103 | 104 | 1030 | 真实 T2W 可直接用于 S2 的病例。 |
| completion | 212 | 27 | 26 | 265 | 仅训练 split 进入 S2 梯度。 |
| Dataset264 | 1035 | 103 | 104 | 1242 | 当前 S2 的唯一固定数据合同。 |
| 官方 validation | 0 | 179 | 0 | 179 | 无公开 seg，仅在最终获批发布时读取。 |

Dataset264 物化和缓存仍应满足：

~~~text
imagesTr = 4552 = (1035 + 103) x 4
labelsTr = 1138 = 1035 + 103
imagesTs = 416  = 104 x 4
labelsTs = 104
included_cases = 1242
completion_paths_overridden = 212
~~~

labelsTr 为 1138 不表示 1138 例都参与梯度；实际训练/validation 身份始终由固定 1035/103 split 决定。

### 3.2 通道、标签与空间合同

~~~text
0000 = t1n
0001 = t1c
0002 = t2w
0003 = t2f

0 = background
1 = NETC
2 = SNFH
3 = ET
4 = RC
~~~

预测只能输出整数标签集合 0、1、2、3、4。任何新 trainer、合成适配器或发布脚本都不得改变该通道或标签语义。

官方兼容 mets 区域定义为：

~~~text
ET = label 3
RC = label 4
TC = labels 1 + 3
WT = labels 1 + 2 + 3
~~~

RC 单独评分，不并入 TC 或 WT。

### 3.3 小病灶术语不能混用

G2 图像与生成 QC 使用：

~~~text
tiny  = volume < 27 mm3
small = 27 mm3 <= volume <= 275 mm3
large = volume > 275 mm3
~~~

官方 mets 解析器本轮只输出 small_instance 与 large_instance，阈值为 27 voxels；它不输出独立 tiny 指标。Dataset264 为 1 mm 等距预处理时二者数值单位相容，但报告中仍必须写明指标来源：

- 不得把 small_instance 改名为 tiny。
- 不得凭 G2 分层伪造官方 tiny F1。
- RC small-instance 可比较 reference 仅 3 例，不能用其单项均值单独决定模型。

### 3.4 固定内部正式评估合同

| 项目 | 冻结值 |
|---|---|
| 评估病例 | 固定 validation 103 例；预测、reference、source ID 一一对应。 |
| 评估器 | BraTS-evaluation 0.0.8 + panoptica 2.1.0。 |
| 配置 | mets。 |
| 体积阈值 | 27 voxels。 |
| overlap 阈值 | 0.2。 |
| 完整性 | 每个候选 prediction/reference 均为 103/103，missings=[]。 |
| 必报指标 | lesionwise DSC/NSD、all/small/large instance TP/FP/FN/F1、103 例 paired delta。 |
| 风险复核 | RC、small/large、FN/FP、改善最大与退化最大病例。 |

训练中 nnU-Net summary 的 foreground Dice 不是正式病灶级指标，不能替代本合同。

---

## 4. 已完成状态与证据

### 4.1 G1/G2 completion 历史状态

| 项目 | 状态 | 保留的解释 |
|---|---|---|
| missing-T2W Stage 6 | 完成 | 265 例 completion 产物已归档。 |
| G2 completion QC | 完成 | 212 train + 53 evaluation；无 pending、needs_regeneration 或 rejected。 |
| 重点人工复核 | 完成 | 47/47 为 pass_technical_visual。 |
| Stage 5 裁切风险 | 已保留 | 初始 FINAL_GATE 曾因固定中心裁切风险为 reject_and_retune；后续是具名 operator override 加 Stage 6/G2 复核放行，不得改写为“自动通过”。 |

这些证据说明 completion 数据可用于当前受控分割实验，不代表生成 T2W 与真实 T2W 临床等价。

### 4.2 S2 完成作业与无效历史作业

| 类别 | 作业或阶段 | 处理 |
|---|---|---|
| 有效 A-1 | 3128521；正式评估 3154513 | 200 epoch、103 预测和审计完整。 |
| 有效 E | 3141629；正式评估 3154514 | 200 epoch、103 预测和审计完整。 |
| 有效 A-1+E | 3141630；正式评估 3154517 | 200 epoch、103 预测和审计完整。 |
| B | 既有 completion-only 完整 checkpoint 与正式评估 | 作为固定整体分割对照。 |
| 旧 r2 | 3124192/3124193/3124194 | epoch 0 前因不完整 nnU-Net 运行时失败；无 checkpoint/预测，不计入。 |
| 旧 r3 | 3125426/3125427/3125428 | 错误 Slurm workdir 或运行前取消；无 checkpoint/预测，不计入。 |

失败记录是溯源的一部分，必须保留；不得重启、重命名为结果或混入任何比较表。

### 4.3 固定 103 例的正式选择结果

#### WT 总体病灶

| 指标 | B | A-1 | E | A-1+E |
|---|---:|---:|---:|---:|
| lesionwise DSC | **0.625729** | 0.580821 | 0.587819 | 0.578306 |
| lesionwise NSD | **0.621359** | 0.569390 | 0.585794 | 0.567496 |
| all-instance F1 | 0.710078 | 0.683197 | **0.712277** | 0.657227 |
| small-instance F1 | 0.277048 | 0.292404 | 0.333083 | **0.342887** |
| large-instance F1 | **0.683899** | 0.662586 | 0.672686 | 0.660221 |
| all FN | 3.116505 | 2.873786 | 2.912621 | **2.834951** |
| all FP | **0.747573** | 1.097087 | 0.970874 | 1.310680 |

#### RC

| 指标 | B | A-1 | E | A-1+E |
|---|---:|---:|---:|---:|
| lesionwise DSC | **0.423506** | 0.421768 | 0.377022 | 0.380900 |
| lesionwise NSD | 0.363712 | **0.386169** | 0.273118 | 0.332156 |
| all-instance F1 | 0.236232 | 0.248505 | **0.421053** | 0.194070 |
| small-instance F1 | 0.000000 | 0.000000 | 0.000000 | **0.166667** |
| large-instance F1 | 0.085437 | **0.103098** | 0.080906 | 0.086916 |
| all FN | 0.067961 | 0.067961 | 0.116505 | **0.058252** |
| all FP | 0.776699 | 0.611650 | **0.038835** | 0.689320 |

### 4.4 当前选择的解释

- 选择 E：WT all-instance F1 为 0.712277，WT small-instance F1 为 0.333083，RC all-instance F1 为 0.421053；RC FP 显著降至 0.038835。
- 保留 B：B 的 WT lesionwise DSC/NSD 最高，且 WT FP 最低。E 不能被表述为“整体分割全面替代 B”。
- 拒绝 A-1：总体 WT DSC/NSD/F1 下降，WT FP 增加，RC 收益不足以抵消风险。
- 拒绝 A-1+E：虽然 WT small-instance F1 最高，但 WT/RC 主指标下降且 WT FP 最高，未证明组合净收益。
- E 的未解风险：RC lesionwise DSC/NSD 为 0.377022/0.273118，RC small-instance F1 为 0；其 RC precision-recall 交换需要由 S2-DS-D 和逐病例复核继续验证。

共同高风险病例和 E 相对 B 的逐病例变化见第 5 节中的风险复核归档。

### 4.5 G1/G2 已冻结的生成器父证据与适用范围

当前 G1 生成器不是“尚未训练的概念模型”。四个模态 checkpoint 已在固定 cohort 上通过 G2 的完整技术与人工复核，以下文件是后续任何受控 Route 的父证据：

| 父证据 | 已冻结事实 | 可作为什么 | 不能作为什么 |
|---|---|---|---|
| G1 checkpoint selection | ../work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/checkpoint_selection.json | 四模态 t1c/t1n/t2w/t2f 均固定 step 150000、zscore、EDM Heun、18 steps、crop 64 的唯一加载合同。 | 不能自动授权 Route 训练或替代 route_config。 |
| G2 checkpoint gate | ../work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/g2_diffusion_qc_gate.json | 父 checkpoint 的 SHA、采样参数、full94+9 复核与 no-op 证据。 | 不是 component_manifest、patch 事务或 MET-AUG route 的批准。 |
| G2 自动/人工报告 | ../work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/automatic_qc/QC_REPORT.md 与 manual_review/HUMAN_REVIEW_2026-07-21.md | 生成器技术风险的可追溯证据。 | 不能证明生成样本会提高 S2 指标。 |

父 gate 的完成事实为：

- 四个 checkpoint 都是 step 150000，且 checkpoint selection 保存了各自 SHA256。
- 固定 validation 为 94 个 lesion-positive 加 9 个 lesion-negative/no-op。9/9 no-op 逐元素不变。
- 自动硬失败为 0；94/94 lesion-positive montage 已人工复核。
- 11 例为 pass_technical_visual，83 例为 pass_with_documented_risk。风险已接受为“生成器技术可用”，不是“无风险”或“训练收益已证实”。
- G2 gate 的 decision=approve 只绑定上述 checkpoint selection；任何未来 Route 还必须绑定新的 manifest、route config、事件审计和 Gate 1/2 结果。

G1 的较早服务器手册仍可见 100000 step 等历史默认参数；后续不得从该手册默认值、文件名最大值或旧脚本推断当前权重。实际调用只能读取本节的 checkpoint selection，并验证其 SHA 与 G2 parent gate 一致。

历史固定 103 例中的生成质量分层事实也必须保留：

- 94 例为 lesion-positive，可用于有真实条件的生成质量统计。
- 9 例 seg 全零，属于 lesion-negative/no-op stratum；生成路径必须返回 was_modified=False，image 与 seg 逐元素不变。
- 103 例始终都属于 S2 segmentation 评估集合；不得写成“103 例均生成病灶”。

### 4.6 G1、G2、S2 的当前实现成熟度

| 层 | 当前已存在的实现/证据 | 已验证能力 | Route A-F 尚缺的能力 |
|---|---|---|---|
| G1 | GliGAN 的 OnTheFlyTumourAugmenter、四模态 150k checkpoint、固定 selection。 | 在 64^3 crop 内按多类标签生成四模态病灶外观；四模态 checkpoint 与采样配置已冻结。 | 旧 augmenter 按整例 label_pool 随机抽样，保留胶质瘤式 SNFH->ET->NETC 级联和激进缩放，不能表达转移瘤组件与 Route A-F。 |
| G2 | `g2_freeze_diffusion_full_eval.py`、full94+9 parent gate、`checkpoint_selection.json`。 | 四模态 150k、z-score、64^3、EDM-Heun 18 steps 已冻结；94 positive + 9 strict no-op 通过自动和人工父级 QC。 | Route A 的真实 24 例 patch 证据与人工复核尚未运行；full-generation composer 仍不可替代 patch QC。 |
| S2 | `met_aug_data_loader.py`、`met_aug_transform.py`、Route A/control trainer、`met_aug_paired_training.py`，以及 `train.sh`/`infer.sh` 隔离。 | 本机 `g1_t2w_bbdm` 与 UHost 项目环境均通过 73/73；UHost 的 nnU-Net 2.8、Torch 2.7.1/cu128、双 H20 CUDA 实算、单 patch 接口与 inference shim 已验收。 | Dataset264 正从 ECNU 直传 UHost；完整数据门通过后仍须运行真实训练路径 smoke 和断点续跑检查。 |
| MET-AUG | `met_aug_core.py`、`met_aug_diffusion.py`、`met_aug_gate.py`、`met_aug_gate2.py`、scripts 12-20 与 UHost scripts 05-11。 | Route A 合同、patient-group 隔离、显式 support、四模态原子提交/回滚、Gate 1/2、人工 finalizer、route approval schema 3、immutable provenance 和训练 smoke 报告已实现；本地 42 个定向测试及全部 73 个 S2 测试通过。 | UHost 上的真实 manifest、Gate 1/2/人工批准、route approval、training smoke、E-continue/Route A 训练和固定 103 例正式评估尚不存在；B-F 尚未实现。 |

因此，当前状态应准确表述为：**G1/G2 父证据和 Route A 桥接代码已完成；真实样本增强 Gate 与增强训练尚未开始。**

---

## 5. 冻结产物登记

### 5.1 选择与风险报告

| 产物 | 路径 | 用途 |
|---|---|---|
| 正式比较 | ../work_space/S2/results/s2_small_lesion_ablation_20260721/final_comparison_20260724.md | 本轮结论与汇总指标。 |
| 机器可读选择 | ../work_space/S2/results/s2_small_lesion_ablation_20260721/checkpoint_selection.json | 唯一的候选、SHA256、评估合同和选择角色来源。 |
| 逐病例风险复核 | ../work_space/S2/results/s2_small_lesion_ablation_20260721/risk_review_20260724.md | 高 FN、RC 和 FP 风险解释。 |
| 运行清单 | ../work_space/S2/results/s2_small_lesion_ablation_20260721/run_manifest.json | 作业、路径和阶段状态。 |
| 官方兼容评估根 | ../work_space/S2/results/s2_small_lesion_ablation_20260724_official_eval_retry_numpy126 | A-1、E、A-1+E 的原始评估输出。 |

### 5.2 当前可复用 checkpoint

| 角色 | Trainer | 本地归档 checkpoint | SHA256 |
|---|---|---|---|
| B | nnUNetTrainerBraTS2026RCCompletionFineTune | ../work_space/S2/results/s2_completion_dataset264_t2w_20260720/fold_0/checkpoint_final.pth | 78eccc59f9217a529cafdd522733de9a1578f0e96d8765ee7c48731027824db5 |
| E | nnUNetTrainerBraTS2026RCFocalCompletionFineTune | ../work_space/S2/results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth | 4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267 |

A-1 与 A-1+E checkpoint 仍保存于 selection JSON 指定位置，但不能作为 S2-DS-D 或 MET-AUG 默认父 checkpoint。

ECNU 上的 E 父 checkpoint 源路径为：
`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/data/ecnu_completion_emergency/nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`；
UHost 已同步到 `/root/brats2026/ECNU_EYU_data/work_space/S2/results/s2_small_lesion_ablation_20260721/remote_snapshot_complete_20260724T0343/focal/fold_0/checkpoint_final.pth`。两处启动前都必须复核 SHA256 等于表中的 E 哈希。

### 5.3 相关实现与规范

| 文档或目录 | 角色 |
|---|---|
| ../work_space/S2/docs/S2_小病灶消融对比执行计划.md | S2 首轮消融、E 选择和 S2-DS-D 约束。 |
| ../work_space/S2/docs/ON_THE_FLY_AUGMENTATION.md | MET-AUG-A 至 MET-AUG-F 的唯一可执行规范。 |
| ../work_space/S2/BraTS2026_S2_RC_v1.0/repository | nnU-Net trainer、测试和发布脚本所在代码库。 |
| ../work_space/G2/results/manifests/nnunet_case_mapping_master.csv | master 身份与 patient-group 映射。 |
| ../work_space/G2/results/splits/splits_final_train_val_test.json | 固定 split 证据。 |
| ../data_space/task1_2026/reference_code/BraTS_evaluation | 官方评估快照说明。 |

---

## 6. 更新后的全流程

~~~mermaid
flowchart TD
    A["冻结 Dataset264 与 G1/G2 completion 证据（完成）"] --> B["B/A-1/E/A-1+E 同一 103 例正式评估（完成）"]
    B --> C["冻结：E 为小病灶基座；B 为整体保守对照（完成）"]
    C --> D["Route A 桥接代码与本地测试（完成）"]
    D --> E["UHost 隔离环境 + Dataset264 缓存门禁"]
    E --> E1["UHost 12/13/14/15/17 无 Diffusion Gate"]
    E1 --> F{"用户授权 24 例真实 Gate 2？"}
    F -->|否| G["保持 E，不产生增强结果"]
    F -->|是| H["18 真实四模态 Gate 2 + 24/24 人工复核 + 19/16 approval"]
    H --> I{"Route A approval 通过？"}
    I -->|否| G
    I -->|是| J["20：真实 nnU-Net + 四 Diffusion 训练 step smoke"]
    J --> K{"显存、finite loss、提交事件和吞吐通过？"}
    K -->|否| G
    K -->|是| L{"用户授权配对训练？"}
    L -->|否| G
    L -->|是| M["同一 E 分叉：E-continue p=0 与 Route A p=0.20，各 200 epoch"]
    M --> N["固定 103 例：B、原始 E、E-continue、Route A 正式评估"]
    N --> O{"Route A 相对 E-continue 有小病灶净收益且保护指标安全？"}
    O -->|否或不确定| P["回退原始 E 或 E-continue 中风险更低者"]
    O -->|是| Q["冻结 Route A checkpoint"]
    P --> R{"最佳模型仍有明确小病灶瓶颈？"}
    Q --> R
    R -->|是且仍有预算| S1["再决定 S2-DS-D 或下一条单变量 Route"]
    R -->|否| R1["冻结最终内部模型"]
    S1 --> R1
    G --> S{"用户授权官方 179 推理？"}
    R1 --> S
    S -->|否| T["保留内部结论，不产生官方预测"]
    S -->|是| U["179 例纯 segmentation 推理、空间/标签审计与 ZIP"]
    U --> V{"用户明确授权提交？"}
    V -->|否| W["归档已审计 ZIP，不上传"]
    V -->|是| X["Synapse 上传、记录 submission ID 与官方返回结果"]
~~~

当前默认顺序固定为先完成 `E-continue vs E+MET-AUG-A`，原始 E/B 只作部署与风险锚点。S2-DS-D 不再阻塞 Route A；只有 Route A 结论后最佳模型仍有明确小病灶瓶颈时才重新评估 D。三天窗口内不并行铺开 A-F，以免失去单变量归因和 GPU 预算。

---

## 7. Phase 0：冻结、维护与无计算准备

### 7.1 本阶段状态

S2 首轮已经完成。当前可做的工作限于文档、代码审查、测试、哈希、source snapshot、manifest 设计和评估汇总，不得借“准备”名义启动训练或在线 Diffusion。

### 7.2 必须持续保护的资产

- Dataset264 raw/preprocessed cache、1035/103/104 split、1138 b2nd cache IDs。
- B/E checkpoint、A-1/A-1+E 审计 checkpoint、103 例 prediction/reference 与 metrics。
- G1/G2 completion、人工复核、历史 Diffusion、失败作业日志和环境记录。
- /public 可用空间应保持高于 1 TiB。若存在存储压力，先停止新任务并输出占用审计；不得自行删除 cache 或 checkpoint。

### 7.3 任何未来运行前的共同 preflight

1. 记录实际项目文件 SHA256、Git HEAD、dirty status、Conda 环境、CUDA/PyTorch/nnU-Net 版本。
2. 验证 Dataset264 split、cache ID、plans、通道顺序与标签值域。
3. 验证 parent checkpoint 的文件大小和 SHA256。
4. 为新实验使用新的 result root；不得覆盖 B、E 或首轮比较目录。
5. 确认 evaluation 环境仍可导入指定版本，并用已归档的 103 例输出进行一个只读 smoke。

---

## 8. 条件后备阶段：S2-DS-D 的正确设计与预注册

### 8.1 目的与范围

S2-DS-D 只检验一个变量：在 E 相同架构和 Focal loss 下，改变 deep-supervision 各输出的 loss 聚合权重是否能改善小病灶表现，同时不牺牲整体和 RC 安全性。

**当前优先级**：本阶段保留为后备，不是 Route A 的前置任务。只有完成 `E-continue vs E+MET-AUG-A` 后，最佳模型仍有
明确小病灶瓶颈且用户另行授权时，才进入本节实现或训练；不得与 Route A 并行消耗当前三天预算。

本阶段不引入 A-1、在线 Diffusion、MET-AUG、额外数据或新的 split。它是干净的 E 对照，不是架构或数据增强的组合实验。

### 8.2 不可变训练合同

| 项目 | S2-DS-D 固定值 |
|---|---|
| 父模型 | E 最终 checkpoint，SHA256 为 4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267。 |
| 结构 | 原版 6-stage 网络；不使用 A-1 架构。 |
| decoder 输出 | 5 个 deep-supervision 输出。 |
| 基础 loss | 与 E 相同的 Dice + Focal CE。 |
| Focal | gamma = 2.0。 |
| RC 类权重 | 3.0。 |
| split | Dataset264 固定 1035/103/104。 |
| 训练 | 200 epoch，fold_0，初始 lr 0.001，每 25 epoch checkpoint。 |
| validation 与评估 | 同一固定 103 例；不调用 Diffusion。 |

### 8.3 正确实现位置

权重只能在 trainer 的 _build_loss 中通过 DeepSupervisionWrapper 施加：

~~~python
base_loss = dice_plus_focal_ce_same_as_E(...)
loss = DeepSupervisionWrapper(
    base_loss,
    [0.40, 0.30, 0.15, 0.10, 0.05],
)
~~~

下列行为一律禁止：

- 不得把上述权重写入 _get_deep_supervision_scales；该方法控制 label downsampling 的几何尺度，不是 loss 权重。
- 不得改变原版 6-stage 的输出数量、plans、通道、标签、Focal gamma、RC 权重或 split。
- 不得让权重向量长度与实际输出数不一致。
- 不得把 E checkpoint 部分加载成 A-1 形状；S2-DS-D 的架构未变，必须全量 warm-start 并生成加载审计。

### 8.4 预注册代码与 smoke Gate

在向用户请求训练授权前，必须完成且归档以下不消耗训练预算的证据：

1. 新 trainer 有唯一名称和独立 result root；命名中包含 S2-DS-D 或等价不可冲突标识。
2. 单元测试检查 5 个 output、5 个权重、权重和为 1、每个 target 的 shape 与输出匹配。
3. 单 batch forward/backward 检查 loss 为有限值、各 loss 分支都有梯度，且不改变 E 的标签 downsampling 语义。
4. 严格加载 E checkpoint，记录 E SHA256、loaded/missing/unexpected keys；任何非预期缺失都阻断训练。
5. 固定 train/validation manifest 与 E 完全一致，并输出 SHA256 对照。
6. 记录代码 snapshot、环境、随机种子、训练参数和预注册决策规则。

建议使用全新目录，例如：

~~~text
work_space/S2/results/s2_ds_d_e_base_<YYYYMMDD>/
  pre_registration.json
  source_snapshot/
  preflight/
  train/
  validation_predictions/
  official_eval/
  risk_review/
  checkpoint_selection_after_ds.json
~~~

该目录只是命名规范；未获训练授权前不得伪造 train、prediction 或 official_eval 完成标记。

### 8.5 训练与产物验收（仅获授权后）

训练启动后必须同时满足：

- 实际 Slurm 作业、nnUNetv2_train 进程、GPU PID 与 trainer 名称一致。
- GPU 可用显存至少 30 GiB；loss、epoch 和 checkpoint mtime 持续且有限。
- 无 OOM、NaN、Inf、Traceback、worker deadlock 或隐式 Diffusion 调用。
- checkpoint 保存于 25、50、75、100、125、150、175、final；中断只可从完整 checkpoint_latest 续跑。
- checkpoint_final、SHA256、103 个 validation prediction、nnU-Net summary 和完成标记齐全。
- 结果目录不覆盖 E、B 或任一旧作业。

### 8.6 S2-DS-D 选择门

评估必须继续使用第 3.4 节的完整 103 例合同，并形成 E vs S2-DS-D 的 paired 表，同时保留 B 作为整体风险锚点。

S2-DS-D 只有同时满足以下条件才能替换 E 为后续基座：

1. 103/103 prediction/reference 完整，代码、环境、parent checkpoint 和输出哈希可追溯。
2. 预注册的小病灶主指标出现有方向、经 paired bootstrap 支持的改善，而非单例驱动。
3. WT all-instance F1、lesionwise DSC/NSD 不出现预注册的实质退化。
4. RC 的 FN/FP、RC DSC/NSD 与 large-instance 风险不出现无法解释的安全回归。
5. 人工复核改善最大、退化最大、RC、small/large 重点病例后，无标签、空间或边界异常。

RC small-instance 样本量仅 3 例，必须完整报告但不能被夸大为独立的确定性胜负证据。若结果互有得失、置信区间不支持净收益或任一安全门失败，继续选择 E。

选择文件必须新建，不能覆盖现有 checkpoint_selection.json。至少写入：

~~~text
base_candidate
candidate
parent_checkpoint_sha256
trainer
split_sha256
evaluation_contract
paired_metrics
risk_review
decision
decision_reason
authorizer
~~~

---

## 9. 当前主阶段：MET-AUG-A 至 MET-AUG-F 的条件路线

### 9.1 当前状态与进入条件

MET-AUG 路线是后续训练 patch 内的合成病灶增强，不是 Dataset264 重建，不修改 validation、locked test、官方 179 数据，也不是现有 S2 结果的补丁。

当前状态：

- 方案版本为 MET-AUG-A-F-v2。
- Route A 的 train-only 组件合同、事务适配器、Gate 1/2/approval、E-continue control、共享配对合同、训练路径 smoke 和推理隔离已经实现。
- 本地 `g1_t2w_bbdm` 环境已核验 nnU-Net `2.8.0`，42 个定向测试及全部 73 个 S2 测试已通过；这不替代 UHost 真实 Dataset264/GPU smoke。
- 双 H20 UHost 已通过 SSH key 连通，代码、四模态 G1 150k checkpoint、G2 parent gate/mapping 和冻结 E checkpoint 已逐文件 SHA256 验证；当前两张 GPU 空闲，没有 Gate、Diffusion smoke 或训练进程。
- UHost 是无 Slurm 容器；现有 `.slurm` 作为可迁移 Bash 入口直接执行，不为单节点额外安装 Slurm。隔离 Conda 环境和 `rsync 3.2.7` 已验收；Dataset264 正从 ECNU 已验收缓存以两路可续传 rsync 直推 UHost。
- 真实 component_manifest、valid-mask manifest、Gate 1/2、人工复核、route approval、训练和正式评估尚未建立。
- 旧 G1 label_modifier.py、on_the_fly_augmentation.py 和 lesion_pool.csv 只可作历史参考，不能直接接入。
- 没有用户明确授权时，不得启动任何在线 Diffusion、离线 smoke 生成或 Route 训练。

Route A 的 `base_candidate` 无条件固定为 E，父 checkpoint SHA256 固定为
`4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`。S2-DS-D 不得在 Route A 启动前
替换该基座；未来 B-F 的基座变更必须来自前一 Route 的正式 Gate 3 选择记录。

### 9.1.1 父 gate 的绑定方式

每个 Route 的 runtime 只能把 G1/G2 full94+9 结果当作父证据，启动时必须同时验证：

1. G1 checkpoint selection 的 SHA256 与 G2 parent gate 中的 checkpoint_selection_sha256 一致。
2. 四个 150k checkpoint、zscore、edm_heun、18 steps、crop 64 与 parent selection 一致。
3. Route 自己的 component_manifest、route_config、代码 snapshot、Gate 1、Gate 2 与人工复核摘要各有 SHA256。
4. Route gate 将 parent gate SHA、component_manifest SHA、route_config SHA、Gate 1/2 SHA 与代码 SHA 绑定为一个新的 decision=approve 文件。

已有的 G2 parent gate 不能被改写、复制成 Route gate 或以“已有 approve”为理由跳过 Route Gate 1/2。

### 9.1.2 BraTS 2025 原则与旧 S2 bridge 的保留边界

BraTS 2025 可继承的只有：训练 step 动态插入、validation 不变、`(1-p)` 原样返回、donor 来自其他患者，以及
与 baseline 保持公平对照。论文的 Regular `p=0.75`、Custom `p=0.60`、0.7 类别替换、0.1-0.8 缩放、第二灶 0.4
和 GliGAN 权重均不直接迁移。论文 Custom 单模型的内部 lesion-wise 平均排名为 5.67，作者也明确说明概率/尺度
随机选取且训练未完全收敛，因此这些数值不是 BraTS-MET 的最优参数证据。

旧 S2 bridge 可保留为底层参考的只有：

- G1 checkpoint selection、模型加载和四模态 sampler 的版本/哈希合同。
- S2/G1 的 channel swap 与 C,Z,Y,X <-> C,X,Y,Z 轴转换逻辑。
- 单 GPU、主 dataloader 进程调用、训练期专用而 validation/test 不调用生成器的边界。

不得把现有 `OnlineDiffusionTransform` 或 `nnUNetTrainerBraTS2026RCOnlineDiffusion` 作为 MET-AUG 训练实现。该旧路径
已由 `train.sh` 的 `completion_online` 分支显式退役。

Route A 已使用新的 S2 专用适配器，真实逻辑接口固定为：

~~~text
MetAugEngine.apply(
    image=data_4ch,
    segmentation=seg,
    valid_mask=valid_mask_patch,
    context=(epoch, rank, worker, case_id, patch_index, patch_origin, full_shape),
) -> data_4ch_out, seg_out, transaction_result
~~~

`case_id`、patch origin/full shape 和 valid mask 由 dataloader sidecar 显式提供；patient group 与 train target 资格
从冻结 component manifest 解析。调用方不能传入一个可漂移的 split/group 值，也不得从 patch 的 image/seg 猜测身份。

### 9.1.3 已实现的 G2/S2 接口层

| 已实现接口 | 所属层 | 输出/职责 |
|---|---|---|
| `scripts/12_build_met_aug_component_pool.py` | G2->S2 | 从固定 1035 train 的实际 `nnUNetPlans_3d_fullres` 分割空间生成不可变组件池，记录 split、patient_group、类别、体积、bbox、raw 源哈希、plans SHA 和 manifest SHA。 |
| `scripts/13_make_met_aug_route_a_config.py` | S2 | 冻结 Route A 的 `p=0.20`、单灶、scale 1、组成-体积分层和 seed。 |
| `scripts/14_prepare_met_aug_valid_masks.py` | S2 | 将四模态 union brain/valid mask 对齐到实际 preprocessed 空间；默认拒绝未审计的重采样。 |
| `scripts/15_run_met_aug_gate1.py` | G2->S2 | 100,000 次无 Diffusion 模拟、Q_route 分布、no-op/失败和泄漏审计。 |
| `met_aug_core.py` + `met_aug_data_loader.py` + `met_aug_transform.py` | S2 | 在临时 data/seg 副本上执行供体、位置、四模态生成和原子提交，并显式携带 patch metadata。 |
| `scripts/17/18/19` + `met_aug_gate2.py` | G2->S2 | 预注册固定 24 例、真实四模态 patch QC、montage、人工 24/24 finalizer 和证据 SHA 防漂移。 |
| `scripts/16_finalize_met_aug_route_a_gate.py` | G2->S2 | 汇总 parent gate、manifest/config、Gate 1/2、人工复核和 runtime code，写 route-specific approval。 |
| `scripts/20_run_met_aug_training_smoke.py` | G1->S2 | 在独立 result root 同时加载 nnU-Net 与四模态 Diffusion，实际执行 batch/train step，记录峰值显存、提交率、finite loss 与墙钟估计；不保存 checkpoint、不运行 validation。 |
| Route A/control trainer + `met_aug_paired_training.py` + `train.sh`/`infer.sh` | S2 | 同一 E warm-start 分叉为 p=0 与 p=0.20；固定 seed 20260724、seed+epoch、单线程、0 train warmup、确定性 cuDNN、禁用 compile；provenance 漂移拒绝续跑；validation/inference 纯 segmentation 隔离。 |

新 S2 transformer 必须在所有四个模态的临时输出、最终 seg、support 和 event_audit 均通过后才一次性提交。任一模态失败、标签失败、placement 失败或审计失败时，返回输入 bit-identical 副本，并记录 no-op/failure reason。旧实现按模态循环写回的行为不能作为此处的事务语义。

底层唯一命令顺序和完整参数模板见 `../work_space/S2/docs/ON_THE_FLY_AUGMENTATION.md` 第 10.4 节：
`12 -> 13 -> 14 -> 15 -> 17 -> 18 -> 人工复核 -> 19 -> 16 -> 20 -> E-continue/Route A 配对训练`。
其中 18 首次加载真实 Diffusion，20 首次验证完整训练路径，均必须使用独立不可变输出。UHost 包装层见同文档第 10.6 节与 `work_space/S2/slurm/05-11`：环境、缓存、Gate 和训练依赖没有通过时，启动器不得产生两臂进程。

### 9.1.4 双 H20 UHost 当前执行链

| 步骤 | 入口 | 当前状态与放行条件 |
|---|---|---|
| 运行时资产同步 | `09_sync_s2_runtime_to_uhost.sh` | 已完成；远端 manifest 的全部关键文件 SHA256 通过。 |
| 项目环境 | `setup_s2_h20_environment.sh` | 已完成；`pip check`、远端 73/73、Torch/CUDA 版本、双 H20 实算与环境 audit 通过。 |
| Dataset264 搬运 | `11_push_dataset264_direct_to_uhost_ecnu.sh`；`10_relay_dataset264_via_vpn_to_uhost.sh` 仅回退 | ECNU->H20 直连已验收，raw/preprocessed 两路唯一 rsync 正在运行；脚本 11 提供原地续传、状态与完整数据门，脚本 10 只在直连不可用时使用。 |
| Gate 1/2/approval/smoke | `05_met_aug_gate_h20_uhost.slurm` | 按 `prepare -> gate2 -> 人工 24/24 -> finalize -> training_smoke` 分阶段执行；不覆盖任何已有证据。 |
| 双臂配对训练 | `07_launch_met_aug_pair_h20_uhost.sh` -> `06_train_met_aug_pair_h20_uhost.slurm` | 只在 route approval 和 training smoke marker 齐全时启动；GPU 0 运行 E-continue，GPU 1 运行 E+MET-AUG-A，两臂各 200 epoch。 |
| 只读监控 | `08_status_met_aug_pair_h20_uhost.sh` | 查看 PID、GPU、磁盘与最新 epoch/loss/异常；不依赖 Slurm stdout 判定训练状态。 |

正式路线仍锁定 `EDM-Heun/18 steps/FP32`。只有真实 training smoke 估算两臂 200 epoch 超过 45 小时时，才写入 `TRAINING_SMOKE_TOO_SLOW.hold` 并转入加速候选评估。DPM-Solver++、UniPC、减少步数或 BF16 都不得直接替换正式路线；必须在同一 24 例上与 Heun-18 做自动指标、墙钟和人工质控配对，生成新的 route approval 后才能启用。

### 9.2 先建不可变 train-only 组件池

component_manifest 只能从 Dataset264 的 1035 例训练集构建，且一次性冻结：

| 项目 | 约束 |
|---|---|
| 空间 | Dataset264 的 1 x 1 x 1 mm 预处理空间。 |
| 组件锚点 | NETC(1) union ET(3) 的 26-connectivity。 |
| SNFH 归属 | 只归给唯一最近核心；并列或无核心关联时不作为供体。 |
| split | donor 只能来自 train；validation、locked test、官方 179 必须为 0。 |
| patient group | donor 与 target 不得来自同一 patient_group。 |
| 排除 | 任意 RC、纯 SNFH、core_volume < 27 mm3、bbox 任一维 > 56 mm、标签/空间/哈希异常。 |
| 审计 | 组件 ID、来源、类别、体积、bbox、seed、输入 SHA256、manifest SHA256 均可追溯。 |

Route E 中“纯 SNFH 转 ET”与“RC 单独迁移”均为不可达分支；实现中不得保留隐式后门。

### 9.3 单病灶事务合同

每次插入必须是原子事务：

~~~text
SELECTED
  -> DONOR_VALID
  -> LABEL_VALID
  -> PLACEMENT_VALID
  -> FOUR_MODAL_SUCCESS
  -> COMMITTED
~~~

任一步失败均返回输入的 bit-identical 副本并写入 reason code；只有四模态输出、标签、支持区域和审计都通过后才可一次性写入 data 与 seg。

共同约束：

- S2 通道顺序固定 t1n、t1c、t2w、t2f；不得直接复用旧 G1 的 t1c、t1n、t2w、t2f 顺序。
- 需要四个 checkpoint 和四个有限、同坐标、同掩膜、同形状的输出；禁止 partial modality。
- 最终 support 距原生病灶和已提交合成灶至少 5 mm，距脑边界至少 3 mm。
- 最终 bbox 任一维不超过 56 mm；inpainting crop 为 64^3。
- 位置搜索最多 50 次，且必须基于最终标签重算 clearance。
- 随机事件由全局 seed、epoch、rank、worker、case、patch、route 派生，成功和失败都写 JSONL。

### 9.3.1 Gate 1 前的测试合同与当前证据

旧 `online_diffusion_contract` 只证明通道交换和轴变换可以往返，不能证明增强安全。当前定向测试已覆盖
patient-group 解析、确定性 donor/位置、单 patch sidecar、成功 support、生成失败/审计失败回滚、provenance 漂移拒绝、
Gate 2 固定分层/唯一性/人工字段和证据 SHA、nnU-Net 2.8 四返回值以及先 transpose 后 crop 的预处理轴序。完整 Gate 仍必须满足：

1. 同 patient_group、validation、locked test、RC、纯 SNFH、core < 27 mm3、bbox 超限的 donor 均被拒绝。
2. target_metadata 缺失、component_manifest SHA 不匹配、route_config 不匹配时硬失败，不能回退为未审计的随机抽样。
3. 固定 event identity 重放时，donor、变换、位置、support 与 reason code 一致。
4. 每个失败注入点（第 1 至第 4 模态、标签、placement、event 写入）都验证 data 与 seg bit-identical 回滚。
5. 成功时四模态只在 final support 内改变，seg 只新增合法标签，support 外与 no-op 样本逐元素不变。
6. 组件大小、bbox、clearance、类别拓扑和 0/1/2 灶计数在最终标签上重新计算。
7. `scripts/20_run_met_aug_training_smoke.py` 必须在单 GPU、主 dataloader 进程实际执行至少 4 个 train step，且在最多 12 步内观察到至少 1 次 `COMMITTED`；短 smoke 只给出墙钟估计，不能伪装成完整训练结果。

其中真实四 checkpoint 的逐模态异常、nnU-Net 2.8 dataloader 生命周期、显存与吞吐只能在 ECNU 项目环境验证；
train-only/RC/纯 SNFH/体积/bbox 的实际计数必须由 scripts 12/15 的真实 manifest 和 Gate 1 报告验收，不能用单元测试替代。

### 9.4 三级 Gate

#### Gate 1：100,000 次无 Diffusion 策略模拟

只模拟选择、分层抽样、标签变换和位置决策，不调用 Diffusion，不修改训练数组。必须证明：

1. val/locked test donor、同 patient_group donor、RC donor、纯 SNFH donor、超小核心和超 bbox donor 均为 0。
2. selected_rate 与设定 p_select 的绝对偏差不超过 0.005。
3. 输出 selected、donor_valid、label_valid、placement_valid 的逐级计数、失败 reason、重试与 fallback。
4. 最终标签再次通过类别、core_volume、bbox、clearance 检查。
5. 有效候选相对冻结 Q_route 的分层频率最大绝对偏差不超过 1 个百分点。
6. MET-AUG-B、MET-AUG-D、MET-AUG-E 的 fallback、第二灶失败和构造分支覆盖率单列；高 fallback 率不得直接进入训练。

#### Gate 2：固定分层四模态 smoke

至少生成 24 个预注册的离线样本：27-49、50-275、>275 mm3 三个核心体积分层各至少 8 个。每例归档原始/增强后四模态三视图、seg overlay、donor/target/位置/route/seed/checkpoint/哈希和自动 QC。

必须确认：

- 四模态 finite、shape/dtype/坐标和写入掩膜一致。
- 写入掩膜外逐体素不变，seg 只在最终 support 内新增。
- 标签为 0、1、2、3、4，体积、bbox、边界、病灶间隔复检通过。
- 人工复核无接缝、脑外插入、标签-影像错位、通道错误、异常强度或粘连。
- 通过 `scripts/18_run_met_aug_gate2_smoke.py` 与 `met_aug_gate2.py` 生成 route-specific QC、证据和人工模板；不得调用 `g2_v2_compose_augmentation.py` 的 full_generation 路径替代。

任何确定性错误、partial modality 或跨模态矛盾都会阻断该 Route。修复后只能用同一固定 smoke 集复测。

#### Gate 3：冻结 E 父 checkpoint 上的配对公平消融

同一个 E checkpoint 必须分叉为 `E-continue (p=0)` 与 `E+MET-AUG-A (p=0.20)`。两侧固定 trainer、
Dataset264 `1035/103` split、fold 0、200 epoch、LR 0.001、每 25 epoch checkpoint、batch、patch、标准
nnU-Net augmentation、seed 20260724、每 epoch `seed+epoch`、augmentation worker 0、丢弃 train warmup 0、
确定性 cuDNN、`nnUNet_compile=0`、checkpoint 选择规则和 103 例评估流程，并使用互不覆盖的结果根。

增强因果效应只取 `E-continue vs E+MET-AUG-A`；B 与原始 E 同时报告，但分别只是整体风险和部署收益锚点。
启动前必须预注册小病灶主指标、RC/large/FP 保护指标、paired bootstrap 和“实质退化”阈值。

### 9.5 Route 顺序与唯一变量

| Route | 唯一新增变量 | 初始对照与分支规则 |
|---|---|---|
| MET-AUG-A | 真实组件平移 + 四模态生成 | p_select=0.20、最多 1 灶、scale=1.0；必须先证明基础事务有效。 |
| MET-AUG-B | 受控核心体积缩放 | 只在 A 通过后执行；用目标体积比例换算线性 scale，并维持 27 mm3 地板。 |
| MET-AUG-C | 合成频率剂量 | 在 B 的固定策略上比较 p_select=0.20/0.40/0.60。 |
| MET-AUG-D | 第二个独立病灶 | 在最佳 C 上增加 p_second=0.20、max_tumours=2；第二灶是独立事务。 |
| MET-AUG-E | 受保护类别拓扑构造 | 从 BestLowerRisk 独立出发；E-SNFH 与 E-NETC 必须先各自通过，才可考虑 E-COMB。 |
| MET-AUG-F | 保持总剂量的自适应频率分配 | 从 BestLowerRisk 独立出发；总期望 p_select 与最佳 C 相同，不能借更多剂量制造收益。 |

串行关系固定为：

~~~text
MET-AUG-A -> MET-AUG-B -> MET-AUG-C -> MET-AUG-D -> BestLowerRisk
                                                          |          |
                                                   MET-AUG-E    MET-AUG-F
~~~

E 与 F 不能默认组合。仅在二者均独立通过 Gate 3 后，才能另立 MET-AUG-EF 交互实验；它不属于本路线默认交付。

### 9.6 Route 失败与回退

| 情形 | 处理 |
|---|---|
| MET-AUG-A 未通过 | 若 Gate/训练失败则保留冻结 E；若配对收益不成立则在原始 E 与 E-continue 中保留预注册风险更低者，不推进后续 Route。 |
| MET-AUG-B/C/D 未通过 | 回退到上一通过 Gate 3 的低风险配置并停止串行路线。 |
| MET-AUG-E 或 MET-AUG-F 未通过 | 不影响 BestLowerRisk；保留 reason 和证据，不强行组合。 |
| 泄漏、partial 写入、标签拓扑错、四模态不一致 | Route 无训练资格；修复后重跑同一 Gate。 |
| Route 基座或父 checkpoint 未正式选择 | 不得以未选择 checkpoint 作为 Route parent；Route A 只能使用冻结 E。 |

完整参数、类别拓扑和实现约束以 ../work_space/S2/docs/ON_THE_FLY_AUGMENTATION.md 的“转移瘤 Route A-F 分期消融路线”为准；本文不应复制后与该唯一规范分叉。

---

## 10. Phase 3：最终模型选择与内部归档

最终模型选择不是“某一指标最高即替换”。每次发生 S2-DS-D 或 MET-AUG Gate 3 后，都要在固定 103 例上重新执行：

1. 完整性检查：prediction/reference 103/103，source ID、reference 哈希、评估器版本一致。
2. 指标检查：WT/RC lesionwise DSC/NSD、all/small/large F1、FN/FP。
3. paired 风险：逐病例 delta、bootstrap、极端病例、RC、small/large 人工复核。
4. 回退检查：对 B 的整体 DSC/NSD/FP 风险、对原始 E 的部署收益，以及 E-continue/Route A 的增强因果差值。
5. 归档：固定一个唯一 checkpoint、SHA256、source snapshot、环境与选择原因。

模型选择规则：

- 当前未经后续实验的默认小病灶基座是 E。
- S2-DS-D 若未通过第 8.6 节，E 保持不变。
- 每条 MET-AUG 只有在 Gate 3 证明小病灶净收益、且整体/RC/FP 不越过预注册保护门时才可晋级。
- 结果互有得失或证据不足时，选择结构更简单、风险更低的父模型，不把实验性收益带入发布候选。
- B 必须始终保留，直至最终发布后的保留期结束。

新选择必须写入全新归档目录，建议：

~~~text
work_space/S2/results/final_model_selection_<YYYYMMDD>/
  FINAL_MODEL_DECISION.md
  checkpoint_selection.json
  paired_metrics.csv
  per_case_delta.csv
  risk_review.md
  model_manifest.json
  source_snapshot/
  environment_freeze.txt
~~~

不得覆盖 2026-07-24 的 E 选择文件；后续选择必须把该文件列为 parent evidence。

---

## 11. Phase 4：官方 179 例推理、审计与 ZIP（条件发布阶段）

### 11.1 进入条件

本阶段当前未授权。只有以下条件全部成立并获得用户明确“启动官方 179 例推理”授权后才能执行：

1. 最终发布 checkpoint 已经由第 10 节冻结，路径和 SHA256 唯一。
2. 内部 103 例选择报告、风险复核和源码/环境快照齐全。
3. 官方 179 例数据仍为四模态、无 seg，且完全独立于训练/选择。
4. 推理 profile 已在同一环境完成只读 preflight，不加载在线 Diffusion、不开启 test-time 训练。
5. 输出目录为空且独立，足以容纳预测和审计产物。

### 11.2 官方输入合同

官方输入必须在实际执行时重新核验：

~~~text
case directories = 179
t1n = 179
t1c = 179
t2w = 179
t2f = 179
seg = 0
flattened inputs = 716 NIfTI
~~~

每例输入展平为：

~~~text
BraTS-MET-xxxxx-xxx_0000.nii.gz  # t1n
BraTS-MET-xxxxx-xxx_0001.nii.gz  # t1c
BraTS-MET-xxxxx-xxx_0002.nii.gz  # t2w
BraTS-MET-xxxxx-xxx_0003.nii.gz  # t2f
~~~

网络主机、挂载目录与传输路径会变化，不能把旧 A800/A100 路径或预计资源当作现时事实。开始前在实际主机重新记录路径、空间、输入 ID 清单和 SHA256；传输只使用已授权凭据，不在命令中展开密码。

### 11.3 推理不变量

- 只加载第 10 节冻结的一个 checkpoint；fold_0 只是当前固定模型 API/storage key，不代表五折 ensemble。
- 不调用 G1 或 MET-AUG transform，不做在线生成，不改写官方影像。
- 不在官方数据上训练、调阈值或选择 checkpoint。
- 若最终模型来自 S2-DS-D 或 MET-AUG，推理入口仍只能是纯 segmentation 推理；训练期增强不会进入推理图。

### 11.4 输出技术审计

每个输出目录必须满足：

1. 恰好 179 个 nii.gz，ID 与输入一一对应，无 missing/unexpected。
2. 文件名为 BraTS-MET-xxxxx-xxx.nii.gz。
3. 标签为整数且只在 0、1、2、3、4。
4. dimensions、spacing、origin、orientation、affine 与对应输入一致。
5. 记录空预测病例，但不通过手工填标签“修复”。
6. 生成 output manifest、空间/标签审计、checkpoint SHA256、输入清单 SHA256、环境和命令记录。

可复用的打包入口位于：

~~~text
work_space/S2/BraTS2026_S2_RC_v1.0/repository/scripts/07_package_official_submission.py
~~~

若实际源码、参数或输出合同发生变化，先更新并测试审计脚本，再开始正式 179 例预测。

### 11.5 ZIP 合同

ZIP 只包含 179 个根目录 NIfTI，不包含 103 例 prediction、CSV、JSON、checkpoint、manifest 或子目录。归档至少包含 ZIP、ZIP SHA256、输出审计和 final model manifest。

官方 179 例没有公开真值，因此本地不能计算官方成绩，也不能用内部 103 例 reference 冒充官方 reference。生成 ZIP 只能称为“通过本地技术审计的提交包”，不是“官方评估完成”。

---

## 12. Phase 5：Synapse 提交与赛后归档（条件发布阶段）

提交是独立授权动作。即使 ZIP 已完成，也不得自动上传。

### 12.1 提交前 Gate

1. 用户明确指定刷榜或最终提交。
2. 现场确认正确的 BraTS 2026 Task 1 queue、当前提交格式、额度、截止时间和登录身份。
3. 若 queue 改为 Docker/container，立即停止；当前文件预测 ZIP 路线不能静默改造成容器路线。
4. 再次核验 ZIP 根目录结构、179 个文件、ZIP SHA256、模型 checkpoint SHA256 与 final model manifest。

### 12.2 上传与归档

只上传：

~~~text
<final_model>_Task1_validation_179.zip
~~~

每次提交后建立：

~~~text
work_space/S2/results/submissions/<submission_id>/
  submitted.zip
  zip_sha256.txt
  official_submission_validation.json
  official_submission_manifest.csv
  final_model_manifest.json
  source_snapshot/
  environment_freeze.txt
  synapse_submission_record.md
  leaderboard_result.json_or_screenshot
~~~

synapse_submission_record.md 至少写入 submission ID、时间、queue/task、provisional/final、模型、checkpoint SHA256、ZIP SHA256、官方返回指标、操作人和备注。

---

## 13. 停止、恢复与存储规则

| 风险 | 立即处理 | 恢复条件 |
|---|---|---|
| S2-DS-D checkpoint/加载不一致 | 停止训练 | E SHA256、严格加载审计、split 与代码 snapshot 全部一致。 |
| S2-DS-D OOM/NaN/Inf/死锁 | 停止该 run，不覆盖 E | 在独立 smoke 中定位并重新预注册参数；不得在正式 run 中临时改关键变量。 |
| S2-DS-D 指标不确定或安全门失败 | 选择 E | 完整保留 S2-DS-D 的证据，不重写历史选择。 |
| Route Gate 发现泄漏/partial 写入/通道错 | 停止 Route | 修复事务后重跑同一固定 Gate 集。 |
| evaluator 导入或版本异常 | 停止评估 | 在独立 brats_eval 环境修复并以已归档数据复验；不污染 segmamba。 |
| /public 可用空间低于 1 TiB | 停止启动新工作 | 先审计占用、保护 checkpoint/cache/评估产物；删除需要单独用户决策。 |
| UHost 根盘可用空间低于 150 GiB | Gate 或训练启动器硬失败 | 先停止新运行并审计数据、checkpoint 与日志；不自动删除产物。 |
| 179 输入或输出计数/几何不符 | 不打包、不提交 | 新建空输出目录，修复后完整重跑；不手工补文件。 |
| Synapse 格式或队列变化 | 不上传 | 获得当前规则和用户确认后再决定下一步。 |

所有恢复都必须新建 run record，记录 parent run、失败原因、修复内容、输入/输出哈希和授权状态。不得取消、重提或覆盖已完成历史作业以“清理状态”。

---

## 14. 运行记录、命名和保留规范

### 14.1 新 run 的最小结构

~~~text
results/<stage>/<YYYYMMDD_HHMM>_<run_id>/
  RUN_MANIFEST.json
  SOURCE_SNAPSHOT.txt
  ENVIRONMENT.txt
  COMMAND.txt
  SHA256SUMS.txt
  metrics/
  qc/
  reports/
~~~

RUN_MANIFEST.json 至少记录：

~~~json
{
  "run_id": "...",
  "role": "s2_ds_d_preflight|s2_ds_d_train|met_aug_gate|met_aug_training_smoke|met_aug_control_train|met_aug_route_train|s2_eval|official_infer",
  "parent_run_ids": ["..."],
  "base_candidate": "E",
  "input_manifest_sha256": "...",
  "split_sha256": "...",
  "parent_checkpoint_sha256": "...",
  "checkpoint_sha256": "...",
  "seed": 20260724,
  "status": "pass|fail|aborted"
}
~~~

### 14.2 不进入 Git 的大产物

不得提交 NIfTI、checkpoint、nnU-Net preprocessed cache、大型 ZIP、临时调试产物、代理配置或任何凭据。Git 中只保留实现、轻量 manifest、报告、可复现配置和不含敏感信息的摘要。

### 14.3 当前保留期

在 S2-DS-D、MET-AUG 与最终发布策略明确结束前，保留所有 Dataset264 cache、B/E 和历史消融 checkpoint、103 例评估、G1/G2 QC、失败作业记录与评估环境说明。任何清理动作都需要独立存储审计和用户确认。

---

## 15. 阶段总表与下一步清单

| 阶段 | 当前状态 | 可以执行的下一动作 | 需要用户授权才可进入 |
|---|---|---|---|
| Dataset264/G1/G2 completion | 已完成并冻结 | 只读审计与保留 | 任何重建或清理。 |
| B/A-1/E/A-1+E | 已完成并冻结 | 使用 E/B 作为后续比较锚点 | 重训任一既有候选。 |
| S2-DS-D | 条件后备、未开始 | 暂不占用当前三天主线 | Route A 结论后再决定设计或训练。 |
| G1 150k + G2 parent gate | 已完成，仅作父证据 | 复核 selection/gate SHA 与人工风险记录 | 以 parent approve 代替 Route gate。 |
| MET-AUG-A 接口层 | 代码、本地/远端 73/73、control 和 training-smoke 入口完成；UHost 资产 SHA、独立 Conda、Torch/CUDA 与 `rsync` 已验收 | 保持环境冻结，不再安装或升级依赖 | 任何环境变更。 |
| UHost Dataset264 数据门 | ECNU 分流 VPN 与 ECNU->H20 直连已验收；raw/preprocessed 两路可续传 rsync 正在运行 | 用 `11_push_dataset264_direct_to_uhost_ecnu.sh status` 只读监控，完成后执行 `validate` | 传输失败后改变来源、目标或参数。 |
| MET-AUG-A 数据 Gate | 真实产物未开始 | 依次执行 12/13/14/15/17 并验收 immutable manifest | 18 的真实 24 例 Diffusion smoke。 |
| MET-AUG-A 训练 smoke | 入口完成、UHost 未运行 | approval 后执行 20，验收联合峰值显存、finite loss、至少 1 次提交和墙钟估计 | 真实训练 step。 |
| MET-AUG-A Gate 3 | 未开始 | 完成 24/24 人工复核、19/16/20 和训练预注册 | E-continue 与 Route A 各 200 epoch 及固定 103 例评估。 |
| MET-AUG-B 至 F | 仅分期规范 | 保持关闭 | A 通过 Gate 3 后按单变量顺序另行授权。 |
| 最终 checkpoint 选择 | 未开始 | 维护 B/E 回退和选择模板 | 将模型冻结为 release candidate。 |
| 官方 179 推理 | 未开始 | 保持数据隔离、维护审计脚本 | 读取/推理官方 179。 |
| ZIP/Synapse | 未开始 | 维护打包合同 | 打包、上传或提交。 |

### 当前最小下一步

1. UHost 环境和运行时资产已冻结；保持当前 ECNU 后台直传唯一实例，只用 `11_push_dataset264_direct_to_uhost_ecnu.sh status` 监控，不再执行本机反向隧道脚本或创建第二份 rsync。
2. 直传完成后执行 `11_push_dataset264_direct_to_uhost_ecnu.sh validate`，复核 Dataset264 raw/preprocessed、4552/1138/416/104 文件计数、1035/103/104 split、1242 included cases、212 completion overrides、1138 组 b2nd 及 UHost 至少 150GiB 可用空间；任一合同不符都不进入 Gate。
3. 在新的 Route A 根目录用 `05_met_aug_gate_h20_uhost.slurm prepare` 执行 `12 -> 13 -> 14 -> 15 -> 17`；这些步骤不加载 Diffusion，逐项验收 1035 train-only、patient-group、valid-mask、100,000 次模拟与 Gate 1 报告。
4. 获得真实 Gate 2 授权后执行 `gate2`，检查 24 个固定且不重复的 target/donor、三档各 8、四模态/NPZ/montage/event SHA；完成人工 24/24 后执行 `finalize`。
5. approval 后执行 `training_smoke`；只有联合 GPU 峰值、finite loss、至少 1 次 `COMMITTED`、无 checkpoint/validation 写入且估时不超过 45 小时，才放行配对训练。超时时保留 hold，先做同一 24 例的加速配对验证，不直接换 solver。
6. 执行 `07_launch_met_aug_pair_h20_uhost.sh`，从同一 E checkpoint 启动两个独立结果根：GPU 0 的 E-continue `p=0` 与 GPU 1 的 Route A `p=0.20`，其余训练合同逐项一致，均使用 `S2_CONTINUE=auto`。
7. 在同一 103 例上正式比较 B、原始 E、E-continue、E+MET-AUG-A 的 overall、RC、官方 `small/large`、FN/FP、DSC/NSD 和 small-instance F1；增强归因只比较后两者。`mets` 不输出独立 tiny；追加 `<27 mm3` 风险审计时必须标成非官方配对 reference-component 分析。
8. 只有上述最佳模型仍有明确小病灶瓶颈时才设计 S2-DS-D 或推进下一条 Route；官方 179 推理和提交继续关闭，等待独立授权。

---

## 16. 参考与最终判定

本总控依赖以下当前文档：

1. ../work_space/S2/results/s2_small_lesion_ablation_20260721/final_comparison_20260724.md
2. ../work_space/S2/results/s2_small_lesion_ablation_20260721/checkpoint_selection.json
3. ../work_space/S2/results/s2_small_lesion_ablation_20260721/risk_review_20260724.md
4. ../work_space/S2/docs/S2_小病灶消融对比执行计划.md
5. ../work_space/S2/docs/ON_THE_FLY_AUGMENTATION.md
6. ../work_space/G2/results/reports/G2_synthetic_data_quality_report_run_3104668.md
7. ../work_space/G2/results/qc/v3_completion_review/run_3104668/HUMAN_REVIEW_2026-07-18.md
8. ../data_space/task1_2026/reference_code/BraTS_evaluation/README.md
9. ../work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/checkpoint_selection.json
10. ../work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/g2_diffusion_qc_gate.json
11. ../work_space/G2/results/qc/diffusion_checkpoint_full94_150000_a800_recovery_20260721/manual_review/HUMAN_REVIEW_2026-07-21.md
12. ../work_space/S2/docs/BraTS 2025.pdf（SHA256 `18fd9192f118422cd05fcc164afb9d53a2ce3558715fc11aec35fb623d0c1c8b`）

只有在下列条件同时成立时，才可宣称“BraTS 2026 Task 1 最终流水线完成”：

1. 已冻结的最终模型通过同一固定 103 例的官方兼容评估、paired 风险复核和完整性审计。
2. 所有获授权的 S2-DS-D 或 MET-AUG 实验均有独立 run、可复现输入、环境、参数、哈希和失败记录。
3. 官方 179 例预测完成 179/179 覆盖、标签、空间和 ZIP 结构审计。
4. 用户明确授权提交，Synapse 返回有效 submission ID 与官方结果。
5. checkpoint、ZIP、源码快照、环境、日志、指标、QC 与人工复核已归档。

在 Synapse 返回成绩前，只能称为“提交包生成并通过本地技术审计”；在用户授权 179 推理前，当前项目状态只能称为“内部 S2 选择完成，E 为小病灶后续基座”。
