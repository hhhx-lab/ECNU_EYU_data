# TumourDiffusion 1mm：导入、对照与训练就绪检查

复核时间：2026-10-09 17:06 北京时间 / 2026-10-09T09:06Z。

状态：`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

2026-10-13历史定位补充：活动目录已由用户修订版3替换。本报告与JSON仍描述20261009原版，不代表当前修订版；用户随后明确要求删除上一版，原30源码已删除，SHA与检查报告仍保留，[删除回执](../20261013_tumour_diffusion_1mm_revision3_readiness/PREVIOUS_SOURCE_DELETION_RECEIPT.json)记录范围。[最新复核](../20261013_tumour_diffusion_1mm_revision3_readiness/README.md)单独记录修好与剩余问题。下方源码链接指向活动修订版，不是旧版原件；旧检查器硬编码活动路径，不能直接重跑它来覆盖旧JSON或冒充原版复现。

## 结论

**可以作为新研究分支的代码基础，但原样尚不满足全量训练条件。** 默认EDM网络本机真实import、64³前向、16³反向和一次优化器更新通过；数据坐标、预处理完整性和评价隔离存在已复现问题。没有实测依据认为它已优于旧150k模型，也不能把一次CPU计算通过当作A100长训练验收。

用户指定的Downloads目录已整体移动至 [G1/01_code/tumour_diffusion_1mm](../../../01_code/tumour_diffusion_1mm)，原始30文件未修改，两个原始空占位目录保留。没有权重，没有训练数据，不是已训练好的模型。新检查脚本和报告在本目录，未混入导入代码。移动后的逐文件SHA是导入基线；搬移前未单独保存SHA，故不宣称做过搬移前后独立逐字节验收。

本次没有安装依赖、远端操作、训练提交、Git操作、下载VAE重建影像或上传Drive。

## 实际输入与检查

- 新代码：[导入原件](../../../01_code/tumour_diffusion_1mm)，旧代码：[当前活动diffusion_gliGAN](../../../01_code/diffusion_gliGAN)。旧活动代码与历史冻结150k runtime不是同一概念。
- 原训练数据合同：[926例成员表](../20260720_v3_multiserver/canonical/splits/current/dataset_membership.csv)，train823/val103，锁定test104不参与训练。
- S2原E空间：[nnUNetPlans.json](../../../../S2/03_experiments/01_baselines_and_original_e/realonly_fold0/nnUNetPlans.json)。
- 检查环境：借用既有Conda `g1_t2w_bbdm`，Python3.10.20、torch2.7.1、MONAI1.5.2，CPU，无CUDA。没有变更这个环境。
- 方法：23份Python AST、`bash -n`、真实import、临时合成NIfTI/RC病灶、真实ROI/crop、参考通道网络前向/反向、源码恢复合同和split分支复核。临时合成数据自动清理。

机器证据：[READINESS_EVIDENCE.json](READINESS_EVIDENCE.json)；原件基线：[IMPORT_MANIFEST.json](IMPORT_MANIFEST.json)；搬移回执：[IMPORT_RECEIPT.json](IMPORT_RECEIPT.json)；逐文件新旧SHA：[SOURCE_COMPARISON.json](SOURCE_COMPARISON.json)。

复现命令，从项目根执行：

```bash
conda run --no-capture-output -n g1_t2w_bbdm python -B work_space/G1/03_experiments/03_diffusion_training/20261009_tumour_diffusion_1mm_readiness/audit_readiness.py
```

检查器退出0表示取证过程成功，不表示模型训练门通过；最终状态字段是 `not_ready_for_full_training`。重新运行会刷新本检查目录的JSON，不改导入源文件。

## 已通过什么

| 项目 | 实际结果 | 边界 |
|---|---|---|
| 语法 | 23份Python和四GPU shell语法pass | 语法不等于运行合同 |
| 真实import | trainer、loader、生成、评价与online入口pass | 仅上述本机环境 |
| 默认EDM网络 | 4,475,999参数，channels48/96/192/384；64³前向finite | batch1、CPU、非AMP/compile |
| 反向更新 | 16³输入，4通道条件含RC；loss13.07339，梯度finite，一次AdamW更新 | 不是完整64³/batch16训练压力测试 |
| RC条件表示 | BRATS_2024产生TC/WT/ET/RC四通道，RC可进入网络 | 不代表CSV及供体位置正确 |
| 缓存保留 | 测试中随机crop不修改缓存label | 不是全部并发/worker稳定性证明 |

MONAI提示planner提供了多余的末尾stride=1，被忽略；默认网络仍可计算。这是应规范的参数合同，不是本次最主要阻塞。

## 必须先解决的阻塞

### 1. CSV与tile交换第一、第三轴，实际病灶未进入crop

[csv_creator.py](../../../01_code/tumour_diffusion_1mm/src/train/csv_creator.py:18)把Nibabel `(x,y,z)`当成`(z,y,x)`。已复现：正确bbox `[10,34,4,9,2,7]`变成`[2,7,4,9,10,34]`，质心`[22,6,4]`变成`[4,6,22]`。即使传入正确bbox，[tile切片](../../../01_code/tumour_diffusion_1mm/src/train/csv_creator.py:97)仍返回0个tile，对照应为2个。

更直接的真实transform测试：在96³中放216个RC体素，导入CSV坐标经过ROI/crop后RC=0，修正XYZ的对照RC=216，二者都能返回64³数组。**程序不一定崩溃，却会训练错误目标。** 应修坐标、加非对称坐标/大病灶覆盖测试，并重新生成CSV；不能仅改注释或继续使用错轴CSV。旧活动代码已有正确XYZ实现及回归测试。

### 2. 预处理未兑现严格1mm及输入完整性合同

[new_affine](../../../01_code/tumour_diffusion_1mm/scripts/preprocess_dataset.py:59)按裁剪shape/取整后shape计算有效spacing。合成例目标1mm，实际NIfTI header为 `[1.02857149,0.96923077,1.00740743]` mm。该写法对应resize像素中心/FOV映射，不能一边保留它一边宣称精确1mm；必须定义真实目标网格、对应采样坐标、回写与逆变换，不能只把header改成1mm。

[process](../../../01_code/tumour_diffusion_1mm/scripts/preprocess_dataset.py:77)只核对shape。已复现T2W affine偏移10mm仍status=ok；非法label9保存为9；输入NaN仍status=ok且输出非finite。需要四模态与seg的shape/affine/spacing一致性、finite、合法整数标签0..4、重复病例冲突和源SHA验收。病例缺失/处理失败目前仅WARN，已有输出直接skip，也没有全量严格完成标记；正式数据生成必须失败即阻断并保留证据。

1mm指体素物理间距，不是统一脑配准、相同shape、相同方向或统一像素强度。整例masked zscore解决的是强度表征，不解决几何错位。

### 3. split与T2W名单必须绑定现有项目，评价不得回退train

[评价入口](../../../01_code/tumour_diffusion_1mm/src/infer/evaluate_generation.py:489)默认val不存在时改为评价所有行。两行train合成表请求val后仍评价train。应缺val或缺split立即拒绝，而不是输出貌似验证指标。

CSV未提供val名单时默认全部train；loader的`--use_all_rows`还会取消train/T2W筛选。应冻结patient-group成员表、train823/val103/test104及源标签版本，拒绝交叉污染。test104不能用来挑点。

导入 `t2w_invalid_cases.txt`有266例，项目既有缺失真实T2W口径为265；本次对照旧完整926例未发现名单交集，但未覆盖全部1030例，所以仍须与项目权威QC逐例核对，不能直接接受266或按总数随意删1例。

### 4. 长训练运行、恢复与S2适配尚未就绪

[checkpoint保存](../../../01_code/tumour_diffusion_1mm/src/train/tumour_main_diffusion.py:409)缺AMP scaler、RNG和数据SHA绑定；每10次loader遍历直接torch.save，无原子写/TERM保存。开启compile保存的state_dict与切换compile恢复还需测试。旧活动trainer已有原子保存、scaler恢复、固定step保存及信号处理，不应在新分支丢掉这些能力。

`global_step`在更新后先判上限再自增，存在多更新一步而计数未增的风险，应通过精确步数与恢复测试修正。非默认lognsr分支调用不存在的`lognsr_loss_fn`，实际名称`logsnr_loss_fn`；默认EDM不受该拼写阻塞，但不能宣称所有schedule均可用。

[train_4gpu.sh](../../../01_code/tumour_diffusion_1mm/scripts/train_4gpu.sh)是本机nohup并行脚本，不是ECNU Slurm提交包；固定GPU编号、8workers/路、batch16、compile都未经服务器测试，`wait`未逐PID汇总失败。需独立slurm、日志与退出码检查、GPU/CPU/内存预算、环境锁和一次性launch审计，再先单卡smoke。

原E/S2计划spacing为 `[1.0,0.8593999743,0.8593999743]`（S2自身轴序），**不是1×1×1mm**。还要验证Nibabel XYZ与nnUNet数组轴序、目标/供体physical grid、归一化、support外逐体素不变以及四模态/RC guardrail。不能因为能import online入口就直接替换K路线生成器。

## 与之前的扩散模型有什么不同

| 维度 | 既有已选150k路线/当前活动代码 | 导入1mm路线 | 判断 |
|---|---|---|---|
| 生成目标 | 四个独立模态模型，局部病灶图像 | 仍四个独立模态模型 | 不是联合四模态新架构 |
| 扩散方法 | EDM、标签条件、Heun18、64³；非VAE latent | 默认同族EDM/标签条件/64³，可选其他backbone | 不是从latent扩散升级像素扩散，旧线本来就在影像空间 |
| 空间 | 原真实MRI/crop合同 | 新增crop+重采样预处理，目标各向同性1mm | 相同crop对应较一致物理范围；实现需修 |
| 强度 | 既有zscore crop合同 | 支持整例masked zscore且crop不再归一化 | 有望减轻训练/增强表征差异，但不自动等同现S2 |
| 缓存 | 原训练loader与历史资源合同 | 只缓存约80³ ROI、随机shift在缓存后重新抽取 | 内存/加载速度潜在优势，尚无A100基准 |
| 网络规模 | 当前活动代码也有planner及RC条件 | shell锁planner48GB，channels48/96/192/384 | planner/RC支持不是全新发明 |
| RC | 旧活动converter已有label4；S2曾另受供体/生成合同限制 | BRATS_2024四条件通道、CC包含RC | 不等于修好S2供体、QC或病灶crop |
| 预算与sigma | 历史保留150k，绑定其checkpoint与QC | shell建议200k、sigma_data=1 | 更多step和参数变化不是质量证明 |
| 恢复保障 | 当前活动版本已有原子checkpoint/scaler/信号处理 | 导入版缺这些保障 | 工程能力回退，需补齐 |
| 权重/实测 | 既有150k权重及G2质量证据 | 此包无checkpoint/数据/新实测 | 新模型优势尚不能下结论 |

要证明优势，应固定真实train823和val103、seed/预算/网络/采样器，分别比较旧表征与严格1mm新表征。全图和病灶区指标、大小病灶、RC、跨模态一致性、FP/FN与下游官方MET分开报告。checkpoint选择与报告若用同一fixed103，仍只能exploratory。VAE epoch4重建替换真实影像又是第二个因素，应独立实验，不能与1mm一起改后归因于某一个机制。

## 推荐投入训练的顺序

1. 修XYZ/tile并重建CSV；冻结真实病例split、标签版本和T2W质量名单。
2. 定义严格1mm网格、方向、强度与回写合同；小批真实数据逐文件SHA/finite/geometry/标签与小病灶保留验收。
3. 补原子恢复、AMP/RNG/数据身份、精确step、失败即退出与评价隔离；加独立Slurm运行包。
4. 单A100做短训练、暂停/恢复等价性、生成与四模态/RC检查，测实际显存、吞吐及worker资源。
5. 前述通过后才冻结新实验批次并提交四路训练；原150k与所有失败证据保持独立。

175 epoch属于S2分割路线，不应自动挪到本Diffusion；本包按optimizer step定义预算，应另行预注册150k或200k及选点规则。

## 目录与文件用途

```text
work_space/G1/
  01_code/
    diffusion_gliGAN/              旧Diffusion活动实现，未替换
    tumour_diffusion_1mm/          本次整体导入原件
      model.py                    扩散schedule、loss、sampler
      requirements.txt            上游依赖声明，未安装/未锁服务器版本
      README*.md, docs/           上游说明与加速讨论，断言以本审计为准
      scripts/                    预处理、名单、四GPU启动与训练观察
      src/preprocess/             nnUNet风格crop/normalize/resample
      src/train/                  病灶CSV、影像训练、独立标签训练
      src/utils/                  数据loader、ROI/crop/noise、标签转换
      src/networks/               时间条件MONAI/可选PlainConv网络
      src/infer/                  标签条件生成、评价、online增强
      lists/                      待项目QC对齐的T2W排除名单
      Checkpoint/, DataSet/       原始空占位，不是权重/数据仓库
  03_experiments/03_diffusion_training/
    20261009_tumour_diffusion_1mm_readiness/
      README.md                   本次结论、差异、门控与定位入口
      audit_readiness.py          不改原件的可复现CPU检查器
      IMPORT_RECEIPT.json         搬移事实与核验边界
      IMPORT_MANIFEST.json        30个原文件大小/SHA与空目录
      SOURCE_COMPARISON.json      与旧活动源码逐文件SHA对照
      READINESS_EVIDENCE.json     10组检查结果及环境/操作边界
```

逐文件完整定位通过项目总纲当前索引维护；本报告不改写上游长README来掩盖原实现问题。
