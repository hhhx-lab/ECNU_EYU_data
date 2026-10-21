# TumourDiffusion 1mm 修订版3替换与复核

复核日期：2026-10-13，北京时间。状态：`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

## 当前结论与定位

2026-10-13后续定位：本报告为修订ZIP3的历史检查，不再表示活动源码。用户随后提供日期改进包，活动37文件和新检查见[日期改进版审计](../20261013_tumour_diffusion_1mm_dated_revision_readiness/README.md)；本版31源码已移至该审计的`previous_source_revision3`并核对SHA，旧指标和本次回执不改写。以下结果仍只描述原ZIP3。

**修订版已替换活动目录，核心坐标/输入检查确实改善，但仍不宜直接全量训练。** 完成15组本机检查：7组pass、4组fail、3组partial、1组unresolved；失败是已复现产品行为，不是检查器崩溃。没有提交训练、安装依赖、下载真实影像或修改修订版源文件。

| 内容 | 路径 |
|---|---|
| 当前活动修订版 | [G1/01_code/tumour_diffusion_1mm](../../../01_code/tumour_diffusion_1mm) |
| 上一版源码 | 用户明确要求后已删除；[删除回执](PREVIOUS_SOURCE_DELETION_RECEIPT.json)保留原30文件SHA |
| 修订版原始ZIP副本 | [TumourDiffusion_1mm_revision3_original.zip](TumourDiffusion_1mm_revision3_original.zip) |
| 替换前后SHA、文件差异、来源 | [REPLACEMENT_RECEIPT.json](REPLACEMENT_RECEIPT.json) |
| 本次测试及环境证据 | [REVISION_READINESS_EVIDENCE.json](REVISION_READINESS_EVIDENCE.json) |
| 原版历史检查 | [20261009报告](../20261009_tumour_diffusion_1mm_readiness/README.md) |

用户指定ZIP为 `/Users/hwaigc/Downloads/TumourDiffusion_1mm(3).zip`，SHA256=`739bdb10813a7a2a62e49eb654f51124b3c0272c00ffaa2a105ca6f9a1103236`。43个ZIP条目、531803解压bytes，31份文件。导入时检查路径越界、重复项、symlink、加密及体积限制；活动文件逐SHA与ZIP一致。旧30文件与20261009基线一致，替换时曾完整保存；用户随后明确要求删除上一版，已于2026-10-13T03:18:44Z删除这30文件共491891 bytes。当前31源码SHA未改，旧检查报告/替换回执和原始新版ZIP保留，未删除旧150k模型或其他Diffusion路线。

修订版改变7文件：`README.md`、预处理脚本、评价脚本、CSV创建器、影像trainer、loader、ROI transform；新增`smoke_test.sh`；其余23文件未变，没有从旧活动版删除文件。`model.py`和网络未变，仍是四个独立单模态EDM，不是新联合四模态架构，也不自动证明质量更优。

## 确认修好的部分

| 上次问题 | 本次实际验证 | 结果 |
|---|---|---|
| CSV第一/第三轴交换 | 非对称NIfTI数组bbox=`[10,34,4,9,2,7]`、centroid=`[22,6,4]` | pass |
| 大病灶tile错轴 | 16³/stride12返回2个tile，400/300体素 | pass |
| RC216体素被crop丢光 | 24次随机offset均保留216，缓存label不变 | pass |
| 模态错位10mm被接受 | 实际`process`返回rejected | pass |
| NaN、非法label9、非整数label2.5 | 四类输入分别被拒绝 | pass |
| 默认网络计算 | 4,475,999参数，64³前向finite、16³反向梯度finite并更新 | pass，CPU batch1 |
| checkpoint非原子写 | 模拟保存中断，已有完整checkpoint内容不变 | pass，未完成tmp保留 |
| AMP状态丢失 | 实际load_ckp恢复scale256，而非默认65536 | pass，CPU scaler模拟 |
| CSV选择随目录枚举变化 | 新增确定的`resolve_csv_path`与CSV SHA身份记录 | 源码确认改善，身份范围仍需补齐 |
| val不存在回退train | 有split列且只有train时明确退出 | 修好该分支，但缺列仍漏检 |

额外真实NIfTI -> CSV -> CacheDataset -> loader检查通过：两例2行中保留train1行，返回`[1,1,64,64,64]` batch，RC64体素存在。这个合成例不是项目真实患者或真实GPU训练验收。

## 剩余问题：按优先级

### P1：预处理失败仍返回成功，已有坏文件被当成可复用

[预处理完成逻辑](../../../01_code/tumour_diffusion_1mm/scripts/preprocess_dataset.py:205)仅打印WARN：只指定NaN病例时最终exit=0，指定不存在的必需病例时0病例输出仍exit=0。[skip路径](../../../01_code/tumour_diffusion_1mm/scripts/preprocess_dataset.py:101)只看5文件存在；已将一份输出替换成坏NIfTI，仍返回`skipped (exists)`。

新增输入拒绝是正确改进，但流程必须非0退出/严格COMPLETE与逐文件manifest，不能让Shell或调度器把不完整数据记成成功。重复病例来源也仍仅WARN取第一份。CSV对非法标签只SKIP/WARN，可能静默改变训练分母，需绑定预注册成员表后严格检查。

### P1：缺split列和跨访视患者泄漏仍可产生虚假验证

[评价split分支](../../../01_code/tumour_diffusion_1mm/src/infer/evaluate_generation.py:499)要求`split`列已存在才执行检查；缺列时请求val仍保留所有行。loader也有同类缺列放行行为。

[CSV拆分](../../../01_code/tumour_diffusion_1mm/src/train/csv_creator.py:245)只按完整case/带末尾访视suffix的ID匹配。实际用同患者`00001-000`与`00001-001`做测试，指定后者val后，前者train、后者val。**它不是patient-group隔离实现**；应使用本项目权威group split，并拒绝跨train/val/test患者。

[评价CSV身份校验](../../../01_code/tumour_diffusion_1mm/src/infer/evaluate_generation.py:431)只检查第一个找到的模态checkpoint；CSV不同仅WARNING继续，并非完整四路拒绝门。应逐模态绑定数据/空间/强度合同，明确探索性重评与验证用途，禁止默认容忍未声明的训练交集。

### P1：训练步数仍多更新一次；精确预算和恢复未兑现

[trainer步数判断](../../../01_code/tumour_diffusion_1mm/src/train/tumour_main_diffusion.py:232)在optimizer更新后、计数增加前检查上限。运行真实train循环，目标2 steps、提供5个batch时，实际执行3次optimizer.step，返回global_step仍2。这会让训练预算与记录不一致；从checkpoint再跑也有同类边界风险。应按已完成更新计数，在更新前判断上限，增加精确停止/恢复测试。

[训练checkpoint](../../../01_code/tumour_diffusion_1mm/src/train/tumour_main_diffusion.py:446)仍每10次loader遍历保存、无TERM安全保存/固定step间隔/RNG与sampler状态/训练seed入口。原子写和scaler恢复已经修好，但**不是完整可重复续训合同**。AMP跳过非finite更新时的计数与异常退出也需验证。

### P1（服务器启动前）：新smoke脚本会删除既有证据

[smoke_test.sh](../../../01_code/tumour_diffusion_1mm/scripts/smoke_test.sh:18)固定`Checkpoint/zz_smoke`且执行`rm -rf "$WORK" "$CKPT"`，末尾默认再次删除；`WORK`可来自外部环境。这违背本项目失败/历史证据保留约束，不能直接执行。应独立时间戳attempt、存在即拒绝覆盖、禁止自动删除、保存日志和门控回执。

该脚本只用`grep ',ok,'`判预处理，至少一例ok就放行，并非全部病例通过；仅训练t1c/t2w两模态，不是四模态训练/生成验收。未运行其破坏性流程；`bash -n`通过不代表可以安全启动。

### P2：header现在是1mm，但真实采样坐标仍不是同一严格网格

[new_affine](../../../01_code/tumour_diffusion_1mm/scripts/preprocess_dataset.py:58)改为`target_spacing/spacing`写header，数组仍由旧`shape/new_shape`像素中心映射重采样。合成例header=`[1,1,1]`，实际resize spacing=`1.0286×0.9692×1.0074` mm，8角点最大采样位置与保存world坐标差=`0.558017` mm。这是取整例的可复现实测，不是所有脑影像的固定误差。

如目标仅是nnUNet数组的**名义spacing约定**，可以明确记录实际映射与nominal grid；如要严格物理1mm与原空间可靠回写，则必须按目标affine真正采样并保存完整逆变换，不能只修改header。报告目前只存shape字符串和true_spacing，未保存逐例bbox/原affine/新affine等完整回写合同。原E/S2 spacing仍不是1×1×1mm，两者还需适配。

### P2：compile恢复、数据身份、可选schedule仍有缺口

- [load_ckp](../../../01_code/tumour_diffusion_1mm/src/train/tumour_main_diffusion.py:54)直接加载state_dict。模拟真实compile保存的`_orig_mod.*`键，恢复到未compile模型时missing/unexpected keys报错；四GPU脚本默认compile，需解包统一保存或冻结相同compile模式并验证。
- CSV SHA确实新增，但不是NIfTI/seg SHA。实际原CSV不变、指向影像内容改变后，`csv_identity`仍相同。新输入/标签/预处理合同必须逐文件绑定，并校验resume的网络、schedule、sigma、归一化、crop、病例成员。
- [lognsr分支](../../../01_code/tumour_diffusion_1mm/src/train/tumour_main_diffusion.py:214)仍调用不存在的`lognsr_loss_fn`，定义名称为`logsnr_loss_fn`。默认EDM不受影响，但该可选路径不可用。
- T2W排除名单仍266条，未变；需与项目权威QC的265口径逐例核对，不随意按数量删1条。
- 四GPU脚本仍nohup固定GPU编号、compile、batch16及8workers/路，不是ECNU独立Slurm/排他launch合同；A100显存/吞吐、完整64³反向、CUDA scaler/compile及四路压力未验证。

## 推荐下一步

先修失败退出与skip验收、split/group隔离、精确step和smoke保留策略；明确物理1mm或名义nnUNet grid的空间目标，再补恢复/数据SHA与服务器独立Slurm。单A100安全smoke及真实小批数据通过后才冻结全量训练。错轴旧CSV必须重建，旧错轴训练checkpoint不可作为等价初始化续用。

本次是替换和检查，不擅自把用户提供的修订版进一步改成另一个版本。VAE epoch4影像仍按用户“先别拉回”不下载；重建图与真实图、1mm与旧空间、G1图像质量与S2分割收益仍分开。

## 复现与证据维护

从项目根用既有Conda `g1_t2w_bbdm`执行：

```bash
conda run --no-capture-output -n g1_t2w_bbdm python -B work_space/G1/03_experiments/03_diffusion_training/20261013_tumour_diffusion_1mm_revision3_readiness/audit_revision.py
```

CPU环境torch2.7.1、MONAI1.5.2，无CUDA。检查器退出0表示取证过程成功，最终readiness仍`not_ready_for_full_training`；真实影像未被读写，合成临时数据自动清理。正式结果15组，无probe_error；初次检查器因新ROI要求n_voxels和小网络调用签名不匹配的回执保留为`AUDIT_HARNESS_INITIAL_SCHEMA_MISMATCH.json`，不把它当作模型失败。此前13组核心结果保留`REVISION_CORE13_EVIDENCE.json`。

`import_revision.py`是一次性替换器，发现历史区/回执存在即拒绝再次覆盖。旧20261009审计与SHA保持历史语义，但原版源码已按用户要求删除，不再具备原版完整本地复现条件；不要在已替换活动路径上重跑旧检查器冒充旧版复现。删除范围与验收见 `PREVIOUS_SOURCE_DELETION_RECEIPT.json`。

## 问题严重性说明

最严重的是**可能悄悄污染结果或破坏证据**：缺split/group检查可造成验证泄漏，数据失败exit0/坏输出skip可让不完整数据进入训练，smoke的rm -rf可能删除既有内容。这三类必须在正式训练前处理，但不代表当前数据已经泄漏或遭到删除。

其他问题有条件性：1mm header/采样映射差异对严格物理回写和S2接入重要；仅研究名义nnUNet数组时应明确空间口径，不等于网络不能训练。compile恢复故障发生于切换compile模式，固定相同模式并通过恢复测试可避免该特定问题；RNG/TERM缺口仍影响长任务可靠恢复。step多更新一次是确定的计数bug，但不是多训练50%：本测试2步变3步，200k预算的对应边界误差通常只是末尾额外一次，不能把小测试比例外推。lognsr拼写只影响可选分支，默认EDM不受它影响。

结论不是模型架构报废或需要推倒重来。当前修好坐标后的默认网络和数据loader有实际CPU可用证据；可以继续修正并做隔离短smoke，但正式全量训练和可信评价须先通过前述数据、split、安全与运行合同门。
