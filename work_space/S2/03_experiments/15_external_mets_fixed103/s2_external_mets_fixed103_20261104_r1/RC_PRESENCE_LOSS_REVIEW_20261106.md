---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# RC存在性、loss分母与小样本评价复核

本次回应：RC有无是否进入loss、类别权重分母是否加indicator、内部留出样本是否太小。只读取既有代码/历史SHA清单/manifest/评分CSV，并在已有Conda `brats2023_seg`环境做CPU合成张量检查；不读取MRI、GT体素或checkpoint，不训练、重评分、安装依赖或提交作业。详细结果、命令和源SHA见[JSON审计](RC_PRESENCE_LOSS_REVIEW_20261106.json)。RC切除腔标注不等于完整治疗前后临床状态。

## 1. 哪个实现、证据到哪里

- 原比赛E：7月21日消融批次[run_manifest](../../02_small_lesion_and_focal/s2_small_lesion_ablation_20260721/run_manifest.json)记录200 epoch、focal gamma=2、RC class weight=3。不是普通loss的Dataset264 B，也不是E_continue。
- K1：10月2日[SOURCE_CHECKPOINT_INSPECTION](../../12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/k1_resume_source_inspection_20261002_20261002T034139Z/SOURCE_CHECKPOINT_INSPECTION.json)的三个custom loss文件SHA，与当前文件逐一相同；三文件也与Git `3eeeb3c8`（7月27日）字节一致。因此下文不是只看后来文件名猜测K1行为。
- 原E历史合同及同期Git支持同一focal设计，但未取得其完整已安装nnU-Net loss源码的独立字节验收；K1的底层已安装Dice源码也未在本次验收。下文底层Dice公式依据保留的nnU-Net参考副本，明确不是新远端运行时验收。
- 外部2025模型的内部训练loss未在本次核查，不把本项目loss结论套到外部模型。

## 2. 类别权重与voxel indicator

[FocalLossMixin](../../../01_code/repository/custom_nnunet/small_lesion_trainer_mixins.py)把背景/NETC/SNFH/ET/RC权重设为`[1,1,1,1,3]`，传给CE/focal；Dice没有单独的RC权重3。Dice与focal系数各1；deep supervision另按输出层权重总和归一化，后者不是RC存在indicator。

[FocalCrossEntropyLoss](../../../01_code/repository/custom_nnunet/small_lesion_variants.py)实际计算：

```text
valid_v = (target_v != ignore_index)
L_focal = sum(valid_v * w[target_v] * (1-p[target_v])^gamma * -log(p[target_v]))
          / sum(valid_v * w[target_v])
```

有`valid`这一体素级ignore indicator，分子/分母都用它；**没有单独的病例级/类别级`I_RC = any(GT==4)`开关**。但分母不是固定`1+1+1+1+3=7`：只统计实际GT标签的权重。因此没有RC体素的patch不会平白多除一个3。没有RC仍是有效阴性监督，softmax把概率误分给RC会提高真实类别的focal损失。

CPU合成检查：4个有效体素标签`[0,1,2,3]`时分母4；`[0,1,2,4]`时分母6；`[0,4,ignore,2]`时分母5；全ignore返回0、梯度有限且被ignore梯度为0。无RC例的RC logits梯度方向会抑制RC误报。该检查不是医学图像测试或模型质量pass。

## 3. Dice有没有presence indicator

现有custom trainer调用`MemoryEfficientSoftDiceLoss(smooth=1e-5, do_bg=False)`，不传类别存在掩码。保留的[参考Dice实现](../../../../../03_reference_code/01_original_root_checkout/Segmentation_Tasks/nnUNet_install/nnunetv2/training/loss/dice.py)对前景通道计算并平均：

```text
Dice_c = (2*intersection_c + smooth) / (GT_mass_c + pred_mass_c + smooth)
L_Dice = -mean(Dice_c)
```

无RC时仍保留RC通道；GT质量为0，预测RC概率质量仍在分母，smooth避免0/0。参考源码合成测试显示RC预测质量增大时loss上升，但在大假阳性质量下该空类Dice梯度很弱，不能据“有限且方向正确”宣称稀有类训练设计已最佳。

所以用户担心的**存在类加权宏平均**，确实不是当前Dice采用的设计；但这不等于focal分母写错。若另做实验，可对GT出现的类使用`sum(w_c*I_c*L_c)/sum(w_c*I_c)`，分子分母必须同一indicator，并处理全空前景patch。同时保留所有已标注体素的CE/focal，以约束无RC病例的误报。整病例有RC不代表每个训练crop有RC；presence应与loss计算粒度一致。未知/未标注区域与“确认无RC”也必须区分。以上是候选设计，未修改loss、重训或追认旧模型。

## 4. 评价分母与RC小样本

用既有G2 `labels_present_after_overlay`登记与已完成三路MET CSV交叉，不读GT体素：

| 子集 | 扫描数 | 患者组数 |
|---|---:|---:|
| RC-present | 16 | **8** |
| RC-absent | 87 | 63 |
| 全fixed103 | 103 | 69 |

同患者可以有RC阳性和阴性时间点，两子集患者数不能直接相加。103是内部validation，不是最终锁定test104。RC阳性只有8个患者组，单看扫描均值不足以作可靠泛化/优越性断言。

现有[统一评分汇总](runtime/compare_fixed103.py)按每项metric的finite indicator计算`sum(valid*score)/sum(valid)`，配对比较取两模型共同finite；不会把NA填0或一律除103。三模型RC病灶Dice都是n=15：`BraTS-MET-01250-001`的既有CSV记1个all-instance RC、0个large-instance RC、lesion Dice为NA，与27 mm^3大小口径一致；不是漏掉一个术后患者的临床分类。

**只看RC病灶Dice会遗漏RC阴性病例上的误报，不能把n15当完整RC能力。** 原CSV仍有all-instance FP/F1，阴性误报如下，未新计算分割指标：

| 87次RC-absent扫描，MET实例口径 | 外部2025 | K1-175 | 原E-200 |
|---|---:|---:|---:|
| 有RC实例误报的扫描 | 0/87 | 4/87 | 3/87 |
| RC假阳性实例数 | 0 | 4 | 3 |
| RC all-instance F1的finite阴性扫描 | 0 | 4 | 3 |

这不是“原预测所有RC体素”的重新统计：大小、实例解析/匹配等仍遵循原MET合同。不同模型F1有效分母可能不同，横比应同时看FP、有效n及共同finite配对，不单看均值。外部2025公开训练暴露边界仍在，0/87不能当独立泛化优越性证据。

## 5. 后续需要什么

合理的描述性报告应并列：RC阳性区域Dice/漏检，RC阴性FP病例率/FP数量，有效扫描和患者数，患者组置信区间，以及是否曾用于训练/选模。临床pre/post分组另需真实逐例治疗状态，不能以RC有无替代。本次未新增bootstrap CI、评分作业或解锁104，也不改变任何旧成绩或科学状态。
