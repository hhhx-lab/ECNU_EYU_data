**一句话结论**

FixV3 使用的是：

> **冻结 E 分割模型权重 + MET-AUG Route A + Candidate A 的标签区域硬写入 + FixV3 多级质控。**

它不是把 A-F 全部叠加，也没有采用标签置换、尺度缩放或多病灶增强。

**一、训练方式**

FixV3 从原始 E checkpoint 加载网络权重，重新建立优化器和学习率计划，从 epoch 0 独立训练到 200：

- `resume=false`，但不是随机初始化；
- 初始学习率 `0.001`；
- seed `20260724`；
- 每 25 epoch 保存；
- 单张 A100；
- 固定训练集 1035 例，固定验证集 103 例；
- FixV3 增强只在训练阶段启用；
- 验证、推理时完全关闭，不加载 Diffusion，也不读取 donor。

**二、每个训练 patch 如何增强**

FixV3 在 nnU-Net 常规空间和强度增强之前执行一次在线增强事务：

1. 每个训练 patch 以 `p_select=0.20` 的概率尝试增强。
2. 未选中时，patch 原样进入 nnU-Net。
3. 选中后，从训练集真实病灶组件池抽取一个 donor。
4. 在目标 patch 的合法脑区寻找安全位置。
5. 将 donor 的原始标签平移到该位置。
6. 调用四个冻结的 G1 Diffusion 模型，分别生成 T1N、T1C、T2W、T2F 病灶影像。
7. 通过 FixV3 QC 后，同时写入四模态和标签。
8. 任一步失败，整个事务回滚，patch 完全不变。

因此它是“训练时实时生成病灶”的在线增强，不是提前生成一批完整病例。

**三、病灶供体怎么构造**

供体只能来自固定的 1035 例训练集，不能来自验证集或测试集。

- 用 `NETC(1) ∪ ET(3)` 做 26 连通域分解；
- 每个连通组件作为一个独立病灶；
- SNFH(2) 只有在能够唯一归属某个核心时才附着；
- 同患者不同时间点也禁止互作 donor/target；
- 排除含 RC(4) 的组件；
- 排除纯 SNFH；
- 排除核心体积 `<27 mm³`；
- 排除任一 bbox 维度 `>56 mm`；
- 按 `classes_present × core_volume_bin` 联合分层抽样；
- 对 `27–275 mm³` 的可学习小病灶给予温和的额外抽样权重。

这不是整例复制，而是从真实病例中拆出一个可追溯的真实单病灶组件。

**四、标签到底有没有置换**

没有。

最终 FixV3 明确采用：

- `scale=1.0`；
- `max_tumours=1`；
- `p_second=0`；
- `preserve_classes=true`；
- 不执行 `SNFH→ET`；
- 不执行 `ET→NETC`；
- 不腐蚀；
- 不膨胀；
- 不创建人工 NETC 核；
- 不创建人工 SNFH 壳。

插入后的类别组成就是 donor 原来的组成。

截图中记得的 `SNFH→ET`、`ET→NETC`、70% 概率、缩放和第二病灶，属于早期胶质瘤方案及后续 A-F 消融设想，最终 FixV3 没有使用。

**五、病灶如何放置**

每次最多搜索 50 个位置，并满足：

- 位于 Dataset264 的显式有效脑区；
- 距脑边界至少 `3 mm`；
- 距原生病灶至少 `5 mm`；
- 不覆盖现有标签；
- 最终病灶 bbox 能放入 `64³` Diffusion crop；
- donor 与 target 不能来自同一患者组。

原病例中的病灶不会被替换或删除，只是在合法位置增加一个新病灶。

**六、四模态 Diffusion 生成**

四个模态分别使用冻结的 G1 150k checkpoint：

- T1N；
- T1C；
- T2W；
- T2F。

冻结合同为：

- z-score 输入空间；
- `64×64×64` crop；
- EDM-Heun；
- 18 个采样步骤；
- 四模态使用同一病灶标签和同一空间坐标；
- 显式处理 G1 与 S2 的通道顺序差异。

Diffusion 只负责生成新插入病灶对应的四模态影像，不用于生成整幅 MRI，也不同于 G1-r4 的“缺失 T2W 补全”。

**七、Candidate A 写入策略**

另一个容易混淆的 A/B/C，是写入边界候选：

- Candidate A：只在病灶标签 support 内硬写入；
- Candidate B：增加 halo 和 cosine blending；
- Candidate C：在 B 上再做 median/MAD 强度 harmonization。

最终选择 Candidate A：

- A 和所有 B 半径均为 `46/48` 通过；
- 按预注册规则，平局选择更简单、更快的 A；
- 所有 C 配置因 harmonization 问题 `48/48` 失败。

所以最终行为是：

> 影像和标签只修改病灶 support 内体素；support 外保持逐体素不变；无 halo，无强度 harmonization。

**八、FixV3 的主要变化**

FixV3 的“V3”主要是质控策略升级，不是第三套标签增强。

它检查：

- Diffusion 原始残差分位数；
- 极端残差的空间形态；
- 每个标签、每个模态的边界连续性；
- candidate 残差保留率与 Q99 强度；
- 四模态类别对比向量；
- Mahalanobis 距离；
- 四模态异常区域 IoU 和质心距离；
- 大 ET 在 T1C 上显著性过低；
- 多个相关软异常是否共同形成拒绝条件。

明确的形态错误、严重异常或多项相关失败会拒绝；单个轻微软异常不再机械地一票否决。

**九、原子提交机制**

四个模态和标签必须全部成功才提交：

- 四模态均 finite；
- shape、坐标和 support 一致；
- 标签仅包含合法类别；
- support 外影像和标签完全不变；
- 审计记录写入成功。

否则返回原 patch 的 bit-identical 副本，绝不留下“只写了两个模态”或“影像写了但标签没写”的半成品。

**十、实际 full-200 增强统计**

200 epoch 共审计 100,000 个训练 patch：

| 项目 | 数量 |
|---|---:|
| 未选中 | 80,000 |
| 尝试增强 | 20,000 |
| 成功提交 | 15,224 |
| 总体实际提交率 | 15.224% |
| 选中后成功率 | 76.12% |

主要拒绝原因：

- 原始 Diffusion QC：2,779；
- 边界 QC：1,928；
- 无合法位置：37；
- 边界支持不足：29；
- 内容 QC：2；
- 无合格 donor：1。

成功提交的真实标签组成：

| 标签组成 | 数量 |
|---|---:|
| ET | 6,362 |
| SNFH + ET | 5,932 |
| NETC + ET | 939 |
| NETC + SNFH + ET | 1,969 |
| NETC | 22 |

这些组成均来自原始 donor，不是标签置换产生的。

**A-F 最终关系**

| Route | 变量 | FixV3 是否使用 |
|---|---|---|
| A | 真实单病灶平移 + 四模态 Diffusion，`p=0.20` | **使用** |
| B | 按核心体积缩放 | 未使用 |
| C | 比较 `p=0.20/0.40/0.60` | 未使用 |
| D | 尝试第二个独立病灶 | 未使用 |
| E | 受保护的 SNFH/NETC 拓扑构造 | 未使用 |
| F | 按原生病灶负荷自适应分配概率 | 未使用 |

所以最准确的命名是：

> **FixV3 = E 权重初始化的 MET-AUG-A 单病灶在线四模态生成，采用 label-only hard commit，并增加 FixV3 QC。**

依据可见 
[在线增强规范](ECNU_EYU_data/work_space/S2/docs/ON_THE_FLY_AUGMENTATION.md)、
[实际训练 provenance](ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_fix_v3_full200_attempt11_20260823_r1/artifacts/nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/met_aug_provenance.json) 和 
[完整事件审计](ECNU_EYU_data/work_space/S2/results/experimental_unvalidated/s2_met_aug_fix_v3_full200_attempt11_20260823_r1/artifacts/nnUNet_results_attempt11/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCMetAugFixV3EmergencyFocalCompletionFineTune__nnUNetPlans__3d_fullres/fold_0/met_aug_events.jsonl)。

需要保留的结论边界是：该 checkpoint 已训练完成，但仍标记为 `experimental_unvalidated`，正式 Gate 尚未通过。