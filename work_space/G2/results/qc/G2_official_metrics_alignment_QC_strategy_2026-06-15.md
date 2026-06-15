# G2 官方检测策略完整总结与 QC 对齐方案

更新日期：2026-06-15  
适用范围：BraTS 2026 Challenge Task 1 Brain Metastases、G2 synthetic data QC、G1/G2/S1/S2 对接、real-only vs real+synth 消融报告

## 0. 来源锁定

本文件只按 2026 Task1 口径整理，不混用往年 Task7/Task8 或其他 BraTS 任务口径。

1. 官方任务页面：<https://challenges.synapse.org/Challenges/DetailsPage/Task1?id=syn74274097#Data%20Files>
2. 官方评估实现仓库：<https://github.com/BraTS/BraTS_evaluation>
3. 官方 MET config：`brats_evaluation/configs/config_mets.yaml`
4. 官方 MET parser：`brats_evaluation/metrics_parser.py`

读取结论：

1. Synapse 页面确认为 `BraTS 2026 Challenge` 的 `Task 1`，任务是 Brain Metastases 自动检测与分割。
2. Synapse 页面 Evaluation 部分写明分割评估使用 DSC、NSD，检测评估使用 F1。
3. Results leaderboard 当前字段与用户给出的 `lesionwise_*`、`small_instance_*` 字段一致。
4. 官方评估包 README 明确说明 `BraTS_evaluation` 是 BraTS segmentation challenges 的 official implementation。
5. G2 所有 QC 与训练后验收都必须围绕 2026 Task1 的 ET/RC/TC/WT 区域和 small-instance 检测指标展开。

## 1. 2026 Task1 任务与数据约束

### 1.1 任务目标

参赛算法需要开发通用自动分割算法，可靠检测并精确勾画不同大小的脑转移瘤，适用于治疗前和治疗后病例。

G2 的含义：

1. synthetic data 不能只服务大肿瘤外观；必须特别关注小病灶检测。
2. post-treatment 场景中的 RC 不能被忽略。
3. 训练增强不能让 S1/S2 在真实验证 fold 上出现小病灶 FP/FN 失控。

### 1.2 官方四标签系统

| label | name | G2 QC 含义 |
|---:|---|---|
| 0 | background | 背景 |
| 1 | NETC | non-enhancing tumor core，非强化肿瘤核心 |
| 2 | SNFH | surrounding non-enhancing FLAIR hyperintensity，T2/FLAIR 高信号区域 |
| 3 | ET | enhancing tumor，强化肿瘤 |
| 4 | RC | resection cavity，切除腔 |

G2 强制规则：

1. synthetic `seg` 必须是单通道整数标签。
2. 值域只能是 `{0,1,2,3,4}`。
3. RC 是治疗后语境标签，不能无 source 依据凭空生成。
4. 不能照搬旧 glioma 的三区域口径把 RC 混进 TC/WT。

### 1.3 官方数据特点

Synapse Task1 页面给出的关键点：

1. 2026 Task1 使用 2025 BraTS-METS 数据。
2. 数据包含 pre-treatment 和 post-treatment mpMRI。
3. 数据来自多个机构、设备和协议，图像质量和空间差异很大。
4. 2025 起 T2W 非强制：有的病例是 native T2，有的病例是 synthetic T2，有的病例可能没有 T2。
5. 数据空间混合：native、T1C 1mm 共配准、SRI24 空间混合存在。
6. 官方会对标注做 QC，并提供 corrected labels。
7. validation/test 真实标签不公开。

G2 强制规则：

1. 跨病例不能写死 shape，例如不能默认全是 `240x240x155`。
2. 病例内四模态和 seg 必须 shape/spacing/affine 一致。
3. 不能用 validation 病例做 synthetic source。
4. corrected labels 必须优先于 raw seg。
5. T2W fake/native 状态可作为数据报告证据，但不能作为是否可训练的唯一判据。

## 2. 官方评估总体流程

官方公开流程分两步。

### 2.1 先运行 Panoptica 评估

官方 README 示例：

```bash
brats-evaluate \
  --config mets \
  --ref_path /path/to/reference/niftis/ \
  --pred_path /path/to/prediction/niftis/ \
  --summary_json ./panoptica_evaluation_summary.json
```

含义：

1. `ref_path` 是参考标签目录。
2. `pred_path` 是预测标签目录。
3. `--config mets` 使用 MET task config。
4. 输出是 Panoptica JSON，不是最终 leaderboard CSV。

G2/S1/S2 要求：

1. 训练后评估必须固定同一真实 validation fold。
2. prediction filename 必须能和 reference filename 一一对应。
3. prediction label 值域必须是 `{0,1,2,3,4}`，不能输出 one-hot 或 region mask。

### 2.2 再解析 MET leaderboard 字段

官方 README 的 MET parser 示例：

```bash
brats-parse-metrics mets \
  --json_path ./panoptica_evaluation_summary.json \
  --vol_threshold 20.0 \
  --overlap_threshold 0.1 \
  --output_csv_path ./parsed_panoptica_mets_stats.csv
```

官方 MET notebook 示例使用：

```bash
brats-parse-metrics mets \
  --json_path ./sample_mets_metrics.json \
  --vol_threshold 27 \
  --overlap_threshold 0.2 \
  --output_csv_path ./sample_mets_summary.csv
```

G2 内部代理口径：

1. 先用 `vol_threshold=27`，因为 Synapse Task1 明确强调小于 `27 mm3` 的小病灶检测。
2. 先用 `overlap_threshold=0.2`，对齐官方 MET notebook 示例。
3. 最终提交成绩以 Synapse scorer/container 实际参数为准。
4. 报告必须写明这是 `official-style proxy`，除非直接使用官方 Synapse scorer 或官方发布的完整 container。

## 3. 官方 Panoptica MET config

官方 `config_mets.yaml` 的关键配置如下。

### 3.1 输入类型

```text
expected_input: SEMANTIC
```

含义：输入是单通道语义分割标签，不是实例 ID map，不是 one-hot。

G2 检测：

1. synthetic label 必须是 3D 单通道 NIfTI。
2. S1/S2 输出也必须是 3D 单通道 NIfTI。
3. label 值域超出 `{0,1,2,3,4}` 直接不合格。

### 3.2 官方区域映射

| 官方 region | 官方 config label | G2 实现 |
|---|---|---|
| `et` | `[3]` | `label == 3` |
| `rc` | `[4]` | `label == 4` |
| `tc` | `[1,3]` | `label in {1,3}` |
| `wt` | `[1,2,3]` | `label in {1,2,3}` |

关键警告：

1. RC 不属于 TC。
2. RC 不属于 WT。
3. TC 是 NETC + ET。
4. WT 是 NETC + SNFH + ET。
5. 旧任务中把全部异常区域混成 WT 的写法不能直接用于 2026 Task1。

### 3.3 实例生成方式

```text
instance_approximator: ConnectedComponentsInstanceApproximator
```

含义：官方近似用 3D connected components 把每个 region 的 mask 拆成 lesion instances。

G2 检测：

1. 逐 region 计算 connected components。
2. 记录 lesion count、tiny/small/large 分布。
3. synthetic 不能出现大量单体素碎片，否则会推高 small-instance FP。
4. 不能把小病灶简单当噪点删除，因为官方 detection leaderboard 正在奖励小病灶敏感性。

### 3.4 实例匹配方式

```text
instance_matcher: MaxBipartiteMatching
matching_metric: DSC
matching_threshold: 0.000001
```

含义：

1. 预测实例与参考实例用最大二分匹配。
2. 匹配依据是实例 DSC。
3. 基础匹配阈值几乎为大于 0 即可进入匹配候选。
4. 后续 parser 再用 overlap threshold 判断检测 TP/FN。

G2 检测：

1. 不能只看全局 Dice。
2. 必须检查每个 synthetic lesion 是否形态连贯、边界合理。
3. 小 lesion 的一个假阳性连通域会直接影响 detection FP/F1。

### 3.5 官方计算指标

config 中全局和实例指标包含：

```text
global_metrics: DSC, NSD, HD95
instance_metrics: DSC, NSD, HD95
```

Leaderboard 当前主展示字段只取：

1. lesionwise DSC mean。
2. lesionwise NSD mean。
3. small-instance TP/FN/FP/F1。

HD95 仍可作为内部辅助诊断，不写成当前官方主字段。

## 4. 官方 parser 逻辑逐项解释

官方 `parse_mets_results` 的核心逻辑如下。

### 4.1 大病灶 lesionwise segmentation

对每个 subject、每个 region：

1. 读取 `reference_instances`。
2. 若 `instance_volume >= vol_threshold`，该实例是 large lesion。
3. matched large lesion：
   - 把 `sq_dsc` 写入 large lesion DSC 列表。
   - 把 `sq_nsd` 写入 large lesion NSD 列表。
   - 把 `sq_hd95` 写入 large lesion HD95 列表。
   - 若 `sq_dsc >= overlap_threshold`，large detection TP 加 1。
   - 否则 large detection FN 加 1。
4. unmatched large lesion：
   - large detection FN 加 1。
   - DSC 记 0。
   - NSD 记 0。
   - HD95 记 373。
5. 对 region 的 FP：
   - 每个 FP 给 large lesion DSC 追加 0。
   - NSD 追加 0。
   - HD95 追加 373。
6. 如果该 subject/region 没有 large lesion，`lesionwise_*_mean_region` 是 NaN。

G2 检测含义：

1. synthetic 不能诱导 S1/S2 产生大量 FP，因为 FP 会进入 lesionwise DSC/NSD 的惩罚列表。
2. 大病灶边界质量影响 `lesionwise_dsc_mean_*` 和 `lesionwise_nsd_mean_*`。
3. 没有 large lesion 的病例不应被强行解释为 Dice=0。

### 4.2 小病灶 detection

对每个 subject、每个 region：

1. 若 `instance_volume < vol_threshold`，该实例是 small lesion。
2. matched small lesion：
   - 若 `sq_dsc >= overlap_threshold`，small TP 加 1。
   - 否则 small FN 加 1。
3. unmatched small lesion：
   - small FN 加 1。
4. small FP 直接取 region 级 `metric_average.fp`。
5. small F1：

```text
small_instance_f1 = 2TP / (2TP + FP + FN)
```

6. 如果该 subject/region 没有 small lesion，small-instance 字段是 NaN。

G2 检测含义：

1. tiny/small synthetic 不能只追求数量，必须控制 FP。
2. ET/TC/WT/RC 每个 region 都要分别看 small-instance。
3. RC synthetic 尤其要谨慎，因为 RC 错误会单独出现在 `small_instance_f1_rc`。

### 4.3 缺失与异常

Synapse Task1 页面说明：若算法未能为某个测试病例产生某项 metric，不会把该 metric 设为最差值。  
官方公开 parser 中 `_handle_missing_data` 对 JSON 的 `missings` 字段又有补零/HD95=373 的处理。

G2 报告策略：

1. 内部评估不允许通过“缺失预测”规避坏结果。
2. 对 S1/S2 预测输出，G2 必须检查每个 validation case 都有预测文件。
3. 如果官方页面和公开 parser 在 missing 行为上存在解释差异，G2 报告要如实写“最终以 Synapse scorer 为准”。

## 5. 当前 leaderboard 字段完整清单

用户给出的字段与当前官方 Task1 Results leaderboard 字段一致。G2 必须保留以下字段顺序。

```text
Submission ID
Date
Participant/Team
Lesionwise_dsc_mean_et
Lesionwise_nsd_mean_et
Lesionwise_dsc_mean_rc
Lesionwise_nsd_mean_rc
Lesionwise_dsc_mean_tc
Lesionwise_nsd_mean_tc
Lesionwise_dsc_mean_wt
Lesionwise_nsd_mean_wt
Small_instance_tp_et
Small_instance_fn_et
Small_instance_fp_et
Small_instance_f1_et
Small_instance_tp_tc
Small_instance_fn_tc
Small_instance_fp_tc
Small_instance_f1_tc
Small_instance_tp_wt
Small_instance_fn_wt
Small_instance_fp_wt
Small_instance_f1_wt
Small_instance_tp_rc
Small_instance_fn_rc
Small_instance_fp_rc
Small_instance_f1_rc
```

G2 文件模板：

```text
work_space/G2/results/qc/official_leaderboard_metrics_template.csv
```

G2 parser/validator：

```bash
python work_space/G2/code/g2_official_mets_metrics_parser.py validate-csv \
  --csv-path work_space/G2/results/qc/official_leaderboard_metrics_template.csv
```

## 6. 官方排名策略

Synapse Task1 Evaluation 页面说明：

1. 排名遵循 DELPHI-based image analysis validation recommendations。
2. 采用 algorithmic ranking。
3. 采用 statistical significance testing。
4. 多维指标先按各指标平均值计算团队 rank。
5. 每个团队的 rank 求和，形成单变量总体 summary measure。
6. 团队按总体 summary measure 排序。
7. 平均排名做成对随机置换，共 500,000 次。
8. 输出成对 p-values。
9. p-values 以上三角矩阵报告。
10. 统计不显著的团队会被分层归组，显著优越的团队会被标注。

G2/S1/S2 含义：

1. 不能只盯一个指标。
2. small-instance F1 提升但 lesionwise DSC/NSD 大幅下降，不是好方案。
3. 单次训练波动需要重复 seed 或 bootstrap 解释。
4. G2 的 synthetic 数据价值要用多指标整体收益判断。

## 7. G2 现有 QC 是否符合官方策略

结论：G2 的训练前 QC 是必要且专业的，但它不是官方 leaderboard 的替代品。必须叠加训练后官方指标验收。

| G2 检查 | 官方关系 | 必须保留原因 |
|---|---|---|
| 文件完整性 | 官方不直接计分 | 缺模态/缺 seg 会导致训练或预测不可用 |
| NIfTI 可读性 | 官方不直接计分 | scorer 和 nnU-Net 都依赖合法 NIfTI |
| 病例内 shape/spacing/affine 一致 | 官方不直接计分 | 防止图像/标签错位 |
| label 值域 `{0,1,2,3,4}` | 官方输入前提 | 输出非法标签会破坏区域映射 |
| corrected label overlay | 官方数据 QC | 训练标签必须优先用 corrected labels |
| source/validation 泄漏 | 比赛合规 | synthetic 不能来自 validation/test |
| lesion component 体积分档 | 直接相关 | 对应官方 large/small lesion 分支 |
| tiny `<27 mm3` | 直接相关 | 对应 2026 detection leaderboard 重点 |
| ROI 边界/z 连续性 | 间接相关 | 影响训练后 FP/FN 与 DSC/NSD |
| 多模态一致性 | 间接相关 | 影响模型是否学习到真实医学信号 |
| teacher model 辅助 | 间接相关 | 提前发现明显错误 synthetic |
| real-only vs real+synth | 直接相关 | 判断 synthetic 是否真正提升官方字段 |

## 8. G2 升级后的检测脚本与转换脚本

### 8.1 G1 raw intake + QC

```bash
python work_space/G2/code/g2_synthetic_raw_intake_qc.py \
  --synthetic-run-root /path/to/G1/run_YYYYMMDD_HHMM \
  --results-root work_space/G2/results \
  --synthetic-run-id run_YYYYMMDD_HHMM_g1_diffusion_v1
```

输出：

```text
results/manifests/synthetic_generation_manifest_{run_id}.csv
results/manifests/synthetic_candidate_manifest_{run_id}.csv
results/manifests/synthetic_accepted_manifest_{run_id}.csv
results/manifests/synthetic_rejected_manifest_{run_id}.csv
results/manifests/synthetic_normalized_mapping_{run_id}.csv
results/qc/qc_metrics_{run_id}.csv
results/qc/diffusion_quality_metrics_{run_id}.csv
results/qc/qc_case_review_{run_id}.csv
results/qc/qc_batch_summary_{run_id}.json
results/reports/G2_synthetic_data_quality_report_{run_id}.md
```

### 8.2 accepted synthetic 到 nnU-Net raw

默认只生成 manifest，不复制大文件：

```bash
python work_space/G2/code/g2_materialize_nnunet_dataset.py \
  --output-root /path/to/nnUNet_raw \
  --synthetic-accepted-manifest work_space/G2/results/manifests/synthetic_accepted_manifest_{run_id}.csv \
  --dataset-id 261 \
  --dataset-name BraTS2026_MET_RealSynth_G1V1 \
  --channel-order g2_official \
  --mode manifest-only
```

训练机上确认磁盘后可用：

```bash
python work_space/G2/code/g2_materialize_nnunet_dataset.py \
  --output-root /path/to/nnUNet_raw \
  --synthetic-accepted-manifest work_space/G2/results/manifests/synthetic_accepted_manifest_{run_id}.csv \
  --dataset-id 261 \
  --dataset-name BraTS2026_MET_RealSynth_G1V1 \
  --channel-order g2_official \
  --mode symlink
```

通道口径：

1. `g2_official`: `0000=t1n`, `0001=t1c`, `0002=t2w`, `0003=t2f`。
2. `s2_current`: `0000=t1c`, `0001=t1n`, `0002=t2f`, `0003=t2w`，用于兼容 S2 现有转换脚本。

G2 建议：团队最终固定一种通道顺序；如果 S1/S2 沿用现有代码，必须在 dataset.json、训练日志和报告中显式声明。

### 8.3 官方 leaderboard parser/validator

解析 Panoptica JSON：

```bash
python work_space/G2/code/g2_official_mets_metrics_parser.py parse-json \
  --json-path panoptica_evaluation_summary.json \
  --output-csv official_metrics_fold0.csv \
  --vol-threshold 27 \
  --overlap-threshold 0.2
```

验证 CSV 是否有官方字段：

```bash
python work_space/G2/code/g2_official_mets_metrics_parser.py validate-csv \
  --csv-path official_metrics_fold0.csv
```

## 9. G2 如何判断 synthetic 数据“好”

### 9.1 训练前只能判断 QC 质量

可写“QC 合格/优秀”的最低标准：

1. `validation_leakage_count = 0`
2. `label_values_valid_rate = 100%`
3. `nifti_readable_rate = 100%`
4. `geometry_consistent_rate = 100%`
5. `hard_reject_rate <= 5%`
6. `manual_review_unresolved_count = 0`
7. `nnunet_integrity_pass = true`

### 9.2 训练后才能判断训练价值

必须比较：

1. `real_only_fold0`
2. `real_plus_synth_{run_id}_fold0`

通过标准：

1. ET/RC/TC/WT 的 lesionwise DSC 不系统性下降。
2. ET/RC/TC/WT 的 lesionwise NSD 不系统性下降。
3. ET/TC/WT/RC 的 small-instance F1 不下降。
4. small-instance FN 不增加，尤其 ET/TC/WT。
5. small-instance FP 不能显著增加；若 FP 增加超过 10% 且 TP 没增加，该批 synthetic 不进入主训练。
6. RC 指标不被 synthetic 拉坏。
7. 若只改善小病灶但损害大病灶分割，只能写成定向消融，不作为主方案。

## 10. 与 G1/S1/S2 的对接结论

### 10.1 对 G1

G1 可以继续输出 GliGAN-compatible raw cases，但必须至少给：

1. run folder。
2. generation_config 或等效说明。
3. checkpoint。
4. seed。
5. source CSV 版本。
6. label channel 数。
7. rc_policy。
8. raw case folder。
9. 四模态 + seg。

G2 负责：

1. legacy suffix 转换。
2. manifest 补建。
3. 自动 QC。
4. accepted/rejected 输出。
5. 给 G1 回传 reject reason。

### 10.2 对 S1

S1 当前自定义 dataset 读取病例目录，并把四模态顺序写成：

```text
t1c, t1n, t2f, t2w
```

它还把 `seg` 拆成：

1. `tumor_label.nii.gz`：原 label 中去掉 RC。
2. `rc_label.nii.gz`：RC 二分类。

G2 对 S1 的交付：

1. 提供原始 5-label seg。
2. 提供可选拆分脚本说明，S1 自己保持 tumor/RC 双头。
3. 明确 synthetic 也必须能拆出 tumor/RC。
4. 如果用 G2 nnU-Net 物化脚本，S1 要选择 `s2_current` 或自行确认 loader 顺序。

### 10.3 对 S2

S2 当前 nnU-Net 转换脚本使用：

```text
0000=t1c
0001=t1n
0002=t2f
0003=t2w
```

G2 默认官方顺序是：

```text
0000=t1n
0001=t1c
0002=t2w
0003=t2f
```

团队必须在训练前固定通道顺序。G2 已在 `g2_materialize_nnunet_dataset.py` 中同时支持：

1. `g2_official`
2. `s2_current`

不能在 real-only 与 real+synth 之间改变通道顺序。

## 11. 一句话结论

官方检测策略的核心不是单纯全局 Dice，而是 ET/RC/TC/WT 四区域上的 large-lesion lesionwise DSC/NSD 与 small-instance TP/FN/FP/F1。G2 升级后的 QC 要先保证 G1 synthetic raw output 可追溯、无泄漏、几何合法、标签合法、医学上合理，再通过 S1/S2 的 fixed real validation fold 用官方同款字段证明它确实提升或至少不损害比赛指标。
