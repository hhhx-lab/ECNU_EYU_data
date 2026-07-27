# Diffusion 150k smoke20 人工技术复核报告

## 1. 结论

- 复核范围：固定分层 smoke 的 20/20 个 lesion-positive 病例、20/20 张四模态三平面 montage。
- `pass_technical_visual`：16 例。
- `pass_with_documented_risk`：4 例。
- `needs_regeneration`：0 例。
- `reject`：0 例。
- 技术硬门：`pass`，20 例、80 个模态输出、480 条区域指标、20 张 montage，0 个 hard failure。
- smoke 决策：`pass_for_full_evaluation_with_documented_risk`。

本结论只允许把 150000-step 四模态 checkpoint 扩展到固定 validation 的完整
`94 lesion-positive + 9 lesion-negative strict no-op` 评估。它不是 G2 最终
`decision=approve`，也不允许据此直接启动 S2 D 或官方 179 例推理。

## 2. 复核方法

人工复核按 `review_index.csv` 的固定顺序执行：先检查最低 tumor SSIM、RC、
tiny/small 和大病灶 tiled 病例，再检查多病灶及 5 例 routine 病例。每例检查：

1. t1c、t1n、t2w、t2f 的轴位、冠状位和矢状位参考/生成配对图。
2. 生成结果是否为空、常数、裁切、越过 support、出现块状空洞、明显重影或 tile 接缝。
3. 四模态在同一 support 内的位置一致性和病灶邻域对比关系。
4. 自动 QC 的 shape、spacing、affine、finite、checkpoint step/hash、support 外信号和病例计数结果。

这里的 Diffusion 输出是条件随机合成，不是确定性图像重建。paired SSIM/MAE
用于定位退化风险和 checkpoint 比较，但不能要求生成影像逐像素复制参考影像；
同样，视觉通过仅代表技术可用，不代表临床真实性或下游分割收益。

## 3. 条件通过病例

| 病例 | 分层 | 最低 tumor SSIM | 人工结论 | 风险 |
|---|---|---:|---|---|
| `BraTS-MET-01134-003` | RC + small | 0.0062 | `pass_with_documented_risk` | tiny/低分辨率 ROI 中生成与参考对比度差异很大 |
| `BraTS-MET-01250-001` | RC + tiny | 0.0385 | `pass_with_documented_risk` | tiny RC 的局部纹理和参考差异明显 |
| `BraTS-MET-01191-003` | RC + large/tiled | 0.6168 | `pass_with_documented_risk` | 大 support 内生成纹理偏移明显，但无 tile 接缝或硬失败 |
| `BraTS-MET-01268-002` | RC + large/tiled | 0.6747 | `pass_with_documented_risk` | 大 support 内生成纹理偏移明显，但无 tile 接缝或硬失败 |

这 4 例没有触发技术硬门，因此不要求立即回退到 145k/140k；但必须在完整
94 例评估中作为 mandatory review strata 继续复核。若同类偏移在完整批次中
系统性出现，或 150k 的分布指标明显退化，再用相同病例、seed、sampling method
和 sampling steps 对相应模态比较 145k、140k。

## 4. 其余病例

其余 16 例均判为 `pass_technical_visual`。5 例 routine 单大病灶病例
`00699-000`、`00532-000`、`01180-000`、`00791-000`、`01058-001`
均有完整四模态输出，未见空白、support 泄漏、裁切、块状空洞、重影或明显
tile 接缝。

多病灶病例中，`00214-000`、`01105-001`、`01105-002`、`01172-004`、
`01119-001`、`01284-002`、`01351-002` 和 `01270-002` 的单一 focus slice
不能同时穿过全部离散连通域，部分轴位/冠状位面板因此近似空白或只显示一个
小区域。矢状面、其他 focus 面和自动统计证明 support 与生成体积存在；本轮把
它记录为 montage 选择局限，不判为生成体积空白。

## 5. 完整评估的强制条件

1. 只生成固定 94 个 lesion-positive validation 病例，不得混入 locked test、completion-only 或官方 179 例。
2. 9 个 lesion-negative 病例单列验证严格 no-op：`was_modified=False`，image 和 seg 逐元素不变。
3. 完整评估继续使用 150k、`zscore`、`edm_heun`、18 steps 和冻结 seed；不能在中途改变采样配置。
4. 对上述 4 个风险病例、全部 RC/tiny/large-tiled 告警和每个分数层病例完成人工复核。
5. 完整 94+9 gate 通过后才生成并冻结 `checkpoint_selection.json` 与最终 `G2_DIFFUSION_QC_GATE`。
6. 只有排除脚本、轴、metadata 和采样配置错误且 150k 明确退化时，才比较 145k/140k。

## 6. 审计产物

- 自动技术结论：`summary.json`、`QC_REPORT.md`、`hard_failures.txt`。
- 自动逐例排序：`review_index.csv`。
- 人工逐例结论：`manual_review_decisions.csv`。
- 指标：`modality_metrics.csv`、`region_metrics.csv`。
- 可视化：`montages/`，20 张。
- G1 生成侧证据：对应归档中的 `metrics.json`、`generation_manifest.csv`、冻结 selection/inventory 和运行日志。

人工复核不改写自动生成的 `summary.json` 和 `QC_REPORT.md`；二者继续保留
`quality_gate=hold_for_manual_review` 的原始机器结论，本文件作为后续人工审计层。
