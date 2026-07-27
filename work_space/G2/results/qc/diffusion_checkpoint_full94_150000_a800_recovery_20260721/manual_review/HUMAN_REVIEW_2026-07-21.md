# BraTS 2026 G2 Diffusion Full94+9 人工复核报告

## 1. Gate 结论

- 复核对象：四模态 Diffusion `150000` checkpoint，在固定 `94` 个 lesion-positive validation 病例上生成，另有 `9` 个 lesion-negative strict no-op 病例。
- 自动技术门：`pass`；自动质量门保持 `hold_for_manual_review`，由本人工层承接最终判断。
- 机器计数：`94` cases，`376` modality rows，`2256` region rows，`376` artifact rows，`94` montages，hard failures `0`。
- 人工复核：`94/94` 张 montage 全量复核完成。
- `pass_technical_visual`: `11` 例。
- `pass_with_documented_risk`: `83` 例，全部 `risk_accepted=True`。
- `needs_regeneration`: `0` 例；`reject`: `0` 例。
- G2 最终人工判断：`approve_150000_no_rollback_comparison_required`。

本结论只冻结 G2 Diffusion checkpoint selection。它不启动 S2 D，不启动官方 179 例推理，也不改变后续 nnU-Net validation/test/official inference 不调用 Diffusion 的边界。

## 2. 运行契约核验

- 远端 A800 生成与 QC 进程已自然退出，PID 文件保留；按完成产物判定，不按 PID 存活判定。
- 远端日志未检出 `OOM`、`NaN`、`Traceback`、`RuntimeError` 或失败关键字。
- 固定 seed: `20260720`。
- normalization: `zscore`。
- sampler: `edm_heun`，steps `18`，noise schedule `edm`。
- large lesion mode: `tile`。
- generation manifest: `376` rows = `94 x 4` modalities。
- no-op 契约：`9/9` negative cases 保持 image/seg 逐元素不变，前后 SHA256 一致。

## 3. Checkpoint 来源

- `t1c`: step `150000`, bytes `53749405`, SHA256 `cc49de179dee75af561df377ba323052da99525a58a99c60d2fe48f2c34d51a5`
- `t1n`: step `150000`, bytes `53749405`, SHA256 `bc98c9423dad396ee235c89893c308b5e6d340667a8b10880825020b6e976ad6`
- `t2w`: step `150000`, bytes `53749405`, SHA256 `1b42542f378375406e38a17ca380a608fd0005be4591ef2bcedabca925c3ff60`
- `t2f`: step `150000`, bytes `53749405`, SHA256 `de2f219fe126dbb7974d61d8fe8697d239d1e0b18f735344feeab36d3f7d9e6c`

四个 checkpoint 均与 generation metrics、checkpoint inventory 和 QC summary 中的 SHA256 对齐。

## 4. 复核队列

必审并集覆盖全部 94 例，因此执行 full94 全量人工复核。分层计数如下：

- `artifact`: `78`
- `large`: `80`
- `low_score`: `10`
- `rc`: `16`
- `smoke_risk`: `4`
- `tiled`: `73`
- `tiny`: `44`

Contact sheets 位于 `manual_review/contact_sheets/batch_01.png` 到 `batch_08.png`。逐例结论位于 `manual_review/manual_review_decisions.csv`。

## 5. 风险接受分层

- `AUTOMATED_ARTIFACT_FLAG_REVIEWED`: `28`
- `LOW_SCORE_TEXTURE_SHIFT`: `10`
- `MODERATE_TEXTURE_SHIFT`: `29`
- `SMOKE_RISK_RECHECKED`: `4`
- `TINY_OR_MULTIFOCAL_FOCUS_LIMIT`: `44`

这些风险均为技术审计风险，不是硬失败。目视复核重点检查了四模态是否为空、是否越出 green support、是否出现明显 tile 接缝、块状空洞、重影、不可解释的全局亮度崩溃或切面几何错位。未发现需要 regeneration 或 reject 的病例。

## 6. Smoke 风险病例复核

| 病例 | 分层 | min tumour SSIM | 人工结论 | 备注 |
|---|---|---:|---|---|
| `BraTS-MET-01134-003` | `artifact;low_score;rc;smoke_risk` | 0.0062 | `pass_with_documented_risk` | smoke 风险病例在 full94 中复核，仍为技术可用但需保留风险记录；低 paired SSIM，反映条件随机生成与参考纹理/对比差异；自动伪影告警已目视复核，未见空白、越界、重影或 tile 接缝硬失败。风险接受：该风险不构成 150k 明确退化，也不触发 145k/140k 对比。 |
| `BraTS-MET-01191-003` | `artifact;large;rc;smoke_risk;tiled` | 0.6168 | `pass_with_documented_risk` | smoke 风险病例在 full94 中复核，仍为技术可用但需保留风险记录；support 内纹理或对比存在可见偏移，但未见硬失败；自动伪影告警已目视复核，未见空白、越界、重影或 tile 接缝硬失败。风险接受：该风险不构成 150k 明确退化，也不触发 145k/140k 对比。 |
| `BraTS-MET-01250-001` | `low_score;rc;smoke_risk;tiny` | 0.0385 | `pass_with_documented_risk` | smoke 风险病例在 full94 中复核，仍为技术可用但需保留风险记录；低 paired SSIM，反映条件随机生成与参考纹理/对比差异；tiny/多灶病例的固定 focus montage 不能代表全部连通域。风险接受：该风险不构成 150k 明确退化，也不触发 145k/140k 对比。 |
| `BraTS-MET-01268-002` | `artifact;large;rc;smoke_risk;tiled` | 0.6747 | `pass_with_documented_risk` | smoke 风险病例在 full94 中复核，仍为技术可用但需保留风险记录；support 内纹理或对比存在可见偏移，但未见硬失败。风险接受：该风险不构成 150k 明确退化，也不触发 145k/140k 对比。 |


上述 4 例与 smoke 阶段结论一致：存在纹理或低分风险，但没有技术硬失败。

## 7. 最低分 10 例

| 病例 | min tumour SSIM | 分层 | 人工结论 |
|---|---:|---|---|
| `BraTS-MET-01301-001` | -0.0038 | `artifact;low_score` | `pass_with_documented_risk` |
| `BraTS-MET-01134-003` | 0.0062 | `artifact;low_score;rc;smoke_risk` | `pass_with_documented_risk` |
| `BraTS-MET-01310-001` | 0.0145 | `artifact;low_score;tiny` | `pass_with_documented_risk` |
| `BraTS-MET-01250-001` | 0.0385 | `low_score;rc;smoke_risk;tiny` | `pass_with_documented_risk` |
| `BraTS-MET-01134-000` | 0.1021 | `large;low_score` | `pass_with_documented_risk` |
| `BraTS-MET-01310-002` | 0.1989 | `low_score;tiny` | `pass_with_documented_risk` |
| `BraTS-MET-00572-000` | 0.2222 | `artifact;large;low_score` | `pass_with_documented_risk` |
| `BraTS-MET-01109-003` | 0.2323 | `artifact;low_score;tiny` | `pass_with_documented_risk` |
| `BraTS-MET-01117-002` | 0.2696 | `low_score` | `pass_with_documented_risk` |
| `BraTS-MET-01096-001` | 0.2929 | `large;low_score` | `pass_with_documented_risk` |


低分主要来自条件随机生成与参考影像的局部纹理/对比差异，尤其 tiny/RC 或单个 focus slice 的病例。它们不表现为空输出、错位、support 泄漏或 tile seam，因此不构成“150k 明确退化”。

## 8. 不触发 145k/140k 对比的理由

1. 自动技术门 94+9 全部通过，hard failures 为 0。
2. no-op negative 9/9 严格通过，证明没有在无病灶病例中误插入或改写。
3. 生成契约固定为 150k、seed 20260720、zscore、edm_heun、18 steps，manifest 与 checkpoint hash 均一致。
4. 人工全量复核未发现系统性空白、越界、tile seam、几何错位、NaN/Inf 或生成崩溃。
5. 风险病例均被记录为 documented risk，而非 reject/needs_regeneration。

因此按总控计划固定 `150000` checkpoint；不运行 145k/140k 回退对比。

## 9. Gate 后边界

- 允许：生成 `checkpoint_selection.json`、`g2_diffusion_qc_gate.json` 和 `SHA256SUMS.txt`，把 G2 Diffusion gate 标记为正式结束。
- 禁止：在本 gate 自动进入 S2 D、官方 179 推理、G2 以外任何后续阶段。
- 后续 S2 D 只能在新任务中按总控计划单独启动，且 Diffusion 只用于训练 patch 按固定概率增强；validation、test 和官方推理均不得调用 Diffusion。

## 10. 证据 SHA256

- `manual_review_decisions.csv`: `948e49b54157f18c0c177e5dacf31eb38ddba6c18b9bd739a2e5e3fefad73d88`
- `mandatory_review_template.csv`: `f8077d200e9d0d3a884401e52385c3bead053d6895e763f3df6d6195a848a76a`
- `summary.json`: `12dba17bb446988b3ca705554ef7a13f703f28c8ff7e4dd9b2c4182416a40b03`
- `metrics.json`: `bc2e9e348811d84a3f3c57048abc76db4368f8af962439d6b5c51a35dc62f234`
- `generation_manifest.csv`: `051e6863b327d4ee4b7892d35c70242426add0e1098848086441e302ea1eb10c`

生成时间 UTC：`2026-07-21T03:05:06.957794+00:00`。
