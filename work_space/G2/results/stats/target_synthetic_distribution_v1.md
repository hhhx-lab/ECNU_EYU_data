# G2 Target Synthetic Distribution v1

生成日期：2026-06-14
口径更新：2026-07-12

## 真实分布参考

1. master 病例数：1295，其中 authentic T2W 1030、V3 completion 目标 265。
2. V2 allowed source：仅 patient-group master train 中的 823 个 authentic-T2W 病例。
3. 历史全量标签审计中含 RC 病例数：167。
4. 历史全量 tiny/small/large lesion 数：3788/3922/2083。

## 第一轮生成目标

1. G1 先交付 10-20 个 smoke cases，G2 完成 QC 和 nnU-Net 转换验证。
2. smoke 通过后，再生成 100-300 个候选 synthetic cases。
3. 第一轮 accepted V2 augmentation 不超过 823 个 real-only train 病例的 25%。
4. 每个 source case 默认最多生成 1 个 synthetic case；多发病例专项实验可单独申请例外。
5. source 只来自 `g1_v2_source_manifest.csv` 中 `allowed_as_v2_source=True` 的病例，绝不来自 val/test。
6. 优先补 small/tiny lesion 和多发病例，但 tiny lesion 比例不应超过 accepted synthetic 的 35%。
7. RC 只基于真实 RC case 做保守变体，第一轮不做凭空生成 RC。
8. 第一轮不做整例 MRI 从零生成，不做无 manifest/log 的 raw output。

## 对 G1 当前方案的约束

1. V2 run 必须记录四模态 checkpoint、sampling、crop size、seed 和 source manifest。
2. V2 raw output 必须先经过 G2 composer，不直接交给下游。
3. V3 completion 与 V2 augmentation 分开 run、分开 manifest、分开消融。
4. V3 只修复 fake/broken T2W，不计入新增 synthetic 数量。
