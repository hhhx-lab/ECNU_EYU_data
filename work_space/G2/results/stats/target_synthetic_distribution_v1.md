# G2 Target Synthetic Distribution v1

生成日期：2026-05-31

## 真实分布参考

1. 可用真实病例数：1295。
2. 含 RC 病例数：167。
3. tiny/small/large lesion 数：3788/3922/2083。

## 第一轮生成目标

1. G1 先交付 10-20 个 smoke cases，G2 完成 QC 和 nnU-Net 转换验证。
2. smoke 通过后，再生成 100-300 个候选 synthetic cases。
3. 第一轮 accepted synthetic cases 不超过真实训练病例数的 25%。
4. 每个 source case 默认最多生成 1 个 synthetic case；多发病例专项实验可单独申请例外。
5. source case 只来自 final_qc_pass=true 的训练病例，绝不来自 validation。
6. 优先补 small/tiny lesion 和多发病例，但 tiny lesion 比例不应超过 accepted synthetic 的 35%。
7. RC 只基于真实 RC case 做保守变体，第一轮不做凭空生成 RC。
8. 第一轮不做整例 MRI 从零生成，不做无 manifest/log 的 raw output。

## 对 G1 当前方案的约束

1. 60% 概率修改标签、70% 概率将 SNFH/ET 转换等操作必须逐例写入 manifest。
2. 缩放比例、插入肿瘤数量、label_kind、seed 都必须可复现。
3. Regular 与 Custom 应作为两种 generation policy，不能混在一个未标记的数据池里。
4. 在线训练方案需要 S1/S2 训练框架配合；本机 Mac 只准备规则、manifest 和 QC，不运行在线生成训练。
