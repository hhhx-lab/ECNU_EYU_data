# K1245 F gate诊断与恢复边界

更新时间：2026-09-23（Asia/Shanghai）  
适用实验：`s2_met_aug_k1245_20260921_r1`及其修复后独立批次  
状态：`experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`

## 1. 原 gate 的真实结果

原计划的 F gate 同时要求：

1. 各病灶负荷层的 small-lesion fraction 单调递增；
2. 每一对相邻负荷层的 95% bootstrap CI 完全不重叠。

train-only 1035 例的冻结结果为：

| 负荷层 | 均值 | 95% CI | 例数 |
|---|---:|---|---:|
| 1 | 0.06415 | 0.03774–0.09434 | 265 |
| 2–3 | 0.19305 | 0.16216–0.22523 | 259 |
| 4–8 | 0.26090 | 0.23103–0.29162 | 269 |
| 9–16 | 0.32889 | 0.29095–0.36844 | 148 |
| 17–24 | 0.38600 | 0.32461–0.44875 | 41 |
| >24 | 0.49219 | 0.43867–0.54721 | 53 |

结论：

- 单调递增：通过；
- CI 完全不重叠：不通过。失败不是 RC donor 数量不足，也不是概率均值不为 0.2；最终 F gate 必须仍记为 `false`。

原始结果证据：
`work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260921_r1/operator/snapshot_20260922T154321Z/LOCAL_OPERATION_RECEIPT.json`。

## 2. 不能做的修复

- 不能修改 r1 的 `f_gate_pass`、原计划、原 pool validation 或旧快照；
- 不能删除或重写 r1 的失败证据；
- 不能把“CI有少量重叠”描述成原预注册 gate 通过；
- 不能直接把 K1/K5 训练混入 r2 的严格门控路线。

## 3. 可行的独立恢复路线

若需要继续研究 F 方向，应建立新的、独立命名的 recovery 计划，先冻结替代统计门，再提交任何训练。建议把门拆成两层：

- `F_STRICT_PREREG_GATE=false`：保留原计划结果，永不改写；
- `F_MONOTONIC_ENGINEERING_GATE=pass`：仅记录单调方向和总体概率校准结果，不等价于科学 gate 通过。

替代统计门可在新计划中预先指定为相邻层差值的 bootstrap CI、预先指定的单调趋势检验，或仅要求单调性加概率均值/上限约束。不能从上述选项中事后挑一个能通过的结果；新计划必须在看到 recovery 结果前固定方法、seed、分母和阈值。

在 recovery 计划通过静态预检、且 RC smoke 与人工复核通过后，K1/K5最多只能作为 `exploratory_monotonic_fallback` 运行；报告必须同时列出 strict gate=false，不得称为原 K1/K5 预注册科学门通过。K5仍需 K2 科学门通过。

## 4. 当前动作

本说明只处理统计阻塞的定义和证据边界，不提交新作业。r2供体池仍按自身计划运行；K2/K4不依赖 F gate，但仍须等待供体池完成、RC/small8 smoke及人工复核。待用户明确采用独立 fallback 后，下一步才是生成新计划、静态预检并按一次性提交规则启动新的 K1/F recovery 分支。
