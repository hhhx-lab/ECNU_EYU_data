# s2_met_aug_k1245_20260923_r2

隔离的 K1/K2/K4/K5 RC-aware 修复后探索实验。r1 的 RC label4 合同失败证据永久保留；r2 不覆盖 r1，也不改写旧 FixV3/Screen125 运行时。

r2 修复了 K1245 子类对合法 label4 的输入验证兼容，并在 smoke 中保存每个候选目标的失败 reason/detail。FixV3 的 RC 边界和跨模态 guardrail仍然保留，RC QC 仍标记为 `not_formally_calibrated`。

执行顺序严格遵循机器可读计划：先构建两个train-only供体池与经验统计，再做RC/X3固定smoke和人工复核，之后才允许K1/K2/K4训练；K5仅在K1/K2都通过预注册科学门后运行。
