# s2_met_aug_k1245_20260921_r1

隔离的 K1/K2/K4/K5 RC-aware 探索实验。此目录不包含也不改写旧 FixV3/Screen125 运行时；远端产物采用相同实验ID并保存独立的计划、SHA、launch audit、日志和失败证据。

执行顺序严格遵循机器可读计划：先构建两个train-only供体池与经验统计，再做RC/X3固定smoke和人工复核，之后才允许K1/K2/K4训练；K5仅在K1/K2都通过预注册科学门后运行。
