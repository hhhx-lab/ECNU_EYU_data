---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# 九路 selected103 官方 MET 横评

本目录是对九份已完成官方 MET CSV 的本地只读再比较；没有重新训练、推理、运行评估器或提交作业。B、E-SNFH、F 均取独立 Extend175 的 selected epoch175，C40 和 D 取 Screen125 的 selected epoch50，K1/K2/K4 取 selected epoch175，原 E 是比赛选择的原始 epoch200 基座，不是 E-SNFH。

九份 CSV 均为同一 fixed103 真实 T2W 开发病例，103 病例顺序和 72 指标列顺序一致；各文件 SHA 与原验收一致。比较遵循冻结比较器 SHA `9ff245240c438beb998652ee59f130fc8c012be29ac3b689c862d04135df6e12` 的指标方向和 complete-case 规则。每对每指标单独取双方 finite 病例，20,000 次 bootstrap，seed `20260831`。原 K1/K2/K4 216 条及 B/E-SNFH/F175 216 条历史记录的配对 n、方向化差值和胜平负已逐条交叉核对。新 bootstrap 的抽样序列跨九路继续前进，CI/p 与旧三路独立计算可能有蒙特卡洛波动。

输出：

- [阅读报告](REPORT_20261008.md)：九路原值和关键同病例对照。
- [完整648个原始均值](results/OFFICIAL_MEANS_648.csv)：每方法72指标、finite n。
- [完整2592条配对统计](results/PAIRED_ALL_2592.csv)：36对×72指标，含双方原始均值、共同病例均值、CI、W/T/L和未校正p。
- [八路相对原 E 的576条记录](results/PAIRED_VS_ORIGINAL_E_576.csv)。
- [身份与验收清单](results/VALIDATION.json)：9份输入路径、CSV SHA、checkpoint SHA及选择epoch。
- [可复算脚本](compare_nine_met.py)：输出目录排他创建，不修改任何输入。

不同训练预算与多因素组合在同一表中仅作探索性观察；fixed103 同时用于选点和报告，不是未触碰测试集或正式优越性证据。原 Screen125 五路125预算横评及旧失败记录均不被此表覆盖。
