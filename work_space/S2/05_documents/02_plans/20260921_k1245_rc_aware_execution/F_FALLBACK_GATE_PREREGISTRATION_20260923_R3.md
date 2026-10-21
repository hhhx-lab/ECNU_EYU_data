# F fallback gate preregistration: K1245 r3

冻结时间：2026-09-23 15:23:07 Asia/Shanghai（2026-09-23 07:23:07 UTC）  
状态：`experimental_unvalidated`；`operator_approved=false`；`formal_gate_status=not_run_not_passed`。

## 原严格门

`F_STRICT_PREREG_GATE=false`。r1要求六档负荷下小病灶比例单调递增，且全部相邻档的95% bootstrap CI不重叠。后一个条件失败，至少三对相邻区间重叠。此结论、原计划和证据永久不变。

## 独立 fallback 定义

本门是**工程资格检查**，不是科学显著性检验，不产生“F科学门通过”或增强有效的结论。它只判断已冻结的训练分布是否能形成方向一致、均值校准且受上限约束的病例抽样概率。

病例负荷按冻结预处理标签中的26连通病灶组件数分为 `1`、`2_3`、`4_8`、`9_16`、`17_24`、`gt24` 六档。单病例 small-lesion fraction 定义为该病例体积严格小于27 mm3的病灶组件数，除以该病例全部病灶组件数；无病灶病例的分母按1处理。按这六档顺序计算病例级比例的算术均值。

fallback 必须同时满足：

1. 唯一冻结训练病例恰为1035例，六个预定负荷档的病例数分别为265、259、269、148、41、53；不得纳入validation病例。
2. 六档 small-lesion fraction 均值逐档非递减，容差为`1e-12`。
3. 六档训练选择概率逐档非递减，容差为`1e-12`；概率均在[0, 0.60]。
4. 以冻结1035例病例逐例赋予所在档概率后，算术均值与0.20相差不超过`0.0001`。
5. 输入r2计划、静态预检、pool完成标记与RC27/small8 manifest身份均与r3机器计划SHA绑定；任何输入漂移均为fail-closed。

概率的唯一生成规则：先取六档small-lesion fraction均值；除以按1035例病例加权后的总体small-lesion fraction；以此比例乘全局标度，再逐档裁剪至0.60。全局标度用固定二分法求得，使按病例加权概率均值为0.20；二分区间[0,10]、迭代200次。二分规则及来源r2冻结池生成器一致。当前快照仅含病例组件数及分层small-fraction汇总，不含1035个逐病例small-fraction值；因此本fallback复核会对冻结汇总做门控、从病例组件数重算病例权重/概率校准，并明确此证据限制。95% CI只作为描述统计，不参与fallback判定；不做p值或显著性筛选。

## 对后续路线的影响

fallback pass 最多允许生成独立K1 exploratory配置。K1训练前仍必须有RC smoke自动24/24通过及24/24人工接受。结果报告始终并列显示`F_STRICT_PREREG_GATE=false`与fallback状态。fallback不改变K1的科学解释，也不允许宣称原strict门通过。

K5不受该fallback直接放行。只有K1和K2分别完成各自预注册训练、checkpoint选择及固定103官方MET科学门并通过后，才可建立K5配置/静态预检/单次提交。K路线是多因素组合，不能据此归因某单一机制。

## 冻结来源

- r1计划：`s2_met_aug_k1245_20260921_r1/K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.json`，SHA256 `5e337319fe63e5c261162d5850c0c4310c1cc729c984a74c0373ea01b4975333`。
- r1 strict失败回执SHA256：`de1f231c46bc61de4c91ecd12983690aac3f6bb9765f4e1ea4d0df83cfe47785`。
- r2计划SHA256 `815494ccba511570afa4c53c23034807ce8f351d06446f25d1c2c4fc4e364018`；完成marker SHA256 `37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1`。
- r2观察回执SHA256 `9f7240cbfb4f63910b1b86608625d94a6d90303ee5f36adf3741a721f55fc021`；RC27与small8 manifest identity分别为 `4bef77f255a1a5fa2ba65fa22b58c84191634859b4040eacf99e13f99059c35c`、`d27ac72359449f157a0afc91f80bc062a03eb4a3c19453f7c842f5498583a2fd`。

机器合同见同目录下r3实验根的`K1245_F_FALLBACK_EXECUTION_PLAN_20260923_R3.json`。这项独立定义不回写r1/r2历史结果。
