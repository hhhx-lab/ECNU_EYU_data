# S2 MET-AUG K1/K2/K4/K5 RC-aware 执行方案

时间戳：2026-09-21（Asia/Shanghai）  
实验身份：`s2_met_aug_k1245_20260921_r1`  
状态：`experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`

## 目标

补齐旧 Route A 中被排除的 label4（RC）供体链路，在不改写任何旧实验的前提下，按原提案运行 K1、K2、K4、K5。所有路线仍从比赛基座原E checkpoint 初始化，固定训练集1035、内部验证103、单A100、125 epochs、每25 epochs留点，并按固定103的 `ema_fg_dice` 最高且并列取最早选择 checkpoint。

## 必须先过的前置门

1. 从冻结训练集重新提取两个独立组件池：RC-aware `anchor={NETC,ET,RC}`、floor=27、bbox≤56 mm；K4小灶池维持不含RC、`anchor={NETC,ET}`、floor=8、bbox≤56 mm。
2. 80 mm RC 不进入本轮。冻结 G1 接口只接受64³，组件必须给四周各留4 voxel上下文，因此最大 bbox 继续为56 mm。
3. G1 label条件开放合法label4通道；四模态输出必须 finite、支持区一致、support外逐体素不变。
4. RC沿用FixV3的全支持 raw/content QC，并增加RC边界及跨模态可见性 guardrail；RC没有既有类别特异校准，因此必须明确标记 `not_formally_calibrated`，不得把工程通过写成正式校准。
5. RC固定24例、K4三个小灶档各8例共24例都须 automatic pass，并完成人工24/24接受；未通过不得启动对应训练。
6. F*权重只由train-only统计冻结，需随病灶负荷单调递增且相邻95% bootstrap CI不跨越；否则跳过K1/K5，不得事后改方向。

## 路线

| 路线 | 冻结组成 | 训练启动条件 |
|---|---|---|
| K1 | E-SNFH* + F* + X1，同时按原详述启用B①+B② | RC smoke通过，F统计门通过 |
| K2 | E-SNFH* + D* + X1 + X2，同时启用B①+B② | RC smoke通过，D统计合同完整 |
| K4 | X3 + E-SNFH*，剂量权重恒等1 | 小灶24/24 smoke通过 |
| K5 | K1 + D* | 仅K1与K2都通过预注册科学门后启动 |

共同约束：X1目标RC占比0.023；RC供体B①固定scale=1；B②目标绝对体积按同表型train-only经验分布抽样，尺度限制`1/1.5..1.5`、最多5次、类别占比相对变化≤25%，失败原样回退；E*仍以最终候选10%尝试，壳系数由train-only反解并受SNFH+ET p10-p90门控；D*使用患者内名次体积分布和经验最近邻距离，第二灶概率0.20、最多2灶；X2从epoch83线性衰减至epoch125的0；X3要求anchor至少8 voxel且保持单连通。

## 作业图与失败纪律

`P0 静态预检 -> P1 CPU供体池/统计 -> P2a RC smoke + P2b X3 smoke -> 人工复核 -> K1/K2/K4并行训练 -> 选点与fixed103官方MET -> K1/K2科学门 -> K5训练与评估`。

每个作业仅提交一次，使用排他reservation和原子launch audit。任一新作业非0退出、FAILED标记、关键SHA/合同漂移或结果消失且无成功证据时立即停止对应分支，不原位续跑、不覆盖、不自动建下一attempt。旧FixV3、Screen125及历次失败证据永久保留。

## 比较与解释

K路线在同一固定103上选择checkpoint并报告官方MET，只能用于探索性开发比较。报告必须给原始指标、同病例差值、wins/ties/losses、20000 bootstrap（seed=20260831）、95%CI、未校正p值和FP/FN保护项；不得仅列差值。K1/K2/K4不是单变量消融，成功只能说明组合候选值得独立验证，不能直接归因给其中某一个机制。
