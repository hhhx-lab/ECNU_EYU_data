# K1/K2/K4/K5 RC-aware 提交留存记录

时间：2026-09-21（Asia/Shanghai）  
实验：`s2_met_aug_k1245_20260921_r1`  
状态：`experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`

## 冻结身份

- 机器可读计划：`work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260921_r1/K1245_RC_AWARE_EXECUTION_PLAN_20260921_R1.json`
- 计划 SHA256：`5e337319fe63e5c261162d5850c0c4310c1cc729c984a74c0373ea01b4975333`
- 远端根：`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1245_20260921_r1`
- 父 FixV3 根：`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_fix_v3_full200_20260810_r1`

## 2026-09-21 首次提交

- 远端静态预检：`pass`；7项K1245合同测试通过，固定划分1035/103正确，原始/预处理数据目录存在，`/public`剩余约29 TB。
- CPU供体池/经验统计作业：`3708834`，name=`s2kpoolr1`，通过held提交、launch audit后释放。
- 首次快照：`RUNNING`，node=`node212`，已出现`POOL_BUILD_STARTED.json`，无`POOL_BUILD_FAILED.json`。
- 本次仅启动P1：RC27与small8供体池及train-only统计。未提交smoke、K1/K2/K4/K5训练或fixed103评估。

## 后续门控

P1必须先产生可解析且符合冻结合同的RC27与small8清单，RC donor数不少于24，F统计门按预注册规则判定。之后先运24+24 smoke和人工图片复核；K1/K2/K4只能在各自前置门通过后启动，K5还必须等K1、K2科学门同时通过。

旧 FixV3、Screen125、Extend175及历次失败证据均不覆盖。固定103同时用于checkpoint选择与报告，最终结果只能作为exploratory。
