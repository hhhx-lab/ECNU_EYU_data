---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# B/E-SNFH/F Epoch175 官方 MET 验证：2026-09-10 r1

本实验只评估已完成并验收的三路 epoch175，不训练，不提交225任务。

## 固定协议

- 冻结 B/E-SNFH/F 的175 checkpoint，按已验收SHA读取，独立复制为只读推理快照；不修改源checkpoint。
- 复用原MET r2的同一103例真实T2W输入和参考标签；预检515个输入/标签文件SHA。
- 三个单卡A100评估并行，无exclusive；采用已成功的逐例单进程推理，预处理/导出worker均为0，无multiprocessing queue。
- 推理网络构造与参数不变，仅允许加载175扩展训练器的确切名称。正式GPU推理前，预检CPU严格加载网络权重。
- 官方MET仍用BraTS-evaluation0.0.8、panoptica2.1.0、numpy1.26.4，mets配置、volume阈值27、overlap阈值0.2。
- 三路各103预测，总309；比较175三路的3个pair，并单独比较各路175对125的3个pair。
- 配对统计沿用20000次bootstrap、seed=20260831；报告每指标实际finite配对n、均值、胜平负、95%CI、未校正p、FP/FN及大小病灶指标。
- 每任务只提交一次，先hold提交并原子写launch audit，再release；CPU比较通过afterok依赖自动启动。
- 失败时保留证据、停止自动推进，不原位重跑、不自动创建新attempt。

## 解释边界

同一103例已参与checkpoint选择，本次为探索性开发验证，不能宣称正式优越性。175对125是额外训练预算的收敛实验，不改写原五路125预算横评。内部EMA、官方分割MET、G1图像质量不得混写。

## 运维

`operator/prepare_bundle.py` 从既有验收材料生成冻结计划和静态SHA清单。`runtime/workflow.py` 依次提供preflight、launch、route、paired动作。旧尝试与旧失败证据只读保留。

远程连接仅从指定env加载凭据，显式启动分流VPN，完成有界操作后关闭本次VPN；不持续SSH、不循环轮询。轻量拉回默认排除checkpoint、NIfTI和events JSONL。
