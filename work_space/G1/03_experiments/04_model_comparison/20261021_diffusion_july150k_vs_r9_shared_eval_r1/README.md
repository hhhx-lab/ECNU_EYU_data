# 七月150k与R9冻结所选模型：共同评价R1

用户2026-10-21授权在统一评分体系评价新旧模型。本批独立生成评价，不改训练或既有选模。没有公认的医学生成单一权威总分；本批使用公开成熟SSIM实现、明确前景分母、配对统计和误差/风险分列，结果仍属开发集探索性比较。

详见[PROTOCOL.md](PROTOCOL.md)。旧四150k与R9所选120k/40k/200k/200k，共八模型。真实原图为主参考，VAE epoch004自重建为次参考；共同val103/94阳性/9阴性不评分，test禁读。旧权重裁块zscore用真实参考内容的固定均值/样本标准差逆变换到共同全病例zscore，附未经换算敏感性结果。不是对生成结果拟合校准。

新模型已按用户澄清锁定为“R9自重建主 SSIM”的四份最佳权重，不是原图R2。选模身份见RUN.json与tests/test_operator.py；本批不重新选点。

冻结时状态：本地13测试通过，包括八权重strict CPU加载/64cube前向、背景抬分反例、共同噪声/覆盖/归一化/病灶亚区与提交依赖/SHA拒绝。真实A100 smoke尚未运行；尚无新比较分数。提交后实际job/阶段见本地非冻结STATUS_20261021.md及operator提交回执，不以本冻结README推定运行状态。CPU审计、四A100配对smoke/全量、CPU汇总组成十作业DAG；全量入口还须全部smoke完成0:0、身份与估时门通过。所有失败保留，无自动重试/恢复监控/训练或S2替换。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
