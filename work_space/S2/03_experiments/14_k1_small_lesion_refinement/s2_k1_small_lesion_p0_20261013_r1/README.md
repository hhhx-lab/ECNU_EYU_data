# K1小病灶P0独立诊断

2026-10-13用户授权修正方案后分阶段执行。先核对既有官方明细与远端冻结输入，再执行标签/身份/概率诊断；不直接启动175轮训练，不授权K5。

执行合同：../../../05_documents/02_plans/20261009_k1_small_lesion_nnunet_refinement/revisions/20261013/PLAN_EXECUTION_20261013.md。

`runtime`为独立诊断代码，`tests`为边界测试，`operator`为一次性访问工具与独立快照，`results`为本地审计产物。旧K1/九路数据只读引用，所有输出保持experimental_unvalidated、operator_approved=false、formal_gate_status=not_run_not_passed。

当前入口：[P0 CPU停止与资产验收](P0_CPU_STOP_AND_INVENTORY_20261013.md)。515资产SHA验收通过；job4051333唯一释放后因本方空参考区域适配器缺口发出定向取消请求，未验收终态/诊断结果，不复用r1运行包或自动下一attempt。官方小灶阈值为原始完整组件物理体积<27 mm³。
