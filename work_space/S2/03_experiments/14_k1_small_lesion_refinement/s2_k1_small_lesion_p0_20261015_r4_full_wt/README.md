---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# K1 P0 全WT来源追踪独立r4

用户2026-10-15明确授权独立修复、全WT邻接/真实合并测试及概率容量处理。本批承接r2成功结果，不复用r3失败运行包/作业4087034，不覆盖旧预测/标签/失败/审计，不授权长训或K5。

当前入口：[2026-10-15状态与提交验收](STATUS_20261015.md)。22静态文件/全WT来源生产追踪和真实nnU-Net大小灶合并边界已通过预检；CPU4100381 held审计pass后释放，未额外查询运行终态，完整1035/692/286/28及四路身份结果待验。

用户最新指示**概率只存服务器，不在本地保存**，已撤销先前本地/清理方案。完整float32/103病例/175与150及RC保护不降级；服务器容量门未过、未提交概率GPU、未拉回概率。operator/probability_draft_not_submitted只是被最新指示否决的历史草案，不可部署。

同fixed103用于选点和报告仅exploratory，技术验收不构成科学优越性。
