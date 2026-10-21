---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# K1小病灶nnU-Net改进方案入口

- [2026-10-15全WT修复独立r4](../../../03_experiments/14_k1_small_lesion_refinement/s2_k1_small_lesion_p0_20261015_r4_full_wt/STATUS_20261015.md)：用户授权后补全所有来源邻接再筛286/28；真实nnU-Net构造大小源74.088/1.728mm³合并degree2及空/RC/物理/crop边界通过，22冻结SHA/20测试通过。CPU4100381 held资源审计pass后释放，完整1035实例/692WT/四路身份尚未验收，无第二进度查询。用户最终要求概率只存服务器，撤销本地及清理方案；未部署概率草案/提交GPU/拉概率，服务器容量仍不足，P/S/L hold。旧r3及失败保留。

- [2026-10-15后续诊断预注册](revisions/20261015/P0_FOLLOWUP_DIAGNOSTICS_PREREG_20261015.md)与[r3真实状态](../../../03_experiments/14_k1_small_lesion_refinement/s2_k1_small_lesion_p0_20261015_r3_diagnostics/STATUS_20261015.md)：1035完整标签路径/SHA、B175/D50各103预测和175/150 checkpoint已核，非完整实例/五路身份验收。r3只追踪小灶遗漏大小灶合并，另有held合同mismatch；4087034取消请求返回0、终态未再核，冻结包不复用。[新独立修复要求](revisions/20261015/R3_STOP_AND_CORRECTION_REQUIREMENTS_20261015.md)保留全WT邻接和真实边界要求。概率两点至少145.29GiB，扣G1预留后不足；原E旧掩码/真实曝光梯度仍缺，P/S/L暂不选，无新长训或K5。

- [2026-10-14空区域CPU修复独立r2](revisions/20261014/P0_CPU_EMPTY_REGION_RECOVERY_R2.md)：执行版下的恢复合同；[2026-10-15完成验收](../../../03_experiments/14_k1_small_lesion_refinement/s2_k1_small_lesion_p0_20261014_r2/STATUS_20261015.md)确认CPU4068477 COMPLETED 0:0、103例身份/往返、35轻量文件逐SHA和独立复核pass。两路共同漏检233；K4新检出7/失检1；往返258/286不是模型上限。保留[历史提交状态](../../../03_experiments/14_k1_small_lesion_refinement/s2_k1_small_lesion_p0_20261014_r2/STATUS_20261014.md)，不修改旧r1；概率/曝光/梯度/G1谱系仍待诊断，整个P0未完成，不授权训练或K5。

- [2026-10-13修正后的执行版](revisions/20261013/PLAN_EXECUTION_20261013.md)：当前入口；用户已授权分阶段执行，先P0，不跳过诊断长训，不授权K5。
- [2026-10-13改进版审阅](revisions/20261013/REVIEW_20261013.md)：保留审阅依据；执行版已纠正标签往返上限、共同漏检身份及完整标签资格，并收紧proxy/G1来源解释。
- [用户2026-10-13修订稿原样留存](revisions/20261013/PLAN_USER_REVISION_20261013.md)：源SHA与本地留存相同，保留第7节外部意见；其Windows临时/旧相对链接未统一修复，项目证据通过审阅文档定位。
- [2026-10-09原讨论稿](PLAN_20261009.md)：历史基线，未覆盖。

P0已经开始：本地五路103病例/11,270条参考组件与官方CSV核对通过，WT小灶46例/286灶一致；没有稳定ID时明确不计算共同漏检、新/失检。远端输入/软件只读核验与后续阶段见[独立P0入口](../../../03_experiments/14_k1_small_lesion_refinement/s2_k1_small_lesion_p0_20261013_r1/README.md)。训练候选及完整CPU/GPU合同须在对应诊断/预检后冻结，旧K1和K5门不变。
