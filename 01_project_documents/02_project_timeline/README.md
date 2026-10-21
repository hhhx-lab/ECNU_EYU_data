# BraTS 2026 时间线与总控入口

本目录保存跨 G1、G2、S1-S5 的阶段计划、关键路径、放行门和最终提交记录。模块内部的实现细节仍放在各自 `work_space/<组>/docs`；跨模块执行以这里的总控文档为准。

## 当前总控文档

- [BraTS 2026 Task 1 最终全流程执行与验收总控手册](BraTS2026_Task1_最终全流程执行计划.md)

该手册覆盖：

- 已完成的 G1 missing-T2W、G2 completion QC 和 S2 completion-only 结果。
- G1 Diffusion V3 四模态 checkpoint 收口与选择。
- G2 Diffusion paired QC、三平面人工复核和 gate。
- S2 completion-online 训练及 A/B/D 正式消融。
- 内部 103 例官方兼容评估。
- 官方 179 例推理、空间/标签审计、ZIP 和 Synapse 提交。
- 服务器、存储、环境、依赖、故障恢复、时间预算和最终 checklist。

## 2026-07-20 状态

| 工作项 | 状态 | 当前结论 |
|---|---|---|
| G1 缺失 T2W 阶段 6 | 已完成 | 265 例完整拉回 |
| G2 completion QC | 已完成 | 212 train + 53 evaluation，0 pending/rejected |
| 47 例重点人工复核 | 已完成 | 47/47 技术复核通过 |
| S2 A real-only | 已完成 | checkpoint 与内部 103 例官方兼容结果已归档 |
| S2 B completion-only | 已完成训练 | Epoch 199、103/103 validation、final checkpoint 已归档 |
| G1 Diffusion V3 | 即将完成 | t1c/t2w/t2f 已到 150000，t1n 快照为 146143/150000 |
| G2 Diffusion QC | 未开始 | 等四模态冻结与 103 例生成评估 |
| S2 D completion-online | 未开始 | 等 checkpoint selection 和 G2 gate |
| 官方 179 例 | 数据已核验 | A800 上 179 例四模态齐全、无 seg，尚待最终模型推理 |
| Synapse | 未提交本轮最终产物 | 等 B/D 决策和 179 例 ZIP |

## 更新规则

1. 每次跨阶段放行后更新总控文档的状态快照和 checklist。
2. 训练完成、QC 通过、内部评估完成、提交包完成和 Synapse 评分必须分别记录，不得合并成一个“完成”。
3. 任何病例数、split、checkpoint、哈希或官方输入变化，都要新建 run 记录，不覆盖旧结果。
4. 大型 NIfTI、checkpoint、缓存和 ZIP 不提交 Git；仓库只保存轻量代码、清单、报告和必要审计记录。
5. S1/S3/S4/S5 保持并行研究线，未经同一数据契约和官方兼容评估验证，不进入当前硬截止提交路径。
