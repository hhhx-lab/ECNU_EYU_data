# 外部分割模型与本项目 fixed103 横评可行性

复核日期：2026-11-04，Asia/Shanghai；公开请求 UTC 横跨 2026-11-03/04 本地午夜，具体时刻见审计。

本次只核查历史清单、成绩、公开代码和轻量模型/镜像元数据，不连接 ECNU，不下载权重或镜像，不安装环境、不执行第三方程序、不提交作业。

- [详细报告](REPORT.md)：现有最佳候选、固定验证集、2026/2025 模型可用性、训练暴露与同口径比较流程。
- [本地历史核验](LOCAL_HISTORY_AUDIT.json)：103病例/69患者组、原始来源、九路成绩及输入 CSV/SHA 核对。
- [公开来源审计](PUBLIC_SOURCE_AUDIT.json)：源码固定提交、文件 blob、Hugging Face 文件大小/SHA、Docker Hub digest、访问失败和查询时间。
- [核验记录](VERIFICATION.json)：本次轻量证据一致性检查；不代表模型已加载或推理成功。

原始公开源码/网页/API响应仅本机留存于 `FULL_PUBLIC_EVIDENCE.json`，由本目录 `.gitignore` 排除，不在 Git 再发布第三方完整源码。报告不附病例可回连名单；模型训练暴露只公布聚合计数。

研究状态仍为 `experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。本文建议不是新的下载、安装或 Slurm 授权。
