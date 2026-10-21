# BraTS_evaluation 来源说明

本目录是官方 BraTS evaluation 仓库的本地快照，用于 2026 Task 1 评估口径核查、脚本参考和 G2/S1/S2 指标对齐。

## 来源

- 官方仓库：<https://github.com/BraTS/BraTS_evaluation.git>
- 拉取日期：2026-06-15
- 拉取 commit：`88e3e39cd5c4137b0831345c78d16bd393624c3a`
- 本地处理：已移除内部 `.git` 目录，作为当前项目的普通参考代码管理。

## 用法提醒

1. 评估 Task1/MET 时重点看 `brats_evaluation/configs/config_mets.yaml` 和 `brats_evaluation/metrics_parser.py`。
2. 若后续要更新官方版本，重新从官方仓库拉取并更新本文件中的 commit。
3. 不要在该目录中写团队自定义脚本；团队自定义脚本放在各自 `work_space/` 或项目明确的工具目录。
