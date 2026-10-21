# ECNU 比赛项目

2026-09-19已按实际用途重新搬迁归类。不是索引镜像，主工作树没有旧目录快捷链接。所有原文件、重复件和失败记录保留。

```text
ECNU_EYU_data/
|-- work_space/
|   |-- G1/  代码、补全数据、Diffusion样本、评估、模型、文档
|   |-- G2/  QC代码、数据发布包、质量报告、运行资料
|   |-- S1/  多任务分割代码、文档、运行资料
|   |-- S2/  代码、增强数据、分路线实验、模型、文档、比赛提交
|   |-- S3/  分割代码与文档
|   |-- S4/  SAM2-UNet代码与文档
|   `-- S5/  SegMamba代码、第三方依赖与文档
|-- data_space/              共享数据契约、参考样例、参考权重
|-- 01_project_documents/    比赛要求、分工、总计划、跨模块总结
|-- 02_reference_materials/  论文、讲义、外部材料
|-- 03_reference_code/       按来源区分的上游代码
|-- 04_paper_figures/        图表、绘图源码、图注
|-- tools/                  当前路径配置与旧路径定位工具
`-- 99_archive/             旧索引、缓存、原目录、原代码及搬迁审计
```

## 模块内部

`01_code` 放代码，`02_datasets` 放数据，`03_experiments` 按研究路线放结果，`04_models` 放独立权重，`05_documents` 放方法/计划/报告，`06_operations` 放运行资料，`07_submissions` 放比赛提交。没有本地内容的类别不创建空目录。

G2的数据发布包放在 `02_dataset_release`，保持manifest、split、QC之间的配套合同。每个数据集、模型和结果批次旁均有 `00_说明_20260919.md`，说明来源、内容、本地计数、相关实验和使用限制。

[G1](work_space/G1/README.md) · [G2](work_space/G2/README.md) · [S1](work_space/S1/README.md) · [S2](work_space/S2/README.md) · [S3](work_space/S3/README.md) · [S4](work_space/S4/README.md) · [S5](work_space/S5/README.md)

## 路径与运行

本地路径由根目录 `.env` 配置，`.env.example`写明每项用途与来源。活动G1配置及G2入口已做必要路径适配，原版代码另存历史区。S1/S2的repository根位置保持不变，内部脚本已按用途分组，并调整活动代码和测试中的调用路径。

S1旧Trainer和备份、S2空模板已归档。S2固定划分及病灶表归入数据目录；服务器资料按ECNU/H20/NYU/Cloud历史包分开。`custom_nnunet`的模块结构和文件内容保留，以保持模型加载身份；冻结实验runtime及历史服务器脚本不改写。

只读定位旧文件：`conda run --no-capture-output -n base python -B tools/project_paths.py "旧路径"`。远端路径不会被替换为本机路径。

旧实验runtime、Slurm脚本、manifest和SHA清单保留原内容，不把旧命令自动改造成新提交。完整MRI原始数据不在本地，运行涉及原始影像的任务前须明确配置数据根，并对新计划做预检。本次未训练、未推理、未安装依赖、未联网、未提交Git。

## 保留与验收

[历史保留区说明](99_archive/README.md)包含旧索引和原代码位置。搬迁记录位于 `99_archive/reorganization_20260919_r2`，文件原始内容按SHA全量核验。

后续S1/S2细分记录位于 `99_archive/s1_s2_reorganization_20260919_r3`；本轮原件位于 [S1/S2历史保留区](99_archive/s1_s2_legacy_20260919/README.md)。旧路径定位工具会衔接两轮搬迁记录。

本次新说明及审计为 `experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`，不改变历史模型/实验的正式状态。
