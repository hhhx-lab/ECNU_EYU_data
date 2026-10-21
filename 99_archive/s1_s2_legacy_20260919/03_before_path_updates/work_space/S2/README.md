# S2：分割、样本增强与比赛提交

整理于2026-09-19。这里是实体文件分类，不是快捷方式视图。

| 目录 | 内容 |
|---|---|
| [01_code](01_code) | 活动代码、测试和必要的包内README/许可证 |
| [02_datasets](02_datasets) | 本模块的数据集、样本、潜变量或可复用增强数据 |
| [03_experiments](03_experiments) | 按研究路线分组的独立实验及其原始结果 |
| [04_models](04_models) | 从代码或混合目录中分离的模型权重 |
| [05_documents](05_documents) | 方法、计划、报告、环境和历史说明文档 |
| [06_operations](06_operations) | 现有Slurm及云操作脚本，旧远端路径只作历史记录，不自动执行 |
| [07_submissions](07_submissions) | 比赛推理、传输、提交包、排行榜及容器保留资料 |

当前分类内原文件 9848 份，9,740,858,317 字节。每个数据集、模型或结果批次内均有 `00_说明_20260919.md`。

原E比赛基座是Dataset264 completion+Focal的epoch200，SHA为`4e5ff8d4a29fc498e4d91f9b0a71c34b818da6cd07d359ccabcc72dcb49e4267`；不是E-SNFH。Screen125与Extend175按独立路线保存，不能混做等预算结论。

历史文档、冻结实验runtime及manifest不改写。活动路径配置见项目根.env和tools/project_paths.py；7份路径适配代码的原件另存99_archive。重新训练或提交前须另做实验预检，不能直接复用旧Slurm绝对路径。
