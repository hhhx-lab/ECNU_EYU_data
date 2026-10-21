# G2：数据接收、质量控制与发布

整理于2026-09-19。这里是实体文件分类，不是快捷方式视图。

| 目录 | 内容 |
|---|---|
| [01_code](01_code) | 活动代码、测试和必要的包内README/许可证 |
| [02_dataset_release](02_dataset_release) | 保留manifests/splits/qc内部合同的数据发布包 |
| [05_documents](05_documents) | 方法、计划、报告、环境和历史说明文档 |
| [06_operations](06_operations) | 现有Slurm及云操作脚本，旧远端路径只作历史记录，不自动执行 |

当前分类内原文件 558 份，195,298,515 字节。每个数据集、模型或结果批次内均有 `00_说明_20260919.md`。

历史文档、冻结实验runtime及manifest不改写。活动路径配置见项目根.env和tools/project_paths.py；7份路径适配代码的原件另存99_archive。重新训练或提交前须另做实验预检，不能直接复用旧Slurm绝对路径。
