# S5：SegMamba分割

整理于2026-09-19。这里是实体文件分类，不是快捷方式视图。

2026-10-13用户明确停用此路线：[旧远端SegMamba轻量留存与删除验收](03_experiments/20261013_remote_segmamba_retirement/README.md)。已保存远端代码/配置/原指标表/曲线/日志节选和文件metadata，指定`/public/home/zqchen/SegMamba-main`已删除，账号可用增至208.79 GiB。本地S5代码/历史文档和Conda segmamba环境保留；不再将旧远端路径当可用模型仓库，轻量留存不能完整恢复权重或训练。S3/S4及原E/K未删除。

| 目录 | 内容 |
|---|---|
| [01_code](01_code) | 活动代码、测试和必要的包内README/许可证 |
| [05_documents](05_documents) | 方法、计划、报告、环境和历史说明文档 |

当前分类内原文件 575 份，7,389,087 字节。每个数据集、模型或结果批次内均有 `00_说明_20260919.md`。

未在本地发现独立结果包的分类没有创建空文件夹；原模板与占位文件在99_archive保留，不推断远端是否有结果。

历史文档、冻结实验runtime及manifest不改写。活动路径配置见项目根.env和tools/project_paths.py；7份路径适配代码的原件另存99_archive。重新训练或提交前须另做实验预检，不能直接复用旧Slurm绝对路径。
