# work_space 逐文件目录

路径全覆盖不等于逐个二进制内容验收。说明依据见最后一列。

| 文件 | 用途与上下文 | 静态结构 | 依据 |
|---|---|---|---|
| [work_space/.DS_Store](../../../../work_space/.DS_Store) | 根配置或其他文件；macOS Finder目录显示元数据；保留，不作为实验数据 |  | path_rule+metadata_and_path |
| [work_space/README.md](../../../../work_space/README.md) | 根配置或其他文件；所在目录的说明入口；历史说明可能滞后于最终验收 | 标题: 比赛研究工作区 | path_rule+document_headings_only |
