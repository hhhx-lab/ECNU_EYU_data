# .claude 逐文件目录

路径全覆盖不等于逐个二进制内容验收。说明依据见最后一列。

| 文件 | 用途与上下文 | 静态结构 | 依据 |
|---|---|---|---|
| [.claude/settings.local.json](../../../../.claude/settings.local.json) | 根配置或其他文件；敏感配置或版本控制内部对象；仅登记路径与文件元数据，不读取内容 |  | path_rule+metadata_only |
