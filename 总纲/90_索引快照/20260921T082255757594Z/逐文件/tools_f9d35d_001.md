# tools 逐文件目录

路径全覆盖不等于逐个二进制内容验收。说明依据见最后一列。

| 文件 | 用途与上下文 | 静态结构 | 依据 |
|---|---|---|---|
| [tools/__init__.py](../../../../tools/__init__.py) | 本地新旧路径解析及目录结构检查工具；Python实现或运行工具；只做AST静态阅读，不导入执行 | 顶层定义:  | curated_context+ast_only |
| [tools/project_paths.py](../../../../tools/project_paths.py) | 本地新旧路径解析及目录结构检查工具；Python实现或运行工具；只做AST静态阅读，不导入执行 | 顶层定义: local_settings, get_path, relocation_tables, resolve_legacy_path, main | curated_context+ast_only |
| [tools/test_project_paths.py](../../../../tools/test_project_paths.py) | 本地新旧路径解析及目录结构检查工具；Python测试 | 顶层定义: LayoutTests | curated_context+ast_only |
| [tools/test_s1_s2_layout.py](../../../../tools/test_s1_s2_layout.py) | 本地新旧路径解析及目录结构检查工具；Python测试 | 顶层定义: RefinedLayoutTests | curated_context+ast_only |
