# 历史保留区

这里保留所有未删的旧索引、缓存、占位文件、原目录结构和活动代码适配前原件。主工作树不依赖本区的旧快捷方式。

2026-10-22维护说明：确认可再生的缓存已清理，测试源码停止Git跟踪但本地保留；下列缓存描述是九月整理时的历史状态。研究原件、失败证据、搬迁记录及历史SHA不改写，逐项处置与恢复边界见 [清理记录](../01_project_documents/05_maintenance/20261022_cleanup/README.md)。

- `20260918_index_only`：上一轮索引文件和775个原符号链接对象。链接仅保存历史target及inode，可能已无法按旧相对路径访问，不能作为现目录导航。
- `cache_files_20260919`：原.DS_Store、__pycache__及pyc，按原位置存放。
- `old_placeholders_and_notes`：原.gitkeep和上一轮空目录说明。
- `original_empty_directory_objects`：搬出内容后的原目录对象，保留本体和来源说明，无空目录。
- `source_before_path_adaptation_20260919`：7份活动代码路径适配前原件，SHA和inode保留。
- `legacy_preprocessing_job_logs`：原根目录logs中的历史日志。
- `reorganization_20260919_r2`：本次搬迁计划、事务日志、路径映射和验收。不作为新的科学实验结果。
- [s1_s2_legacy_20260919](s1_s2_legacy_20260919/README.md)：S1/S2细分后的旧源码、占位文件、改路径前原件、旧导航及腾空目录对象，原内容保留。
- `s1_s2_reorganization_20260919_r3`：S1/S2脚本、文档和运行包细分的搬迁、回归测试与验收记录。不修改旧实验状态。
