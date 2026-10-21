# 比赛工作区整理总入口

整理批次：`20260918T153706Z`（北京时间 2026-09-18 23:37:06 开始）。范围为整个 `work_space`，包含 G1、G2、S1、S2、S3、S4、S5 和论文图表。

本次只做本地分类、目录规范化和完整性复核，不训练、不访问服务器、不操作 Git、不删除任何原件。所有新目录索引、说明和审计资料均为 `experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`。这不改变历史实验自身的状态。

## 从哪里看

| 模块 | 入口 | 重点 |
|---|---|---|
| G1 | [分类、文档和结果时间线](modules/G1/README.md) | VAE微调、缺失T2W补全、扩散模型训练、图像质量评估 |
| G2 | [分类、文档和结果时间线](modules/G2/README.md) | 数据划分、生成数据接收、QC、人工审查与数据发布 |
| S1 | [分类和文档](modules/S1/README.md) | 多任务分割代码、数据接口、运行资料 |
| S2 | [分类、文档和结果时间线](modules/S2/README.md) | 原E比赛基座、增强探索、FixV3、Screen125、Extend175、官方MET、推理提交 |
| S3 | [分类和文档](modules/S3/README.md) | 保留其独立分割分支，不猜测远端结果 |
| S4 | [分类和文档](modules/S4/README.md) | SAM2-UNet分支 |
| S5 | [分类和文档](modules/S5/README.md) | SegMamba、第三方依赖与许可证 |
| 论文图表 | [图表与素材](modules/paper_figures/README.md) | 原图、可编辑源码、图注、PDF |
| 跨模块 | [历史总览文档](modules/ROOT/README.md) | 工作区说明、S1/S2总结、FixV3笔记 |

先读 [研究路线与关键版本](RESEARCH_LINEAGE.md)，再进入模块。逐文件最终分类以 [FILE_CLASSIFICATION_FINAL.csv](FILE_CLASSIFICATION_FINAL.csv) 和各模块的 `FINAL_CLASSIFICATION.md` 为准；初始自动清单保留供追踪。

## 实体目录规则

| 名称 | 内容 |
|---|---|
| `10_code` | 原 code 目录，含历史工程及第三方代码 |
| `11_project_*` | S1/S2原有完整项目包，内部布局不拆散 |
| `20_data` | 数据、潜变量、预处理或数据说明，按原包保留 |
| `30_results` | 原结果、训练记录、推理输出、实验与失败证据 |
| `40_documents` | 原方法、计划、环境及操作文档 |
| `50_slurm` | 调度脚本及其配套资料 |
| `60_cloud` | 原云端操作与恢复资料 |
| `70_tests` | 原测试代码 |
| `80_submission_container` | 既有提交容器资料，仅归档，不构建 |
| `80_paper_figures` | 原论文图表集合 |

只有原先存在的目录才做规范化，没有为了凑模板创建空目录。目录名不加虚构实验日期；整理时间由本批次统一记录，实验日期来自原路径或具体证据。

`code`、`data`、`results`、`docs` 等旧路径保留为相对符号链接，指向同一个原件，不是第二份数据。43项迁移保持原目录层级深度，避免破坏代码中的 `Path(__file__).resolve().parents[...]`。日常浏览优先用本目录的分类入口；批量扫描不要同时跟随新旧入口，否则可能重复统计。

冻结实验的 run ID、attempt编号、Dataset编号、内部文件名和 SHA 文本不改。像 `27。result`、上游工程的 `main 3` 等名称仍保留为历史身份，清晰名称由索引补充，不能凭外观随意改掉合同标识。

## 完整清单

| 资料 | 用途 |
|---|---|
| [FILE_CLASSIFICATION_FINAL.csv](FILE_CLASSIFICATION_FINAL.csv) | 17,787原文件的最终分类、日期来源、原路径、新路径、大小、SHA、属性 |
| [DOCUMENT_REGISTER_FINAL.csv](DOCUMENT_REGISTER_FINAL.csv) | 633份文档、表格、参考及作者资料逐项登记；初始631项清单亦保留 |
| [EXPERIMENT_AND_COLLECTION_REGISTER.csv](EXPERIMENT_AND_COLLECTION_REGISTER.csv) | 56个实验/结果集合及本地计划、失败、完成标记索引；父子集合可能重叠，不相加计算总文件数 |
| [CHECKPOINT_REGISTER.csv](CHECKPOINT_REGISTER.csv) | 61个checkpoint文件，保留路径与SHA，不加载权重、不自动指定优胜版本 |
| [EVIDENCE_METADATA.csv](EVIDENCE_METADATA.csv) | 历史JSON中有限的状态、时间、job和实验元数据；不是重新验收 |
| [DUPLICATE_GROUPS_PRESERVED.csv](DUPLICATE_GROUPS_PRESERVED.csv) | 2,282组同大小同SHA的重复件，全保留，不去重 |
| [CACHES_AND_PLACEHOLDERS_PRESERVED.csv](CACHES_AND_PLACEHOLDERS_PRESERVED.csv) | 337个缓存、系统元数据或占位文件，全保留 |
| [SOURCE_BACKUPS_PRESERVED.csv](SOURCE_BACKUPS_PRESERVED.csv) | `.bak`、`.bak2`源代码备份，全保留 |
| [SENSITIVE_PATHS_INDEX_ONLY.csv](SENSITIVE_PATHS_INDEX_ONLY.csv) | 配置类敏感路径只列索引、属性与SHA，不转录内容 |
| [ALL_DIRECTORIES.csv](ALL_DIRECTORIES.csv) | 原始目录、原属性、是否空目录与新位置 |
| [HISTORICAL_ISSUES.md](HISTORICAL_ISSUES.md) | 旧格式问题、SHA差异、缺日期及不能推断的事项 |

这里的“文件分类完整”指所有原文件都有记录与用途类别，并不表示逐份重新科学验收或重读所有二进制内容。二进制影像、checkpoint、压缩包不解包、不改写，通过SHA保证原样留存。

## 完整性与回退

- [BASELINE.json](BASELINE.json)：整理前逐文件SHA、字节数、inode、mode、owner、mtime、扩展属性摘要。
- [MOVE_PLAN.json](MOVE_PLAN.json) 和 [MOVE_JOURNAL.jsonl](MOVE_JOURNAL.jsonl)：43个实体目录的新旧路径与实际迁移日志。
- [ROLLBACK_PLAN.json](ROLLBACK_PLAN.json)：逆序路径恢复方案；未执行。恢复时也应将新索引和别名转存，不删除原件。
- [VERIFICATION.json](VERIFICATION.json)：整理后全量复核结果。
- [FINAL_ACCEPTANCE.md](FINAL_ACCEPTANCE.md)：最终中文验收与限制。

18个原空目录留下带批次号的历史说明，原目录inode保留；原有 reservation 仍是既有锁，不被释放、重建或当成成功。新增说明属于整理元数据，不属于旧实验的运行证据。重新执行会检查“目录成员必须完全一致”的老工具前，应先查看空目录登记，不应盲目运行。

## 后续新增约定

新实验在所属模块 `30_results/experimental_unvalidated` 下创建独立 `任务_YYYYMMDD_rN`，实验开始后冻结其身份名；不要覆盖旧r版本或失败attempt。新文档用 `YYYY-MM-DD_主题_文档类型.md`；历史日期未知就标 `undated`，不能用下载日期冒充实验日期。新任务的代码版本、配置、split、checkpoint SHA和证据应一并登记。

本次不修改工作区外的 `data_space`、仓库根目录、`time_pipeline`，也不替用户提交未提交改动。
