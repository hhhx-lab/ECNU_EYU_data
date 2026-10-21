# 历史问题、未知项与保留决定

整理批次：`20260918T153706Z`。以下均不自动修正原件，不解释为本次整理造成的错误。

## 三个.json实际是键值文本

位于 `S2/30_results/27。result/runtime_attempt_evidence`：

- `LAUNCH_ATTEMPT_01_DUAL_GPU_CANCELLED.json`：内容为 `status=...` 等逐行键值，记录启动前取消。
- `LAUNCH_ATTEMPT_02_FAILED.json`：同类键值文本，记录物化前失败，job3396610、exit1:0。
- `LAUNCH_ATTEMPT_03_CANCELLED_GPU_BINDING.json`：同类键值文本，记录GPU绑定问题导致推理前取消。

因此JSON解析失败是历史扩展名与实际格式不一致，不是已证明的文件损坏。本次不将它们转成JSON，不改名、不删除。详见 [HISTORICAL_JSON_PARSE_ISSUES.json](HISTORICAL_JSON_PARSE_ISSUES.json)。

## 一处既有静态SHA差异

`S2/results/experimental_unvalidated/s2_met_aug_bef_epoch175_official_met_20260911_r2/STATIC_SHA256SUMS.txt` 第6行记录 `runtime/snapshot.py` 的SHA为：

`21cd3ea6ac0bbd4c3cccb315ea9aae8f338ae870eee7c802f2ac08c8fa5d04ad`

它与整理前现存脚本的基线SHA不同。清单和脚本各自原样保留，不重新生成旧清单，不用整理动作伪装合同一致。两者的原始SHA都在 `BASELINE.json` 中。本次整体验证检查的是原件有没有被整理改变，不是追认该历史静态合同通过。

详见 [HISTORICAL_SHA_REFERENCES.csv](HISTORICAL_SHA_REFERENCES.csv)：635条可定位的SHA引用与本地基线匹配，1条上述差异，6,352条未按清单相对上下文定位到本地原件。后者可能涉及远端绝对路径、未拉回的重文件或不同执行目录，不能据此声称文件损坏或实验失败；本次未访问远端补验。

## 日期与命名

目录中的日期优先保留原表示，仅作索引日期；来源标注为filename/nearest parent，不当成已核实的执行时间。只有run号、没有日期的资料标记 `undated`。文件mtime保留为文件系统属性，不冒充实验日期。

历史 `27。result` 的用途已经由其计划确认：原E在27例合成T2W内部验证上的推理与评估；计划时间UTC 2026-08-13 18:44:35，对应北京时间8月14日。其模糊原目录名仍保留，规范解释见研究路线文档。

冻结run ID、attempt、Dataset及上游源码目录名保持原样。因为一些runtime明确检查 `plan["experiment_id"] == root.name`，强行把它们全部改成漂亮名称会破坏身份合同。

## 空目录和占位内容

18个原空目录逐一加入 `HISTORICAL_EMPTY_DIRECTORY_20260918T153706Z.md`，未删除原目录。文件记录见 [EMPTY_DIRECTORY_NOTES.json](EMPTY_DIRECTORY_NOTES.json)。其中旧 `SUBMISSION_RESERVATION` 的inode和锁身份不变，不是新建reservation，也不表示释放锁。

原本仅有 `.gitkeep` 或 `.DS_Store` 的目录在 [ALL_DIRECTORIES.csv](ALL_DIRECTORIES.csv) 中标记 `placeholder_only`。这类目录不能据此宣称已经有本地训练数据或结果。额外说明属于整理元数据，不加入旧实验的科学证据。

## 分类复核

初始自动分类中的45个其他文件已逐项补分：PID归历史进程日志，`.bak/.bak2`归源码备份，`.gitignore/.gitmodules/.dockerignore`归配置，`.env*`仅列敏感配置索引，AUTHORS归第三方作者资料，`py.typed`归类型标记，`EXPERIMENTAL_UNVALIDATED`归状态证据，JS归代码。

复核后全部17,787个原文件都有明确用途类别，见 [CLASSIFICATION_REVIEW.csv](CLASSIFICATION_REVIEW.csv) 与 [FILE_CLASSIFICATION_FINAL.csv](FILE_CLASSIFICATION_FINAL.csv)。分类不代表每个文件语义都已科研复核；旧PID尤其不代表当前服务器还存在对应进程。

所有重复组、旧失败、恢复包、缓存、临时命名原件和压缩包完整保留。此次不做科学结论修订、格式修复、去重或清理。
