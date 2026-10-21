# 工作区整理最终验收

批次：`20260918T153706Z`。本机验收UTC：`2026-09-18T15:47:55.458723+00:00`；本机本地时间：`2026-09-18T23:47:55.458861+08:00`。

## 完成结果

- 已整理整个work_space，覆盖G1、G2、S1至S5及论文图表；43个外层实体目录规范化并保留旧路径兼容链接。
- 17,787个原文件，34,746,825,213字节；1,435个原目录全部保留。全部原文件有最终用途分类，无未分类文件。
- 633份文档、表格及作者资料有逐项登记；56个实验或结果集合按主题和日期建立导航，61个checkpoint文件按路径及SHA登记。
- 2,282组重复件、337个缓存/系统元数据/占位文件、两份源码备份均保留，无去重、无原件删除。
- 18个原空目录加入历史说明，原目录inode保留；既有reservation不删除、不重建、不释放。最终空目录数为0。

## 完整性

全量重新读取原文件并计算SHA256，17,787份全部与整理前基线一致。大小、inode、mode、owner、mtime和扩展属性摘要一致；全部旧文件路径指向同一原件。1,433个Python路径保持原层级深度。详情见 [VERIFICATION.json](VERIFICATION.json)。

原目录名中的实验身份、内部运行结构与历史SHA文本保留。旧路径只是别名，扫描时不要同时递归跟随新旧目录。父子结果集合存在交叠，不能把56项的文件数直接相加。

## 已知历史问题

3个.json实际为键值文本，1处静态SHA清单与整理前现存脚本不一致，均原样保留并独立登记。6,352条历史SHA引用未按本地路径解析，不推断远端文件缺失或损坏。详见 [HISTORICAL_ISSUES.md](HISTORICAL_ISSUES.md)。

本次没有重跑训练、推理、官方MET或完整科学验收；“整理验收通过”不是“模型效果正式通过”。原E比赛基座和E-SNFH已区分，Screen125与Extend175不合表，图像质量与分割收益不混写。

## 交付入口

- [总入口](00_START_HERE.md)
- [研究路线和关键版本](RESEARCH_LINEAGE.md)
- [最终逐文件分类](FILE_CLASSIFICATION_FINAL.csv)
- [最终文档登记](DOCUMENT_REGISTER_FINAL.csv)
- [新旧目录映射](MOVE_PLAN.json)及[实际迁移日志](MOVE_JOURNAL.jsonl)
- [回退方案](ROLLBACK_PLAN.json)：未执行，回退前必须复核并转存新增元数据，不能删除旧原件。
- `DELIVERY_AUDIT.json`：最终索引、链接与JSON检查。
- `CATALOG_SHA256.json`：最终新增资料的SHA封存；此前阶段性登记保留为历史记录，不覆盖。

所有本次新增资料的状态：artifact_status=experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed。未修改工作区外内容，未联网，未安装依赖，未执行Git或上传操作。
