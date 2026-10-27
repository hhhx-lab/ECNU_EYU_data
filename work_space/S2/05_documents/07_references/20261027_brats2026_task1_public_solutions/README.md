# BraTS 2026 Task 1 领先队伍公开方案深度调研

调研日期：2026-10-27。范围为 BraTS-METS 2026 Task 1，不混同 GoAT、PEDs 或往届比赛。

本目录保留公开来源证据和可复核报告。调查不涉及训练、服务器连接、镜像拉取或未知代码执行。

原始论文全文抽取、源码/README镜像和网页正文仅保留在本机，被本目录`.gitignore`排除，不随Git再次发布第三方完整材料。Git保留中文报告、链接、身份/权重/镜像元数据和`RESEARCH_AUDIT.json`中的本地证据大小/SHA；其他电脑可按原始公开URL自行取得授权资料。物理索引覆盖不等于所有全文均由Git分发。

## 阅读入口

- [完整中文调研](REPORT.md)：21队身份、论文、方法、代码、模型与复现边界。
- [链接直达表](LINK_INDEX.md)：逐队论文页面与PDF、源码、重点文件、公开权重和演示材料。
- [论文与检索来源](SOURCES.md)：32篇METS与1篇跨赛道论文、检索路径和访问限制。
- [证据验收](RESEARCH_AUDIT.json)：覆盖计数、文件SHA256与操作范围，不是模型科学验收。

重要结果：MicroBT冠军的COMPASS-METS源码/训练路线公开，但冻结权重没有下载地址；TheRealBoys三套权重与Bio-Intel模型包有实际公开Hugging Face文件元数据。文件存在不等于已加载、已推理或复现比赛。

## 证据文件

| 文件 | 内容和边界 |
|---|---|
| `TEAM_MAP.json` | 截图21个目标名、并列名次、对应论文编号、作者/机构/方法检索关键词 |
| `PAPER_EVIDENCE.json` | 33篇公开PDF正文抽取、URL、UTC、PDF大小/SHA256；未保留PDF二进制，文本可能断词/公式乱码 |
| `TEAM_EVIDENCE.json` | 20个队伍与aymuos个人账号的公开成员/身份链；不采集私有联系方式 |
| `REPOSITORY_EVIDENCE.json` | 23个GitHub仓库元数据、递归文件树、README及Git tree SHA；不是已核commit SHA |
| `CODE_EVIDENCE.json` | 重点公开配置/源码/模型许可的只读抽样证据，不是本项目运行依赖 |
| `WEIGHT_EVIDENCE.json` | 4个HF模型仓库的文件元数据、LFS SHA/大小与public/gated状态；没有下载权重 |
| `CONTAINER_EVIDENCE.json` | 3个Synapse镜像名的匿名manifest请求，均401，未获取镜像/digest |
| `WEB_EVIDENCE.json` | 用户指定网站、官方规则与Google学术结果；MICCAI分类只保留相关部分 |
| `EXTRA_SOURCE_EVIDENCE.json` | Bitbucket目录、作者主页、Orchestrator注册表等 |
| `SEARCH_EVIDENCE.json` | arXiv/HF/Zenodo实际检索，保留空结果和失败 |
| `RELEASE_EVIDENCE.json` | 4个重点GitHub仓库releases检索，当时均为空 |

本次仅公开调研与本地文档/索引更新；未连接ECNU、启动VPN、提交/修改作业、读取影像、下载模型/镜像、安装或执行第三方代码。截图名次是用户提供证据，不冒称独立取得完整官方最终榜。

研究状态：`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。公开方法可定位不代表在本项目已有效。
