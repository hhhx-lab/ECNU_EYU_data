# S1/S2 实体细分验收

本机验收时间：2026-09-19 15:11:11 +08:00；UTC：2026-09-19 07:11:11。

## 实际变更

- 132项文件搬迁。S1脚本分为标签准备、审计与划分；S2的51个活动脚本分为数据准备、评估与提交、组件池与掩膜、Route A、原E续训、FixV2、FixV3、环境工具八组。
- S1的两个数据加载器备份、旧Trainer和S2的旧标签检查脚本归档；空的train.py、inference.py及空README原件也保留。
- 固定划分的5份TXT/JSON和历史lesion_pool.csv从代码、文档中移入数据目录，配套说明。不推断历史病灶表与当前组件池等价。
- 文档按方法、计划、操作、实验记录、参考材料归位。两份不同的S1服务器手册均保留。
- S2服务器资料按ECNU、H20、NYU、Cloud历史包分类。环境文件、历史脚本、launch记录保持原字节，没有执行或重提。
- 60个活动代码或测试文件完成路径适配。连同导航、定位工具等，本轮83个发生内容更新的文件均保存了更新前原件。

## 保持不变

`custom_nnunet`全部30个源码文件的模块路径及SHA保持不变，保护Trainer发现和checkpoint身份。S2原实验结果、冻结runtime、模型、失败证据、原计划和SHA清单未改写。没有训练、GPU推理、服务器操作、依赖安装或Git操作。

## 验收结果

- 整理前166项回归测试通过；整理后同一166项全部通过。
- 18项路径、结构、原文件引用、固定划分、Trainer包一致性与Shell语法测试通过。Shell仅做`bash -n`，不执行。
- 3项提交打包相关测试未运行，避免创建ZIP；未验证真实数据上的完整训练或推理流程。
- 全量19,626份原文件，共36,009,759,371字节，经SHA256、大小、inode、权限、所有者、mtime核对，无丢失或变更。需要修改的工作副本，其原件在历史区通过同样核验。
- 本轮9,961份基线文件也全部核验。空目录0，活动符号链接0，源码语法错误0。
- 1,378个新导航链接检查通过，无失效链接。历史原文中的旧路径不批量重写，可用`tools/project_paths.py`查询现位置。

## 追溯

- [搬迁计划](PLAN.json)
- [实际搬迁记录](MOVES.jsonl)
- [更新前原件记录](ADAPTATIONS.jsonl)
- [最终原件位置](ORIGINAL_LOCATIONS_final.csv)
- [机器验收](VERIFICATION_final.json)
- [旧版本与原件保留区](../s1_s2_legacy_20260919/README.md)

这是本地结构整理验收，不是新的科学实验或正式性能门控。artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
