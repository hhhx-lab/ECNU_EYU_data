# BraTS评分与Camyla公开分割手稿调研

2026-11-10公开资料调研。入口：[详细报告](REPORT.md)、[Word可编辑版](REPORT.docx)、[PDF阅读版](REPORT.pdf)、[10篇手稿索引](camyla/MANUSCRIPTS.md)。

重点：2025 METS validation省略病灶级；2026官方validation不排名，7/26宣布检测/分割两张最终榜；27mm³、RC独立、FP及NA支持集需要明确。现有fixed103四路成绩是同合同内部横评，不是官方名次或完全独立泛化。

Camyla只取得10篇公开示例，不是40篇已发表SCI。最高相关为 [HRFuse-Net脑转移瘤原稿](camyla/manuscripts/10.pdf)，但代码链接404、数据血缘/表图/引用和小样本仍有缺口。

## 本地资料

- Camyla系统PDF、10示例PDF、31个ideas JSON、双语README及LICENSE：45个官方文件，31,821,359bytes，逐Git blob/大小/SHA核验。
- 另存PROTEAS原始Scientific Data论文与公开网页；不下载MRI、患者数据或权重。
- 2025/2026论坛107/93未删除主题完整元数据，选取9/11主题及其50/61回复全文。
- 官方公开API/规则wiki、两代指标代码参考、Camyla公开Issue/Release/项目页和检索范围审计。
- [归档manifest](ARCHIVE_MANIFEST.json)、[执行与验收记录](EXECUTION_AUDIT.json)、[PDF文本清单](camyla/PDF_TEXT_INVENTORY.json)。原件本机保存，Git仅原创报告、身份/索引与轻量回执。

## 复核工具

使用Conda base标准库、现有curl/pdftotext/pdfinfo与Chrome Playwright Bridge，无新增依赖。Word使用本机codex-doc-tools包装命令和本报告临时专属中文模板，7张表/3个Office Math公式；全局模板不改。LibreOffice导出发生中文字体替代，最终PDF改用同一Markdown经Pandoc HTML5/native MathML和现有Chrome打印，嵌入已安装宋体/黑体的报告字形子集，不安装字体或改全局配置。14页PDF/全文/字体/公式与所有表格已验收，见 [DOCUMENT_QA.json](DOCUMENT_QA.json)。Word和PDF来自同一源稿，但不承诺分页一致。`tools/`是公开参考归档与文档复核工具，**不是Slurm/评分/训练入口**。

`collect_forum.py`只消费公开官方API；不保存临时签名URL或登录凭据。API消息gzip传输字节与解码文本分记SHA，首次字段解析和文本解码问题在审计中保留，不影响最终原件核验。`acquire.py`拒绝覆盖已有原件，失败响应及状态保留。

不自动更新运行包、不重新评分或提交远端任务；剩余30篇及最终ranking精确实现明确未取得。研究三项状态保持 `experimental_unvalidated / operator_approved=false / formal_gate_status=not_run_not_passed`。
