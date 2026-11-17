# BraTS METS历年validation可用性

2026-11-17独立公开只读复核，接续11月10日评分规则资料，但不覆盖旧归档。入口是[中文报告](REPORT.md)、[结构化结论](AVAILABILITY_REVIEW.json)与[来源身份](SOURCE_INDEX.json)。

2023/2024/2025旧榜均可浏览；三个文件提交队列9615344/9615545/9615901的当前完整轮次已结束，无新轮次。2023、2024明确Challenge Closed；2025赛后管理员称重开时间待定。GLI持续评测不是METS，内部fixed103不是官方validation179；未来开放和训练暴露仍须另核。

## 本次动作

- Chrome + Playwright MCP Bridge读取官方任务页、实际旧榜和2025重开问答；一条Google公开检索辅助定位，没有上传/发帖/登录或读取秘密。
- Conda base标准库调用系统curl，有界匿名GET和精确官方POST round/list读列表，每件45秒/8MiB上限、不自动重试、不覆盖原件。一个2023辅助容器队列403已保留并停止，不影响已成功核查的三年文件队列。
- `tools/build_evidence.py`仅读本地原件，核HTTP回执的大小/SHA、队列年份/METS/dashboard绑定、完整轮次和截止UTC、关闭提示、管理员原文以及三年榜实际渲染。输出结构化复核和索引，不计算模型成绩。
- 原始wiki/文档/browser和失败响应留本机；`.gitignore`限定发布边界，Git保留原创报告、工具、来源回执及轻量队列元数据，物理索引仍登记本地原件。
- 无ECNU/VPN、依赖安装、MRI/GT/模型下载、Slurm/比赛提交或104读取。原科学三项、旧结果和失败均保持。

## 复核命令

首次构建使用既有Conda base，不安装依赖：

```bash
conda run --no-capture-output -n base python -B 02_reference_materials/05_brats_metrics_camyla/20261117_validation_availability/tools/build_evidence.py
```

脚本拒绝覆盖已生成结论；以后要查询现时开放状态应建新的日期批次，不重新运行并覆盖本批结果。验证和执行摘要见[结构化复核](AVAILABILITY_REVIEW.json)，项目索引由Conda base catalog build/check验收；本批不是科学质量或官方提交pass。
