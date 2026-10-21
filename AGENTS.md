# ECNU项目协作与总纲维护

## 先读总纲

开始G1/G2/S1/S2、数据、模型、实验或整理任务前，先读 `总纲/README.md` 和对应模块专题。路径查 `总纲/当前索引.md`，旧路径用 `tools/project_paths.py`。不要仅凭历史README或文件夹日期推定当前实验状态。

## 每次更新

- 每次修改代码、文档、数据合同、文件位置，或新增训练/评价/选择/失败恢复证据后，结束前同步总纲。
- 有研究语义变化时，更新对应专题、`总纲/维护/知识登记.json` 的批次描述/证据/复核时间及 `总纲/更新记录.md`。
- 从项目根用现有Conda base运行：
  `conda run --no-capture-output -n base python -B 总纲/维护/catalog.py build --reason "变化说明"`
  然后运行相同入口的 `check`。没有检查pass时，不宣称总纲已同步。
- 新增文件可自动路径收录，但实验结果结论必须人工核对。若缺证据，明确写待核实。

## 研究与保留边界

- 不删除旧文件、旧失败、重复来源或历史快照，不为修正文档而改写原始实验runtime、SHA和审计。
- 原E与E-SNFH、标准B与增强B、125等预算与175扩展、图像重建与分割增益必须分开。
- 数据split、checkpoint SHA、运行合同、评价版本、paired分母及CI/p方向必须绑定。
- 总纲与新研究汇总默认 `experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`；工程验收pass不等于科学优越性。
- 保持Conda优先，不用sudo pip，不混用系统/Homebrew Python；未经用户允许不安装依赖、不改PATH或VS Code的Homebrew Git设置。
- 索引/文档任务不授权远端操作、作业提交、Git提交或平台上传。凭据仅保留在约定env中，禁止复制进总纲。
