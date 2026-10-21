# R9 固定验证集生成评价与选模

用户授权：2026-10-18“生成质量评价和最佳模型”。只评价已有R9，不重训、不改R9/旧失败/数据split，不自动替换S2。

协议见PROTOCOL.md；配置见config/RUN.json。运行runtime/prepare.py构建val103参考并CPU审计40候选；四路真实A100 smoke全部COMPLETED0:0且COMPLETE、重复性/覆盖/容量/时长门通过，才由operator/manage.py full独立held审计释放四路完整评价和依赖选模。

operator/manage.py smoke仅执行一次准备与四smoke，冻结协议/配置/源码/测试/提交器SHA。operator/manage.py snapshot仅一次即时轻量只读快照。full不重提smoke，不自动失败重试。凭据由约定外部env加载，不在本包存储。

选择前须40候选完整103例/94阳性/9NA、4seed证据及SHA/QC一致；SELECTION.json是开发集探索性最佳推理候选，不等于临床质量审批、分割增益或外部泛化。人工复核未完成；test104不读图评分。预注册有限可视图：每模态每候选，按排序第一阳性病例/seed20261013/最大病灶bbox中间Z；三面板重建/生成/真实，固定[-3,3]，不按分数挑图。

experimental_unvalidated / operator_approved=false / formal_gate_status=not_run_not_passed。旧monitor保持PAUSED，未提交前或有失败不得宣称分数/最佳模型已产生。
