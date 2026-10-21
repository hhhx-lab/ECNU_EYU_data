# R7数据合同与快速依赖提交授权

用户原话：“修订，修订后不要全量检测了，太慢了，直接快速进入训练！”

本授权承接R6支持合同修订建议：真实信号统计mask与全原生FOV几何保留域分离，保持原标签与零值，不扩mask/造信号、不删体素病例；原两覆盖门仍false。无新独立全926诊断/逐连通域人工质量门，记waived_by_user_not_pass，不将豁免称实测通过。

允许一个新CPU预处理及四个新单A100依赖训练作业在快速静态/合成CPU工程检查与实测容量通过后一次held提交。四GPU afterok新CPU，开始训练前检查完成marker与数据身份，失败不启动。内置生成输入的SHA/finite/合法标签/几何检查仍执行；不重复源诊断扫描，不安装依赖，不覆盖原失败。

训练/恢复/固定val103选点与test104锁定预算继承原计划，除明确列出的mask/几何和门控顺序变更外不改。受控TIMEOUT仅完整checkpoint合同不变才可独立恢复；其他失败停止通知，不自动重试。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
