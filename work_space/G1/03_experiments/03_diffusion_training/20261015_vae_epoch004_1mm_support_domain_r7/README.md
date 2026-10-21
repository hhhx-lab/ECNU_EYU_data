# G1 epoch004严格1mm支持域修订R7

用户授权修订统计mask/几何域，并豁免再次独立全量检测，快速提交训练。
原R5/R6不修改；STRICT_ORIGINAL_GATE=false、NONZERO_COMPARISON_GATE=false。

保留真实三源统计mask、外部置零、原生标签与零信号。几何域为完整原生FOV，不称脑组织支持，不label扩mask、不膨胀、不删病例。四模态共享全FOV严格1mm、float32和标签0-4，保存统计/mapping及各类原生/目标计数。不同spacing不能承诺374体素一一保留，重采样整类丢失仍停止；独立全病例/逐连通域质量复验和人工等待记waived_by_user，不伪造pass。

一次CPU192 8CPU90G24h生成926=train823/val103输入；四单A100作业afterok依赖该CPU并再次检查完成marker/数据SHA。batch16/64³/eager非DDP/AMP/seed20261013，AdamW lr2e-4 beta1=.9，每路200000成功更新，完整5k/final/safeTERM checkpoint。fixedval103原选择规则保留，test104锁定不读图/不评分。

快速检查限定合成CPU回归/真实import/逐文件SHA/Slurm资源路径/当前容量（只读header核算）与独占held审计；不再次全量读取源影像诊断。必要输入SHA、finite、geometry、label和输出检查内置实际预处理，不能豁免损坏文件。

预提交四GPU是本次用户明确授权的依赖式启动变更，不冒称原“全部P1/P2门过后才提交GPU”已满足；源与输出身份仍强制，预处理失败则训练不能开始。失败保留停止，无自动下一attempt/降batch/重复提交。

artifact_status=experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed；仅exploratory。状态见STATUS_20261015.md；冻结后README不可改。
