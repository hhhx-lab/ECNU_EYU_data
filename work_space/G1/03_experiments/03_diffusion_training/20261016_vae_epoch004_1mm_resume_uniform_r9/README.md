# G1 epoch004 1mm统一量化接续R9

用户2026-10-16明确授权：采用统一策略、修好训练入口并接续。新独立批次不改R7/R8、原失败、原37上游源码或原生标签。

保留physical_spacing_v2原算法、共同真实三模态mask、全原生FOV、masked z-score、严格1mm；取消病例特判补点。目标网格导致类别消失记录loss_policy=uniform_original_quantization_warning_v1及各类体素/物理体积，不宣称全细节保留。真实来源/SHA/几何/finite/非法标签错误逐例收集，最终有错误不生成训练COMPLETE。

514候选逐源marker/实际目标/六文件SHA核验后引用；513原样引用，01110-002仅重建统一量化seg并重建marker/索引，四影像mask逐文件引用；412剩余构建。逐例输出索引/当前CSV，train823+val103共926，test104不读图不评分；无病灶rows病例显式记录，不exclude改变固定分母。

训练入口核验reference_parents、每个链接真实目标、markers、报告及数据SHA；修复marker SHA变量被Path覆盖错误，完整成功/漂移拒绝入口测试。四各单A1008CPU64G48h、64cube batch16 AMP eager非DDP，seed20261013 AdamW2e-4 beta1=.9各200000成功更新，完整5k/final/safeTERM checkpoint；预处理cpu1928CPU90G24h成功afterok后自动调度。

独立全量复验/人工/P2独立smoke等待仍waived_by_user_not_pass，当前不宣称A100实际训练质量/完整科学门通过；后续nnU-Net实际预处理等价未单独核验。原fixedval103评价/test104锁定不改。

唯一held审计提交并释放，失败保留不自动新attempt；仅轻量文件本地回传，无NIfTI/checkpoint/Drive/Git/安装/无关作业/删除旧数据。

experimental_unvalidated/operator_approved=false/formal_gate_status=not_run_not_passed。
