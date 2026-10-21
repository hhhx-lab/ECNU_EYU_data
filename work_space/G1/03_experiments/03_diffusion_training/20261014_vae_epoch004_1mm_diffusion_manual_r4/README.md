# G1 epoch004 1mm Diffusion 独立 R4

用户明确授权“修复预检超时并建立独立新批次提交”。R3远端CPU回归120秒超时，尚未sbatch，完整源码/SHA/审计/失败保留，不原位覆盖。R4继承用户验收豁免：P0_STRICT_ACCEPTANCE=false，P1人工/P2独立smoke等待waived_by_user，不伪造通过。

单次诊断在远端固定OMP/MKL/OPENBLAS/NUMEXPR=1并禁用CPU测试的CUDA可见性；同一原R3七测试21.3秒全部通过。旧timeout没有分阶段输出，不能断言唯一根因。新预检使用相同CPU环境、300秒有界时限、unbuffered及逐阶段日志，超时也保留stdout/stderr/阶段回执。

一个CPU预处理和四个afterok单A100训练；train823/val103严格1mm、真实T1N/T1C/T2F共同mask、locked test104不读取影像。四模态独立随机init，seed20261013、64³、batch16、AMP、非DDP/eager、AdamWlr2e-4beta1=.9，各200000成功更新。训练参数与R3逐字段一致，不降级；活动37源码未改。数据实际消费SHA/finite/geometry/合法label、空间预算、安全TERM和完整checkpoint仍保留。入队不等于已训练或科学质量通过。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。仅exploratory，不下载大数据或Drive/S2部署。
