# epoch004重建数据训练1mm Diffusion：ECNU R1

用户于2026-10-13明确授权执行已审阅的R1方案。授权包含独立工程修复、数据验收、四模态smoke及门控后的训练，以及完整合同不变时的受控TIMEOUT恢复；不代表科学优越性批准。

研究状态始终为`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。

当前P0部署后静态预检在磁盘容量检查处停止，未执行sbatch、无canonical job ID。冻结包错误地要求100 GiB，而本阶段只输出轻量证据；路径级实际可用78265712640 bytes（72.89 GiB），与此前/public总盘df显示26T不同，原因尚未确认为quota或其他文件系统机制。不把总盘空闲替代路径级容量。失败及已占用远端r1根永久保留，不重用、不覆盖，不自动新attempt。

P1、四模态GPU smoke、人工复核和正式训练均未开始，不能跳过上游门。每模态预算200000成功optimizer更新，不是175 epochs。只回传轻量证据，不下载全量NIfTI，不上传Drive，不自动接入S2或评价locked104。

9项本地P0测试pass；远端bash -u dry-run、py_compile、真实runtime import和远端9项测试均未到达，不能宣称远端静态pass。详见[阶段记录](reports/STATUS_20261013.md)。异常后按冻结方案停止该分支，未启动自动重试heartbeat。

用户随后要求仅查询账号容量与占用：[只读存储报告](reports/STORAGE_OCCUPANCY_20261013.md)确认账号home/data/projects各路径均返回2 TiB总容量、72.89 GiB可用，而/public总盘可用25.67 TiB；quota工具未返回明细，限制类型待管理员确认。主要已统计目录zdn_deepcpg874.70 GiB、G1 566.42 GiB（epoch004重建355.26 GiB）、SegMamba-main135.90 GiB。未清理/迁移/改失败合同/重提。

随后[清理候选只读筛查](reports/CLEANUP_CANDIDATES_READONLY_20261013.md)记录pip下载缓存9.20 GiB、618个Conda压缩包10.19 GiB、索引缓存1.17 GiB，较低风险候选合计20.56 GiB；SegMamba解压数组94.61 GiB仅3例抽查一致，须1475例全量验收和依赖确认后另获授权。未清理、移动、安装或提交，不默认删除环境、浏览器、原始/重建数据、模型或旧失败。

[两个项目身份与G1分级清单](reports/TWO_PROJECT_IDENTITY_AND_G1_CLEANUP_20261013.md)确认zdn_deepcpg是单细胞甲基化预测、人/鼠H5占851.76 GiB；G1旧缺失T2W V3高分辨率mask/weight80.47 GiB理论可重建，但1030份原空间transform均缺失，暂不能删。23个非最终主run checkpoint64.70 GiB和run3104668中间输出12.84 GiB仅归档审查候选，历史引用/备份未核；epoch004重建、正式数据/权重和旧失败均保护。本次仅整理报告，不进行远端腾挪或清理。

`plan/`保存冻结方案、执行授权与身份；`operator/`为单次有界操作及独占审计；`runtime/`为本批次独立代码；`slurm/`为本批次提交入口。实际作业状态以canonical提交回执、sacct和阶段完成证据共同判定。
