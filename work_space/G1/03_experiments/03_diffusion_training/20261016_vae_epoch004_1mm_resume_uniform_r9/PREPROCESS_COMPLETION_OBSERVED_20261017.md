# R9预处理工程完成观测与验收边界

## 输入与本周期动作

- 用户授权统一原量化warning策略；不重提、不补点、不排除病例，固定train823/val103/test104。
- 唯一只读证据：[snapshot_20261016T220156Z](operator/snapshot_20261016T220156Z/SNAPSHOT.json)。本地UTC2026-10-16T22:02:05.016150，远端UTC2026-10-16T21:50:57.682327；两端时钟分别记录，不混用推算时长。
- 现有snapshot.py已执行一次，结束码0，finally关闭本次VPN；随后仅在Conda base解析本地JSON、stdout尾部及冻结计划。无第二SSH、源影像重扫、远端写入、数据下载或作业操作。

## 已确认事实

| 项目 | 证据 |
|---|---|
| CPU4128262及batch | sacct COMPLETED 0:0，09:59:43；squeue中CPU已离队 |
| PREPROCESS_REPORT | status=complete，cases_ok=cases_processed=926，errors=0，warnings=5 |
| COMPLETE | status=complete，cases=926，split_counts=train823/val103；FAILED不存在 |
| 统一策略 | loss_policy=uniform_original_quantization_warning_v1；label_resampling_algorithm=original_physical_spacing_v2_labelwise_threshold_no_case_exception |
| 包身份 | 60文件冻结清单SHA80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6，远端drift=[]；本地清单SHA与数量一致 |
| PREPROCESS_REPORT SHA | cae62fcfcd97a4c6d5494fee5252be36ea4d1df7cbc3279f87e3a482011c02b8；stdout report_sha256一致 |
| COMPLETE SHA | 2dab27498800acc6f43c9baccb07792ad1b4c021d007e538633f6534e43812be |
| stderr与关键错误 | stderr0bytes，快照关键错误列表为空 |
| 当前余量 | 202212638720bytes，188.3251953125GiB；不据此放行其他数据 |

stdout最终完成记录尾部reference_parents与本地REUSE_CANDIDATES冻结父清单一致：R7 package41c47c0516e9818c82319da2a370507ebc3fd773ef121f4baed786159e5a4158，R8 packagea73f614ae36e2b237f8c65929faa5547aa191a9701bcc00e689e23bbb806a745。这只是父身份字段核对，不是独立重算所有父文件或链接内容SHA。

## 五例量化警告

| 病例 | lost_classes |
|---|---|
| BraTS-MET-01110-002 | 4 |
| BraTS-MET-01132-002 | 2 |
| BraTS-MET-01166-003 | 2 |
| BraTS-MET-01245-003 | 2 |
| BraTS-MET-01339-001 | 4 |

完整警告名单来自stdout最终resolution_warnings，与REPORT/COMPLETE摘要数量5一致；快照未拉取逐例原生/目标体素及体积，不推断新增病例的大小。R9取消原01110-002补点，量化损失不转成真实错误，也不宣称全部细节保留。

## GPU状态与剩余核验

T1C4128266 PENDING Resources；T1N4128267/T2W4128268/T2F4128269 PENDING Priority。四路sacct运行均00:00:00；训练STARTED/RUN_CONTRACT/progress/FAILED/SAFE_TERM/LATEST_CHECKPOINT/COMPLETE均无。afterok依赖已不再阻塞，仍未开始训练，无新增提交或release。

本次确认CPU工程完成及完成报告核心字段，尚不称独立全量身份验收通过。当前快照没有COMPLETE的完整cases_detail/case_manifests、CSV内容、926逐例marker/index和链接实际目标明细，不能独立验收513全引用、01110-002仅seg重建、412新构建及全926索引/CSV绑定。后续应在下一周期的唯一只读操作中取得轻量manifest/索引/报告与引用metadata证据，不增加本周期查询或重扫源影像。

冻结runtime/train.py的数据消费门仍逐父包/marker/真实引用目标/六输出SHA/原生身份/CSV/逐例索引/无例外/警告一致性核验。该门尚无实际GPU执行成功证据；26CPU测试不是A100训练或质量pass。

STRICT_ORIGINAL_GATE=false、NONZERO_COMPARISON_GATE=false、independent_full_revalidation=waived_by_user_not_pass不变。nnU-Net等价和评价实现仍待独立验证，test104不读图、不评分。训练200000成功更新及后续固定val选择任务未完成，heartbeat继续ACTIVE，不删除。

experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。旧R7/R8失败与全部历史证据保持。
