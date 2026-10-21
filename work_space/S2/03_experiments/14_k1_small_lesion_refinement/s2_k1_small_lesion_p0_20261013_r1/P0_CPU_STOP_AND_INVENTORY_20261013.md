---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# P0资产验收与CPU主动停止记录

## 当前结论

输入与源码清单已通过；CPU诊断未完成。唯一新job `4051333`（`s2k1p0cpu1`）经held审计后释放，但本方随后发现实例匹配适配器漏测空参考区域，已核实作业身份并定向发出取消请求。回执为 `cancel_requested_known_runtime_edge_bug`，**不是已验证的CANCELLED终态**。本次不再次查询进度，不原位修复或重提，不自动创建新attempt；旧训练、推理、MET及失败证据不改。

## 时间与证据

| 操作 | 本地UTC | 远端UTC | 证据 |
|---|---|---|---|
| 单次只读清单 | 12:53:06.862582至12:53:36.531414 | 12:42:12.247585至12:42:31.710640 | `operator/readonly_inventory_20261013T125306Z/REMOTE_INVENTORY.json` |
| 唯一held提交及释放 | 部署目录时间戳13:03:49 | 12:53:43.209264 | `operator/deploy_cpu_once_20261013T130349Z/LAUNCH_RECEIPT.json` |
| 定向停止 | 完成13:05:48.747244 | 12:54:44.015341 | `operator/stop_known_unsafe_cpu_20261013T130539Z/STOP_RECEIPT.json` |

日期均为2026-10-13，Asia/Shanghai为UTC+8；本地与远端时间分别记录，不用一个时钟替代另一个。清单hostname=`login1`，`/public`总盘248T、已用223T、可用26T、90%；这不是新实验路径配额的独立检查。三次操作各有本次VPN关闭回执；没有持续SSH或循环轮询。

## 已完成的工程检查

- 515份input/reference逐路径SHA与冻结manifest一致；14份依赖SHA通过，1035/103 split数量与隔离通过。
- K1 selected175 SHA=`28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204`；150 SHA=`ba7ec030030913ace2939091d07f500ee7041d0f9e615a1fe16f535d805c3338`。
- K1/K4各103个既有预测、没有已保存的概率npz；本次未重跑旧推理或官方评估。
- 清单读取installed源码/version，不冒充实际import。后续部署预检确实完成两套现有环境的实际import和非空合成测试，但未覆盖所有边界，不能称完整运行验收。
- CPU计划SHA=`d9d353513e5ce6e95e905c843f6174db17c3e651f8a060fff07876b2edbcf387`，13份静态文件、63份installed软件SHA冻结。新CPU合同范围仅fixed103标签往返与K1/K4实例身份，不代表全部P0已执行。
- Conda base本地10项审计/稳定ID测试此前通过；这些测试未运行真实panoptica空区域匹配，不能替代漏测边界。
- 停止后本地复核计划及13份静态文件与transfer_bundle SHA完全相同，11份Python语法和Slurm语法通过，10项本地测试复跑通过。见 `operator/FROZEN_LOCAL_VERIFICATION_20261013.json`；只检查本机，没有新的远端查询或完整CPU诊断pass。

此前12:42:26至12:46:25的panoptica导入timeout快照原样保留。其失败与后续资产验收并存，后续成功不改写历史失败。

## 官方小灶单位纠正

冻结panoptica 2.1.0/BraTS-evaluation 0.0.8的NIfTI reader提取spacing，`instance_evaluator.py`计算 `volume_ref = voxel_count_ref * prod(voxelspacing)`。既有K1输出的volume/voxel_count有15种比例，不恒为1。因此官方小灶资格是**原始完整区域病灶物理体积<27 mm³**，大灶为>=27 mm³；不是原始体素数<27，也不在crop或1mm训练空间重新分类。旧稿与旧结果不覆盖，本批并列记录体素数、spacing、物理体积和原始身份。

P0-A的WT46例/286小灶本来按官方JSON volume<27计数，故计数不变。K1/K4/原E/B175/D50 TP仍为46/52/52/53/58；没有稳定身份矩阵，不能据此推断共同漏检或逐例新增/失检。

## 停止根因与后续修复门

`runtime/official_identity.py:21`的 `match_arrays` 无条件进入官方matcher。冻结 `panoptica/instance_matcher.py` 的 `map_instance_labels` 有 `max(ref_labels)`；空参考区域会触发空序列错误。官方完整评价流程对零实例另有处理，本方适配器未正确承接该边界。这是本方代码及预检缺口，**不是已观测到的OOM、CUDA或数据SHA漂移**；取消时未见FAILED/COMPLETE标记，不能断言作业已实际触发该异常。

恢复前必须在新的独立合同中修复并实际测试：双方空、仅预测非空的FP、仅参考非空的FN、非空正常匹配、label4、物理体积/身份/.2阈值，以及与官方完整流程的计数一致性。不得吞异常、忽略空病例、减少103病例或把没有参考的区域当作FP=0。冻结r1运行包、reservation、launch audit、停止回执及可能的远端输出全部保留；新的attempt需另行授权，不用r1原位替换。

CPU诊断、稳定身份交叉表、标签往返结果尚未验收；175/150概率、曝光/梯度/G1谱系仍未执行。未提交S/L175或K5。fixed103用于选择及报告，只能exploratory；硬标签往返不是理论召回上限，工程pass也不等于科学优越性。
