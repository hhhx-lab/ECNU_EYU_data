---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# r3停止后独立修复要求

记录UTC：2026-10-15T07:45:57Z。本文件是已发现缺口的修复/验收要求，不是新的attempt、运行包或提交授权。保留预注册原件、冻结r3副本、CPU4087034及全部失败/取消请求，不原位补丁后重跑。

## 全原始WT来源邻接

1. 原网格先枚举全部原始WT26连通组件，保留稳定ID、reference SHA、geometry、原始完整物理体积及small标记。
2. 所有原始组件都经历同一transpose/crop/resampling/export身份追踪；使用全来源计算目标组件source-degree，不仅遍历286个小组件。对其所在病例的非小组件必须参与邻接，否则无法区分小灶与大灶合并。
3. 分类先依据全来源邻接和训练/返回支持计算，最后才筛选286个小参考及28个r2官方未检出条目。保存全来源表和全目标邻接表以便独立复算，不能只增加最终类别名称。
4. 二值独立追踪与完整多类WT支持的交集仍是可观测支持代理，不能宣称识别了多类竞争中的唯一真实来源；无目标重叠/竞争/复杂合并分裂保留不可判定类别。
5. 对消失、裁剪、合并/分裂、保留但matcher失败分别留直接证据；检出标签与物理small资格不互相替代。与r2身份、286/28分母和actual restored geometry/SHA联合验收。

## 必须新增的测试

- 小灶与非小原始WT组件合并：全来源degree应为2，不能报survived_match_failed。
- 小小合并、单源分裂、同一目标多源且单源多目标、非小灶fragment：原始small资格不改变。
- 裁剪/重采样消失/标签竞争/返回支持无目标/官方matcher漏检分开；含空区域、标签4、非1mm、<27与=27mm³。
- 真实nnU-Net变换集成测试必须覆盖至少一对原始来源到同一目标，不能再用一例一个来源的通过代替合并覆盖。若真实数据无此边界，保留真实无该边界说明，并在相同运行接口构造边界；不虚构真实病例。
- 独立复算全来源邻接、稳定ID一一性及筛选286/28；分类规则单测不能替代生产枚举函数集成测试。

## 提交审计工程修复

r3另有held allocation contract mismatch；现有本地失败仅含assertion，不够定位具体差异。

- 下次合法操作先只读取得4087034终态及原始launch/held相关证据，不再重复本次进度快照，不复用旧reservation。
- 新独立attempt必须在assert前以排他审计保存原始scontrol输出、完整actual/expected及逐字段比较结果。数字/单位等价只按Slurm结构化语义规范化，不能删资源断言或把不匹配当通过。
- 对时间、内存、CPU/GPU、非exclusive、stdout/stderr新根、Command逐项核验；持有唯一canonical job后任何失败都禁止release，并保留定向取消审计。
- 在reservation、test-only之后、held之后及release之前重核FAILED/停止guard，避免检查后出现阻断仍放行。各步互斥，不能并发两个操作共用VPN生命周期。

## 其余真实阻塞

原E旧预测未在限定目录找到，需定位历史文件及SHA，不能按JSON顺序造ID或用重新推理伪装历史输出。概率两点最低payload145.2905 GiB；当前扣其他已授权预留后空间不足，必须获得核实可用容量，不删旧证据、不降float32、不缩103、不挪未授权路径。

补齐真实数据和测量后再选单变量P/S/L；历史曝光无追踪就记未知，不把概率/小灶体素比当参数梯度，不提前训练或提交K5。
