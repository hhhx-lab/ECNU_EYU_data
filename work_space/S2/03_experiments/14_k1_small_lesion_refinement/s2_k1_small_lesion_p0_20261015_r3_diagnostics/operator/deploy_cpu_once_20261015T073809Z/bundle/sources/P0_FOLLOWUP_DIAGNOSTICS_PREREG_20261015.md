---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# P0后续诊断预注册（查看新分类与概率结果之前）

用户2026-10-15明确要求执行四步。承接20261013执行版，保留r2成功和r1失败。新批次`s2_k1_small_lesion_p0_20261015_r3_diagnostics`只做诊断，长训S/L、K5和旧模型改写不在本次提交范围。

## CPU补充

- 原始完整训练标签必须由固定1035 train IDs和患者映射读取，不用1mm预处理小组件重新定义资格。这里“原始”指预处理前的完整训练参考标签，映射实际含raw与corrected；校正来源及声明源路径原样单列，不冒充未经校正的raw。记录SHA、geometry、26连通区域完整实例、稳定ID、物理volume<27mm³；同时保留非小组件身份以识别碎片来源。影像生成/完成来源单列，不能冒充真实采集影像。
- fixed103原始WT小参考286实例逐个复现真实transpose/crop/resampling_fn_seg/export。独立二值身份掩码作为追踪通道，结合完整标签变换后的WT支持区；它是追踪工具，不是原标签多类竞争的理论证明。
- 记录standalone训练支持、完整WT交集支持、返回原网格支持及实际往返WT组件的重叠邻接。无支持、crop外、标签竞争、合并、分裂、复杂/无目标重叠分别记录；不把28个MET未检出全称消失，不强制把不可判定塞入三分类。
- 与r2稳定ID和官方detected逐例绑定；28个未检出条目逐例表，另有全部286表。凡追踪结果和r2实际geometry/标签不一致即停止。
- 原E/B175/D50只使用已验收、能获得实际预测掩码的资产。只有JSON体积或记录顺序不足以配对；缺原E预测时列blocked，不能凭相同体积造ID或把重新推理冒充旧结果。

## 概率合同

冻结K1-175 SHA `28e81950b675cd0f5ca26cc1509f9271b2fa827b524f379024b417c0a5402204`，K1-150 SHA `ba7ec030030913ace2939091d07f500ee7041d0f9e615a1fe16f535d805c3338`。单A100非exclusive、sequential_in_process、预处理/导出workers0，不训练，完整float32概率保持远端独立目录，不覆盖历史预测。提交前按实际geometry核容量并扣除其他已授权活动批次冻结的存储需求；空间不足不降精度/缩病例/删旧数据。

原多类argmax始终单列控制。WT阈值0.10/0.15/0.20/0.25/0.30/0.35/0.40/0.45/0.50不改；baseline RC=4支持区逐体素保持，支持区外只有WT总概率>=阈值且严格大于RC概率才分配WT，WT标签取1/2/3中最大概率。相等不选择WT；不引入或吞掉RC。原argmax与该受限决策不是同一工作点，即使阈值0.50也不宣称相同。

分别记录训练网格原实例映射掩码与外扩1体素ROI、原网格ROI内max及邻近大灶混淆；消失掩码不能以空数组0代替，要使用几何ROI或标不可评估。175/150两个点都做既有预注册弱响应/低响应分组，不依结果改阈值。官方MET同时列原值/paired共同病例均值/n/W-T-L/CI/未校正p，bootstrap20000 seed20260831，保护全部WT FP、WT Dice/全部大F1与ET/TC/RC，不能只报召回。

## 曝光/深监督/梯度

先绑定旧K1源码、sampler、loss和G1 checkpoint/CSV谱系。旧events不足即historical exposure unavailable。后续短程无optimizer.step诊断独立采样种子/次数与不确定性、原始身份同步空间变换和深监督尺度；不伪称175轮历史曝光。不把小灶体素占比或logit概率当真实网络参数梯度；未实测梯度则L优先级hold。

## 最终选路

补齐诊断后才写选路记录，P/S/L只选一个先试；技术/科学保护门沿用执行版，未冻结的ET/TC/RC及全部F1保护项在训练前补齐。标签往返不是模型上限；五路线异预算比较和fixed103开发调参只作exploratory，不作因果或正式优越性结论。每次只一次即时快照、VPN操作闭环；失败保留停止、不自动新attempt。
