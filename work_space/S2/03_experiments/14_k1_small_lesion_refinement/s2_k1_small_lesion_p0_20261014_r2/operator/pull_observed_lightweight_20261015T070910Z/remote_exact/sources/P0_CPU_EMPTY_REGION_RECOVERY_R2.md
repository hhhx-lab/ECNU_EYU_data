---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# P0 CPU空区域修复独立r2执行合同

用户2026-10-14授权：修复空区域处理、补齐真实边界测试，再授权独立attempt。承接20261013 CPU范围及固定资产，不改变研究条件或旧r1。仅CPU fixed103原始完整病灶身份、K1/K4既有预测匹配、冻结nnU-Net硬标签往返；S/L训练、概率GPU诊断、K5不在本批授权内。

## 保留与唯一执行

r1计划SHA d9d353513e5ce6e95e905c843f6174db17c3e651f8a060fff07876b2edbcf387及job4051333不复用。操作先做一次只读终态快照并保留stdout/stderr/标记及SHA；如果旧job仍活动或终态缺证据，不提交新批。新根s2_k1_small_lesion_p0_20261014_r2使用独立日志、reservation、静态pass、授权和held launch audit；仅一次sbatch，失败不原位重跑或自动再建attempt。

## 修复语义

按冻结官方ConnectedComponentsInstanceApproximator提取组件。空参考返回零参考记录、FP=全部预测组件数；空预测返回每个完整参考组件一条未匹配记录、FP=0；双方空为零记录/零FP。只对双方非空调用同版本MaxBipartiteMatching。不得修改安装库、吞异常、跳过空病例或将无参考的预测默认为FP0。匹配阈值1e-6，官方检出DSC>=0.2；原始完整物理体积<27mm³为小灶，资格不按crop/1mm空间重分类。

## 真实边界预检门

在现有brats_eval环境实际运行适配器与官方完整Panoptica_Evaluator，而非源码/假mock：双方空、仅预测两个FP、仅参考两个FN、label4正常、26连通、完全不相交、DSC等于/低于/高于0.2、等体积不同身份、非零crop坐标还原、26/27mm³、非1mm体积。每例对照官方组件数、FP、reference voxel/volume/is_matched/DSC及检出TP/FN；官方NaN仍为undefined，不补零。

额外从冻结fixed103的真实病例选择空RC和含RC各一例，对K1/K4逐实例指纹及FP与原官方JSON核对，绑定case/reference/prediction SHA及geometry。预检只重算这少量实例匹配，不重跑旧GPU推理、不改旧官方结果；完整103正式诊断在唯一CPU批运行。

segmamba中真实检查空标签的nnU-Net往返、label4、非零crop及1mm同spacing身份；515input/reference、冻结依赖/installed软件、所有新文件SHA与真实import/语法/Slurm路径通过后才写ENGINEERING_EXECUTION_AUTHORIZATION.json并held提交。工程授权不改变operator_approved=false或正式科学门。CPU4核/64GB/1天、无GPU/worker、GNU MKL，保持103病例和原方法。

## 验收

COMPLETED0:0、无FAILED、103病例/4region/3method参考矩阵和病例计数、WT286小灶身份及K1/K4历史TP46/52一致，往返官方MET可解析、VALIDATIONpass/COMPLETE.ok/轻量SHA齐全。标签往返不是理论上限，二路线交集不是五路交集，往返FN不直接当下采样消失。完成拉回仅计划/config/manifests/evidence/logs/JSON/CSV/MD/ok，排除NIfTI/checkpoint/events。每次一次即时快照，不sleep/循环轮询/持续SSH。
