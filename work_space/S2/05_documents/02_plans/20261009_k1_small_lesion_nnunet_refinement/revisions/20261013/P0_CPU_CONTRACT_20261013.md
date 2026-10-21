---
artifact_status: experimental_unvalidated
operator_approved: false
formal_gate_status: not_run_not_passed
---

# P0 CPU参考身份、标签往返与既有预测匹配合同

承接用户授权的修正执行版；2026-10-13新清单已核验515文件SHA、冻结依赖、1035/103 split、K1-150/175 SHA。installed源码读取不替代真实import，本批静态预检须在两套现有环境实际import及合成边界测试，才能唯一held提交CPU作业。

## 1. 官方单位纠正与冻结

远端brats_eval中的panoptica2.1.0/BraTS-evaluation0.0.8，NIfTI路径通过SITK/Nibabel checker提取spacing，`instance_evaluator.py`计算`volume_ref=voxel_count_ref*prod(voxelspacing)`。K1既有2254参考记录volume/voxel_count包含0.125/0.25/2.0600等15种比例，不恒为1。**本批官方小灶资格为原始完整region组件物理体积<27 mm³，>=27 mm³为大；不能按原始体素数<27或训练空间重分类。**既有46例/286个是按JSON的volume<27得到，数值不改。

这落实执行版“先核实volume单位”的门控；纠正用户原稿及旧文字的“27原始体素”假设。所有旧原件/原始MET保留。本批同时记录体素数、spacing、物理体积、原始geometry和小灶资格。ET=3，TC=1/3，WT=1/2/3，RC=4；26邻域的cc3d实际默认须用合成对角组件验证。

## 2. 本批范围

仅fixed103：原始完整参考身份、冻结标签预处理→one-hot导出往返、K1/K4既有预测的逐实例官方匹配。train1035实例清单、曝光/深监督/梯度/G1实际训练谱系、K1-150/175概率GPU诊断均另列后续阶段，不冒充本CPU批已完成。

父数据根没有raw目录；fixed103共享input/reference已核515SHA，可以原样只读用于本批。不得查看/改写attempt-11。K1/K4预测SHA须逐文件对其原PREDICTION_SHA256绑定；官方输出的病例/TP/FN/FP与原CSV逐例一致。原E/B175/D50身份矩阵暂未纳入本批，不用两路交集冒充五路交集。

## 3. 方法与身份

- 使用现有segmamba nnunetv2=2.8.0、冻结plans/default reader/transpose/crop_to_nonzero/resampling_fn_seg，以及实际`convert_predicted_logits_to_segmentation_with_correct_shape`反向导出。
- 用四通道非零掩码构造完全相同crop，不重新训练/normalize图像；normalize不改变seg，data resampling与标签诊断无关。每病例记录bbox、spacing、原始/裁剪/1mm shape；-1 background映射0后one-hot，5类合法；one-hot作为有界scores走实际概率重采样+argmax，明确非真实模型logits。
- 所有中间/往返NIfTI写新根；旧MRI/预测/模型只读。往返不叫所有1mm模型召回上限。
- 官方region提取和cc3d使用安装的panoptica approximator，完整参考组件身份由case/region/原标签SHA/shape/spacing/origin/direction/组件坐标SHA组成；不按summary list index配对。
- 官方MaxBipartiteMatching使用DSC、匹配阈值1e-6；匹配之后DSC>=0.2才计small TP。FP为所有未匹配预测，不是只有小预测FP。匹配参考ID通过full-volume mask验证，记录相同体积身份不合并；每reference ID每method恰一行。
- K1/K4逐实例计数及volume multiset须与原官方JSON/CSV一致；不一致立即FAILED留证，不自动放宽。两路intersection/new/lost只有完整stable-ID矩阵后计算。
- 本批label-roundtrip对每参考灶记录是否匹配及Dice；逐实例预处理输运/语义竞争/合并/分裂分类尚需下一独立mask诊断，不凭往返FN直接标“下采样消失”。

## 4. 验收与提交

CPU无GPU，单进程4CPU64GB，最长1天，不启动多worker或multiprocessing queues。真实导入、合成26连通/physical-volume边界/标签4/身份区分/官方匹配/.2阈值与nnU-Net 1mm同spacing往返静态测试通过后，逐文件SHA/py_compile、独立reservation、held launch audit再释放，禁止重复提交。

启动写P0_CPU_STARTED，任何异常写FAILED后非0退出；最终103病例/4region参考及3method矩阵（K1/K4/label_roundtrip）计数齐全，官方MET往返输出、原预测计数交叉核对、VALIDATION pass和COMPLETE.ok；拉回轻量JSON/CSV/MD/ok/log，NIfTI保留远端。每次仅即时快照，不等待训练、轮询或重复SSH；新失败停止，不自动建attempt。

同fixed103用于开发与报告仅exploratory，工程pass不代表科学增益。三项研究状态不变，S/L训练/K5均不授权。
