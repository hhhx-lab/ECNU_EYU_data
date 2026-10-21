# R9固定val生成评价与选模 R1

2026-10-18用户要求执行生成质量评价与最佳模型选择；这是新评价授权，不重训/重提R9。原训练包/失败/checkpoint只读保留。experimental_unvalidated，operator_approved=false，formal_gate_status=not_run_not_passed。

## 冻结规则

- 四模态独立候选20000、40000、60000、80000、100000、120000、140000、160000、180000、200000；共40模型。各候选绑定R9包SHA、实际checkpointSHA、RUN_CONTRACT/完整checkpoint训练配置/data_identity/schedule/safe_boundary。不得用latest默认或只验第一模态。
- 固定membership val103/69患者组；依R9已发布标签94阳性、9无病灶。9例NA/no-op不进SSIM分母，不能自动exclude真实错误或改823/103/test104；test不读图、不评分。
- 每候选seed20261013-16，按case/crop/modality/base_seed的SHA256派生，跨候选不含step，使用相同噪声。冻结EDM Heun18/CFG1/churn0、FP32 eager、64cube，随机训练偏移关闭，按训练XYZ中心crop/pad；大病灶用冻结训练CSV的64cube重叠tiles，不resize标签或丢小类。
- 标签条件TC/WT/ET/RC保持原通道；原CSV行用于生成覆盖，原始1mm前景26邻域连通域用于病灶评分，不把重叠tile体积重复加权。生成cube以固定Gaussian权重混合；仅前景替换进重建目标，前景外字节不变，记录覆盖/finite/支撑事务QC。SSIM bbox内背景由该事务保持，须解释其可能使SSIM偏高，不等于完整分布质量。
- 主指标：重建目标per-lesion bbox SSIM，病例内用唯一连通域原体素体积加权，病例间等权，再4seed等权。数据保持已冻结全病例zscore；SSIM/PSNR固定data_range=6，继承原nnunet评价约定，不按候选极值调整。SSIM明确XYZ裁窗，沿Z做7x7 Gaussian(sigma1.5) valid平面SSIM，任何维度小于7则全bbox population-moment SSIM；不跳过薄/小病灶或返回背景兜底。高优，距全局最大值<=1e-6取最早step，未取整原始float参与选择。
- 次要目标：同一生成体（前景生成、背景仍重建图）与authentic真实val图比较，不据此换选点；不为次要目标换成真实背景。authentic按原生真实T1N/T1C/T2F统一mask各自zscore，使用相同physical_spacing_v2和R9目标grid/统计mask，独立记录来源SHA/归一化统计/geometry。无新全源扫描，仅val103源文件。
- 输出per-lesion/per-case/per-seed原始SSIM/前景MSE/MAE/PSNR与ET/NETC/SNFH/RC和<=27/>27体素分层。PSNR无穷明确NA/perfect-match原因，不截成100；不凭背景SSIM选点。
- 所有候选完整同一94阳性分母、4seed和QC通过才可选择。患者组bootstrap20000/seed20261013保留组内访视并维持病例等权，输出候选均值95%CI/选择相对200k的paired差CI、组级W/T/L与未校正双侧exact sign-test p；4生成seed不是独立患者。选模后的CI为探索性，不是无偏外部泛化CI。

## 门与运行边界

先本地合成XYZ/非对称/RC/薄病灶/边界crop/tile重叠/聚合/并列/患者组bootstrap/身份错配测试；再CPU计算节点读取val参考和审计全部40checkpoint、真实A100小样本重复性与支撑事务；未pass不放行全量。四个独立A100评价job，各8CPU64G48h，FP32单GPU，不恢复/改训练。只在独立目录一次held审计释放，无自动retry/newattempt/降batch或依赖降门。

不持久化全量生成NIfTI，不下载原图/中间模型；只写轻量指标/身份/进度及有限预注册可视图。容量必须当次测，既有179.822GiB不放行无关数据。评价程序/协议冻结前不看候选分数。完整结果后可拉回轻量数据与所选四模型，用户“最佳模型”交付范围不等于替换S2增强器。

有限可视图固定为每模态每候选按排序第一阳性病例、seed20261013、最大唯一连通域bbox中间Z；面板依次重建/生成/真实，固定[-3,3]窗与红色病灶边界。参考缓存为103例标签/重建/真实zscore的NPZ，只存远端独立评价目录，不下载；不是全量生成影像，不覆盖R9。提交前通过val marker目标shape计算33bytes/voxel压缩最坏上界、临时RAM/磁盘裕量与16GiB结果余量，当次df不足则停。

量化警告继续保留，不能把warning当job失败，也不能降级身份/几何/finite/SHA/非法标签真实错误。独立全量/P1/P2过去waived_by_user_not_pass不转成实测pass。本轮工程/QC与开发集选择完成也不构成人工临床质量审批或分割增益；后续人工复核、S2接入、test授权另行处理。
