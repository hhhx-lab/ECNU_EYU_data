# 共同生成评价预注册R1

## 对象与边界

七月G1 V3四150000 checkpoint vs R9已冻结所选T1C120000/T1N40000/T2W200000/T2F200000。不按本批分数重新选择旧/新权重。固定R9评价reference/REPORT SHA13e57c28d597e7801e503219da15350eda68a044550ed0d436edee84c1c70d3c与SELECTION SHA541d0fd245ba66841f35574aed3ad227efa9363c580211a1c8733f528d808902。沿用已审计val103严格1mm真实/自重建参考缓存与标签，94阳性共同病例，9阴性不生成、NA不填1，test不读取。开发集反复用于选择，不能宣称无偏外部泛化或科学优越性。

用户再次明确新模型指“R9自重建主 SSIM”选出的最佳模型，不是原图R2。选模指标仍为旧选择程序的自重建主SSIM；本批另列统一真实主指标与自重建次指标，不把本批分数称作原选模分数，也不重新选模。不存在公认唯一“最权威”的医学合成总分，本批是固定透明、多指标的探索性对照。

## 共同生成

同XYZ、TC/WT/ET/RC四标签条件，同64cube FP32 eager单A100、EDM Heun18/CFG1/churn0/sigma_data1/sigma_min.002/sigma_max50/rho7/snr_shift0。seed20261013..16，按病例/模态/裁窗起点/base seed派生且不含模型版本；每次调用前重置，旧新共用初始噪声。各唯一26邻域连通域bbox外扩5，确定性64cube覆盖、32步幅大病灶tiles，重复裁窗去重；不resize标签、不合并相邻病灶、不丢薄小病灶。全部生成上下文Gaussian混合，不复制参考背景、不mask置零后评分。来源空间/量化lost_classes保留参考报告，不自动改分母。

旧模型训练为per-content crop zscore，R9为whole-case zscore。对旧每块预测，先用同一真实参考裁窗的有效内容均值与样本std(ddof1、std<1e-8取1)做pred*std+mean，回到共同whole-case真实zscore。边界padding不参加统计，pad预测不参加混合；统计仅由真实参考固定，不从生成结果估计、不为了SSIM优化。共同1mm域换算不等于重现旧原生训练分布，需作为推理适配公开。旧原始zscore输出另报真实参考未适配敏感性结果，不能择高值改变主结论。自重建次指标也使用同一真实标定后的生成体，不另标定。

## 主指标与误差

主：真实参考前景中心的3D SSIM-map，skimage.metrics.structural_similarity，7x7x7均匀窗、data_range6、K1.01/K2.03、population covariance(use_sample_covariance=False)。使用真实生成上下文；只在每个唯一组件前景中心读取SSIM-map，不把bbox空背景分数纳入平均。组件间按体素体积、病例间等权、四生成seed均值，固定负值不截断。原图是生成相似性参考，不是条件纹理真值；随机合理纹理也可能SSIM低。

同组件halo5裁窗内计算SSIM-map，窗口半径3、组件位置距裁窗边至少3；真图像边界统一reflect padding。覆盖mask必须覆盖halo5内有效原图位置，未覆盖/非finite/非法标签/几何或SHA错误失败不降级。小病灶也使用同7窗，邻域含真实生成上下文，另报<=27/>27体素及ET/NETC/SNFH/RC分层；tiny得分不能单独当可诊断性证明。

分列前景MAE/RMSE/PSNR(data_range6)、参考与生成前景对比差、|zscore|>8比例、zero比例；PSNR完美匹配为NA加原因，不截100。生成4seed的前景波动只描述多样性，不作为越大越好。真实/自重建SSIM同实现同生成事务，旧原始敏感性同实现。有限固定可视化只供后续检查，不自动称人工pass。没有ImageNet FID/LPIPS冒充医学权威，也不生成未经验证的综合分。

## 配对统计

四seed先在病例内平均，不作四倍患者。共同94病例保留同患者多访视，患者组cluster bootstrap20000/seed20261013，病例等权，输出每版均值/CI、R9减旧paired差CI、患者组W/T/L。双侧exact sign p仅为组级方向性检验，四模态主SSIM做Holm校正；不拿sign p当病例等权平均差检验。其余指标/分层探索性，报告有限分母、方向和NA。无独立测试集，显著性不等于临床可用或分割增益。

## 执行与验收

本地合成背景抬分、非对称XYZ、边界/薄/大病灶覆盖、噪声配对、归一化逆变换、统计与八权重strict CPU加载测试。远端只复用现有Conda segmamba，不装依赖；CPU验证所有参考缓存SHA/103病例身份/八权重实际SHA及strict加载。四smoke各同时测试旧新、重复性/RC/小/大/阴性，估时不能超过48h的85%；其后四全量afterok全部smoke，最后CPU汇总afterok四全量。单次held资源/依赖/脚本审计后释放，不重复sbatch/自动retry；marker/包SHA/COMPLETED0:0全满足才称完成。资源/容量当次检查，不以旧余量放行。

只保存轻量JSON/CSV/进度/审计/有限PNG，不持久化生成MRI或新参考NPZ，复用旧缓存只读。原文件/失败/选择不改，旧四checkpoint仅缺少远端可信副本时上传原字节约205MiB到本批；不下载源图、不用Drive/Git/安装/删除数据。手动查询，不恢复heartbeat。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
