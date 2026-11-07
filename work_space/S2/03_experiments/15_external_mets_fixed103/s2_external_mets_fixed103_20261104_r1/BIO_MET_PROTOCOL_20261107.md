# Bio-Intel加入同fixed103统一MET评分

用户2026-11-07授权“对它进行一样的评分，加入统一评分里”。使用4319113的完整A5+B2作者后处理预测，不重跑推理，不改模型/标签/参考/split。继承4319117已完成三路合同和精确评价源码，原冻结attempt、FAILED、逐例成绩及bootstrap区间永久保留。

## 输入与执行

1. 独立attempt `compare_bio_four_models_r1`，单个held CPU作业：cpu_192G、8CPU、64GiB、24h、无GPU、无requeue，核验后释放同JobID。检查当次分区/QoS/队列/容量，不以旧容量或空quota输出放行。
2. 同fixed103内部验证103扫描/69患者组、同参考压缩文件逐SHA；test104不读。Bio完成marker/manifest/STARTED/运行合同与JobID/SHA绑定，sacct allocation/batch均须COMPLETED0:0。
3. 同Python3.10.18、BraTS-evaluation0.0.8、panoptica2.1.0、NumPy1.26.4；二进制及全部评价Python/YAML源码SHA须与原合同一致。原run_official_met_eval.sh逐字复用，mets、27mm³、parser DSC0.2及原NSD/匹配参数不改。
4. CPU作业校验Bio预测与参考native几何、finite、标签0–4，再对Bio103例运行原评分器，保留72指标/NA。只读取这103份GT分割，不读取原MRI或其他测试集。
5. 复用已完成EXT2025、K1_175、original_E_200逐例CSV、历史复现和原216条配对行；不重算三路评分/CI。新Bio与三路各72项配对仅消费共同finite病例，20000次患者组重采样、seed20260831；同原算法同时输出扫描等权及患者等权区间。
6. 新独立报告输出4×72=288条均值、6×72=432条配对及四路72项完整表。旧216条配对保持原顺序/数值，追加216条涉及Bio的配对。评分前后复核父结果SHA，避免修改历史。

## 验收与边界

提交前执行bash -n、3.10 AST、模拟held/release与汇总回归、服务器实际环境导入和输入身份预检；这不是新模型GPU smoke或质量pass。最终只有sacct COMPLETED0:0、同JobID COMPLETE、103例/72列、288均值/432配对及轻量产物SHA通过才称完成；未完成不能展示Bio新分数。无自动retry或恢复监控。

fixed103是内部验证，不是锁定测试。Bio完整集成训练曝光、2025公开训练池重叠、K1开发选模边界保持；不称官方排名、独立泛化或科学优越性。RC/其他指标逐项显示有效n，NA不补0；TC/WT不含RC，HD95等沿用原评分器值。豁免仍为waived_by_user_not_pass。

状态保持experimental_unvalidated、operator_approved=false、formal_gate_status=not_run_not_passed。最终轻量CSV/报告/manifest可回传；无新增模型/MRI/NIfTI下载、安装或删除旧数据。
