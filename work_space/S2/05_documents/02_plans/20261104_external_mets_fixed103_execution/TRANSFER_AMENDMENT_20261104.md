# 执行授权补充：Mac临时中转指定资产

2026-11-04，Asia/Shanghai。本补充保留PLAN.md/PLAN.json原快照，不覆盖原S0失败审计。

用户在ECNU资产端点IPv4/普通HEAD均失败后明确授权：“下载到 mac吧，开个临时目录，传完给它删了”。因此仅资产输送边界调整为本机临时中转：2025固定digest镜像layers/config、Bio-Intel固定revision ZIP及官方BrainIAC.ckpt。推理及最终资产仍在服务器，不回传MRI/预测，不下载其他模型。必要公开源码/元数据按原计划取得，不改变数据/模型/参数/评价/作业图。

本机只用现有Conda标准库/curl及已经配置的网络出口，不装Docker/新依赖、不改系统代理/DNS/PATH。临时目录由本批`.env`给出，必须新建、0700、限定于本机缓存专属目录；开始前记录可用空间与资产大小预算。模型不进入项目Git、LFS、正式模型目录或其他共享位置。

2025registry raw manifest/config/layers逐digest核验；镜像展开需要保留原WorkingDir/Entrypoint/Cmd/层次及diff_ids，不能换latest或重造算法。Bio ZIP逐bytes/SHA及无越界目录检查；BrainIAC首次下载仅建立准确bytes/SHA，后续远端核对同SHA再由隔离作业加载。全部URL/revision/digest与许可依据保留，不在审计保存临时registry token、signed redirect或凭据。

传输必须独占新远端asset目录，先 fresh容量再SSH/SCP，每份临时与远端bytes/SHA一致且资产集合完整后，生成传输验收。只删除本次明确登记的临时副本及其专属空目录；不清空全缓存、不删除远端资产、旧数据、失败记录或本机其他文件。传输失败保留清单并报告，未经成功验收不把删除或提交标成完成。

新增Bio专用Conda环境及作者锁定依赖已独立获用户清单授权，不改旧环境。临时中转成功也不等于模型运行通过：仍须隔离容器/完整资产、容量、CPU测试、正式脚本和冻结包，随后第一批J01–J04 held审计释放；smoke门通过后才全量J05–J07。不提前提交占位作业或恢复监控。

实际执行入口：[R1状态](../../../03_experiments/15_external_mets_fixed103/s2_external_mets_fixed103_20261104_r1/STATUS_20261104.md)。科学状态继续`experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。
