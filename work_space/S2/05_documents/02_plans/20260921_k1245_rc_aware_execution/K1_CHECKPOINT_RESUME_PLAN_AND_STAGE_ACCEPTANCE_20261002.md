# K1 累计175轮续训方案与准备验收

研究状态：`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。本次只准备独立续训，不中断原训练，也未提交新GPU作业。

## 现场状态

| 项目 | 2026-10-02即时证据 |
|---|---|
| 原训练 | job `3827543` / `s2k1train175r3`，gpu18，`RUNNING 0:0` |
| 原时限 | 5天；此前9天延时请求权限拒绝，本次未再请求 |
| 最新训练日志 | Epoch 69 |
| 最新完整checkpoint | `checkpoint_latest.pth`，`current_epoch=50`，249747530 bytes |
| checkpoint SHA256 | `bbff9a65cf72d3451e6a693e09fbc81b42857803968ef822f6d3412fb5ce846c` |
| 完整状态 | 网络292张量、优化器106项、AMP与日志finite；日志数组长50 |
| 下一轮LR | `0.0007387283529495573`，对应累计epoch50/175，不重置剩余预算 |
| 真实CPU恢复 | 原trainer真实加载网络/优化器/logger/current_epoch，逐项相等，pass |
| CUDA AMP恢复 | 本次CPU不能验证；必须在未来单A100作业STARTED前验证 |
| `/public` | 248T总量，222T使用，约27T可用，90% |

只读源检查本地UTC `2026-10-02T03:42:24.024407Z`、远端UTC `03:31:00.707453Z`；独立准备操作源快照远端UTC `03:38:43.500965Z`、CPU验收远端UTC `03:39:25.614241Z`，本地收取结果UTC `03:50:16.802396Z`，本次VPN关闭回执UTC `03:50:23.177101Z`。两台时钟约差11分半，不能直接混用计算时长。

源检查见[原checkpoint检查](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/k1_resume_source_inspection_20261002_20261002T034139Z/SOURCE_CHECKPOINT_INSPECTION.json)；准备验收见[远端阶段证据](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/k1_resume_stage_20261002T034840Z/pulled/evidence/REMOTE_STAGE_VALIDATION.json)与[VPN操作回执](../../../03_experiments/12_k1245_rc_aware/s2_met_aug_k1245_20260923_r3_fallback/operator/k1_resume_stage_20261002T034840Z/LOCAL_OPERATION_RECEIPT.json)。拉回JSON逐文件SHA通过。

## 独立运行包

本地根：`work_space/S2/03_experiments/12_k1245_rc_aware/s2_met_aug_k1_175_resume_20261002_r1`。

远端根：`/public/home/zqchen/projects/ECNU_EYU_data/work_space/S2/experimental_unvalidated/s2_met_aug_k1_175_resume_20261002_r1`。

计划SHA256：`7f1de71844ee75e8ff78831acd01f4f6281813996adb65250fe61e51c931e5f6`。运行包11份文件SHA/编译、Slurm与派生环境`bash -n`通过；本地7项门控测试通过，含终态允许空squeue、RUNNING不提交、非TIMEOUT失败停止、175学习率不重置。远端stage完成逐文件SHA、冻结源码校验、实际trainer导入、RC27/config真实加载、1035/103 split验证和CPU完整恢复。

运行时只读复用原训练目录中的冻结trainer和原研究plan/config/pool/auth。37份源合同/源码SHA绑定已验收远端版本，**不假定本地部署前草稿与远端逐字一致**。源远端plan实际文件SHA为`94b5f324c6b268d384700e3a809fe78d8a259b0feb6d2508ca990381ea8bf3bc`，与原STARTED及远端auth/config绑定一致；本地部署前plan的`620f66...`不能替代这一运行合同。原文件未改写。

不安装或改变环境。凭据仍只由约定env加载，远端Python仍为segmamba；新的结果/日志/事件/CPU测试只写新根，不写原输出。

## 超时后的唯一提交流程

1. 使用`operator/access_resume.py submit-after-timeout`，每次只做一次原job即时squeue/sacct快照。仍RUNNING/PENDING/初始化则返回不提交；若原job已COMPLETED 0:0则不续训。
2. 仅原job确为TIMEOUT且已不在squeue时继续。任何意外FAILED标记、关键错误、非TIMEOUT失败或SHA/合同漂移都停止，保留证据，不原位重跑或自动新attempt。若TIMEOUT产生了原FAILED包装标记，需人工核实其只代表TIMEOUT再调整独立验收合同，不能跳过。
3. 重读**超时后最终完整latest**：网络/优化器/AMP/logging/current_epoch、历史LR、下一轮LR与原175曲线验收；源候选25到恢复epoch逐点SHA/finite/model合同检查。今天latest50只是诊断，不作最终冻结。
4. 把最终latest和已存候选复制到新结果目录，latest副本0600，原候选副本0400；保留源job TIMEOUT、STARTED、stdout/stderr及训练日志/JSON证据的只读副本与SHA。不得用会连带改动源的硬链接。
5. 在新测试目录CPU真实恢复最终副本，重新预检；排他reservation、唯一`sbatch --hold`、原子LAUNCH_AUDIT记录canonical job id，再释放。提交或释放失败不得再次sbatch。
6. 新作业调用安装版nnUNet CLI `--c`，禁止`-pretrained_weights`或缺checkpoint回退从头训练。单A1008CPU64GB5天，GPU网络/优化器/AMP/logger/current_epoch与175LR全部核对后才写RESUME_STARTED/PASS。

最后训练验收候选25/50/75/100/125/150/175、final175、103例nnUNet validation、状态finite和旧证据SHA不变；按EMA最高/并列最早选点。官方MET及paired仍另作验收，不能以CPU恢复或训练完成代替科学结果。

## 解释与风险

源`save_every=25`，因此最后完整checkpoint之后**所有未保存轮次**均须重算，不只超时时那部分单轮训练。源最新log69/ck50说明这一区别已实际存在。

原175 horizon、seed20260724、RC27/G1/增强机制保持不变；这不是重启原E、不是新的225轮实验，也不混入原Screen125 equal-budget横评。`F_STRICT_PREREG_GATE=false`继续如实报告，K5仍等待K1/K2各自科学门，同fixed103选择与报告仅exploratory。

当前结论：**准备阶段pass、最终恢复输入未冻结、续训作业未提交、GPU恢复尚未验证**。不能保证尚未执行的后续GPU训练绝不出错；超时后必须再次走完整验收。
