# R9最佳模型：固定val103规则所选，已下载

**本目录是R9四模态所选最佳checkpoint的本地交付入口。2026-10-21北京时间12:38下载与二次SHA验收完成。**

训练来源为VAE epoch004自重建的完整病例train823/val103，严格1mm统一量化；不是缺失T2W补全全集，也不是新原图R2训练。四路R9训练最终各200000成功更新，选模从每路20000..200000共十候选选择，所选四模型不必都是200000。

## 四份本地模型

| 模态 | 所选step | 本地checkpoint | 大小bytes | 主病灶SSIM | authentic原图次SSIM |
|---|---:|---|---:|---:|---:|
| T1C | 120000 | [diffusion_120000.pt](t1c/diffusion_120000.pt) | 55806207 | 0.4388053354 | 0.1763220244 |
| T1N | 40000 | [diffusion_40000.pt](t1n/diffusion_40000.pt) | 55806143 | 0.5208156814 | 0.4378083264 |
| T2W | 200000 | [diffusion_200000.pt](t2w/diffusion_200000.pt) | 55806207 | 0.4979818169 | 0.3857347179 |
| T2F | 200000 | [diffusion_200000.pt](t2f/diffusion_200000.pt) | 55806207 | 0.5806873985 | 0.4579002466 |

共223224764bytes，约212.88MiB。绝对目录：

`/Users/hwaigc/比赛+课题/ECNU_EYU_data/work_space/G1/03_experiments/03_diffusion_training/20261018_r9_fixedval_selection_r1/best_models_fixedval103`

| 模态 | SHA256（远端传输前/后、本地及二次核验均一致） |
|---|---|
| T1C | `c66c9610634a6e1a06ffdf5df201f627036000c053100a9b8f02299922f892f4` |
| T1N | `ec418ac7dbdcb72cf6202af30f10d6b2ca79249294842e85e875f7b895a5d6f9` |
| T2W | `27ea3175e26b7aeaaba28f30a553cd71fa2eab0c7e79989355da46ef39aa28a5` |
| T2F | `c642ea88fe73e479a2ddf5d663fe220c5bcada31e03cfea9d981ff24d0acc93b` |

远端来源目录为 `/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_diffusion_1mm_vae_epoch004_20261016_resume_uniform_r9/training/<mod>/weights/diffusion_<step>.pt`。逐文件完整绝对路径、大小、SHA、选点、两端UTC和验收路径见[MODEL_MANIFEST.json](MODEL_MANIFEST.json)。没有另导出或改写checkpoint，是原文件字节下载。

## 最佳的定义与限制

- 全量评价4159189-92、CPU选模4159193及batch均COMPLETED0:0；四evaluation COMPLETE和SELECTION已完成，非只做smoke。40候选各103例，94阳性评分、9例无病灶NA/no-op不入SSIM分母；4seed20261013-16。
- 主参考是VAE自重建图，病例内唯一病灶体积加权SSIM、病例间等权、四seed均值；距最大值<=1e-6取最早step。次参考是真实原图与同一生成事务，不据次得分换选点。Heun18/CFG1/churn0/FP32/64cube；具体口径见[冻结协议](../PROTOCOL.md)。这是本批验证规则下最佳候选，不代表其他指标都最好。
- 训练包60文件SHA `80f432b454ea80cf3d4142cb87d194c7913c0eabd2da473b3821b06d10df63f6`；评价包24文件SHA `3486d52cfa995aff8cd439932384d8393224b90630cb878793aba01f7b41d2d4`。本次逐文件无漂移。
- 原始SELECTION SHA `541d0fd245ba66841f35574aed3ad227efa9363c580211a1c8733f528d808902`，与selection/COMPLETE一致；本次核对全部40候选摘要SHA/分母/种子/得分及最早并列规则，所选四权重实际字节SHA与SELECTION、参考报告一致。
- 未独立重验全逐seed/视觉/bootstrap原始证据，未反序列化checkpoint或做本地推理；人工质量审核未进行。病灶外背景保持的评分事务不等于完整生成分布质量，不称临床可用性、分割增益或外部泛化通过。test104与新R2 test130未读图评分。

## 证据与存储

配套64份轻量文件共24267993bytes：原始SELECTION/COMPLETE、40候选完整摘要、参考REPORT、训练/评价配置与冻结manifest、四路RUN_CONTRACT/完成标记等，保留原字节。入口：[轻量结果](../operator/pull_best_20261021T043517Z/lightweight)、[原始选模JSON](../operator/pull_best_20261021T043517Z/lightweight/evaluation/selection/SELECTION.json)、[下载验收](../operator/pull_best_20261021T043517Z/PULL_ACCEPTANCE.json)、[本地二次复核](../operator/pull_best_20261021T043517Z/LOCAL_VERIFICATION.json)。未下载MRI、参考NPZ或其他候选模型。

远端库存UTC2026-10-21T04:24:31.978733/本地收到04:35:49.637313；传输后远端四模型重新计算SHA完成UTC04:27:15.153671/本地收到04:38:25.814470；本地验收04:38:56.151576、二次验收04:38:56.287090分别保存。两端时钟不混用。

传输后账号目录总3TiB，可用1261167443968bytes约1.147025TiB，占用62%；本地二次验收时可用171301937152bytes约159.537GiB。系统可用量会被其他活动改变，不用下载前后差值代表模型大小；未清理、迁移或删除服务器数据，未重新查quota上限。只读核验与四次scp下载，本次自建VPNfinally关闭；没有sbatch/release/改作业/自动retry/恢复监控/安装或Git操作。

## 与最终200000模型区分

旧[final_models_200000](../../20261016_vae_epoch004_1mm_resume_uniform_r9/final_models_200000/README.md)保存四路训练最终checkpoint，全部保留不改。T2W/T2F恰好选中200000，和旧最终文件SHA一致；T1C最佳120000、T1N最佳40000与旧最终200000不同。需要本批所选模型时以本目录和MODEL_MANIFEST为准，不使用latest或一律200000代替选模结果。

这些完整checkpoint由冻结训练runtime保存，包括网络、optimizer、scaler、RNG/MONAI、sampler/permutation/cursor、data_identity/training_config/schedule；本次按SHA识别未加载。推理只消费网络，不把120k/40k当作可替代最终200k继续训练的自动恢复指令。后续加载须使用项目兼容环境与可信checkpoint，不临时安装依赖或混用系统/Homebrew Python。

artifact_status=experimental_unvalidated；operator_approved=false；formal_gate_status=not_run_not_passed。
