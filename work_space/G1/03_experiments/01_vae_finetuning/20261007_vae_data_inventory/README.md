# VAE epoch4与Diffusion重训数据核对

复核日期：2026-10-07，北京时间。`artifact_status=experimental_unvalidated`，`operator_approved=false`，`formal_gate_status=not_run_not_passed`。

本地证据落盘UTC：2026-10-07T07:13:08.402622+00:00；对应远端UTC：2026-10-07T07:02:04.887363+00:00，hostname=login1。两次只读操作各自显式启动VPN，结束关闭本次VPN或保留既有VPN；无sleep、调度器轮询、新训练、编码或评估作业。

## epoch4身份

用户记得的epoch4确实存在：远端 `g1_vae_extended_finetune_20260901_r3_attempt03` 训练配置/历史完成20轮，训练job记录3577166。`TRAINING_JOB_VALIDATION.json` 和 `EPOCH_SELECTION_AUDIT.json` 为pass，选择epoch4，训练完成标记存在。本轮未重新读取该历史作业sacct。

该延长实验从预训练 `autoencoder_epoch273.pt` 初始化，不是从历史3轮checkpoint续到累计20轮。固定quick-val20例，选点同时考察T2W和条件模态安全约束，epoch1-5为合格候选，最终选择epoch4。不是无约束全指标最大值，也不是fixed103验证后的正式优越性结论。

完整checkpoint路径：

```text
/public/home/zqchen/projects/ECNU_EYU_data/work_space/G1/results/experimental_unvalidated/g1_vae_extended_finetune_20260901_r3_attempt03/training/checkpoints/checkpoint_epoch_004.pt
```

大小251518356 bytes，SHA256 `5d86a0b3cbc5df34d64d5a0ff376eae453634a609bd24dec1a1191fde800b13b`，本轮远端只读重算一致。没有下载checkpoint。

## quick-val原值

以下是quick-val20模型空间均值，epoch4原值由同一次baseline加选点审计delta复算，8位小数。不能混用历史run3034060的完整103例均值。

| 模态 | 指标 | 预训练VAE | epoch4 VAE | epoch4减预训练 |
|---|---|---:|---:|---:|
| T1N | tumor SSIM | 0.69375971 | 0.72198040 | +0.02822069 |
| T1C | tumor SSIM | 0.33383937 | 0.41242830 | +0.07858894 |
| T2F | tumor SSIM | 0.67562201 | 0.69905795 | +0.02343594 |
| T2W | tumor SSIM | 0.66415458 | 0.71073725 | +0.04658267 |
| T1N | tumor MSE | 0.00240313 | 0.00114696 | -0.00125617 |
| T1C | tumor MSE | 0.01869180 | 0.01147510 | -0.00721671 |
| T2F | tumor MSE | 0.00208481 | 0.00108695 | -0.00099785 |
| T2W | tumor MSE | 0.00291130 | 0.00075320 | -0.00215810 |
| T1N | whole SSIM | 0.21311822 | 0.21141106 | -0.00170716 |
| T1C | whole SSIM | 0.13674838 | 0.14945644 | +0.01270806 |
| T2F | whole SSIM | 0.20643137 | 0.20614510 | -0.00028626 |
| T2W | whole SSIM | 0.20872038 | 0.21015879 | +0.00143841 |

## 后续验证与数据缺口

threeway fixed103验证所查批次均有 `STAGE1_FAILED.json`：20260910 r1 job3671857退出127，20260911 attempt02 job3679388退出1，20260914 attempt03 job3681139退出1。attempt03 stderr为 `RuntimeError: empty tumor in fixed103: BraTS-MET-01117-003`。这是原验证实现的拒绝证据，不授权自动跳过病例；未改写失败或创建恢复attempt。所查目录未见完整全量验证成功报告。

旧G1 V3代码 `data/selected_vae.json` 仍绑定历史 `run_3034060/best_model.pt`。4120个latent共10800860160 bytes，1030例split为823/103/104，不能改名冒充epoch4数据。mask包含多种分辨率，本轮粗计数不能替代逐后缀/逐病例空间验收；空间JSON通配未命中不等于空间信息不存在。对G1/data及G1/results/experimental_unvalidated限定深度盘点未发现已完成的epoch4全量latent仓库，并非穷尽远端所有磁盘。

## 用户确认重训四模态病灶Diffusion

这条线使用真实四模态NIfTI + 原始/纠正seg + patient-group split + 病灶索引，训练时zscore与64³ crop。`src/utils/data_utils.py`直接LoadImaged读取MRI/label，然后执行病灶crop；不经过VAE。

原Diffusion本地清单：

- [dataset_summary.json](../../03_diffusion_training/20260720_v3_multiserver/canonical/splits/current/dataset_summary.json)：926例，train823/val103，原DataSet为symlink_only。
- [dataset_membership.csv](../../03_diffusion_training/20260720_v3_multiserver/canonical/splits/current/dataset_membership.csv)：926行；SHA256 `7844e222f83162faf626b161a3e0bf4c739e363dbbbedc958790702486669e85`。
- [lesions.csv](../../03_diffusion_training/20260720_v3_multiserver/canonical/splits/current/lesions.csv)：7403行，train6669/val734；SHA256 `80031cf4d3612cc77eef13a44dcaaf257394aecb2cbbbfa2fa1f18f191ecd859`。

按原方案重训，应拉回926例四模态及正确seg，共4630个影像/标签文件，配套split/病灶清单/归一化合同。locked104不加入训练，265例无真实T2W者不作真实四模态监督源。先核验远端存在性、纠正标签、总大小、本地空间，再冻结逐文件SHA传输；旧清单云服务器路径不能原位改写，须建立新映射。本地目前约161 GiB空闲，不推定足以容纳原始全集。

若坚持用epoch4 VAE的**重建影像**训练Diffusion，这是新的研究分支：先补完整VAE验证，独立生成图像空间数据，保持标签/geometry并验收边界/小病灶，与原真实影像路线分表比较。latent不能直接送入现有像素空间Diffusion；重建可能平滑细节或改变分布，不等于原始影像质量升级。本次未执行或授权新分支，也未启动大体积下载。

## 证据入口

- [初始只读盘点](operator/readonly_snapshot_20261007T071109Z/REMOTE_VAE_DATA_SNAPSHOT.json)。
- [逐文件远端SHA索引](operator/selected_evidence_pull_20261007T071253Z/REMOTE_EVIDENCE_INDEX.json)。
- [拉回回执](operator/selected_evidence_pull_20261007T071253Z/LOCAL_OPERATION_RECEIPT.json)：96份轻量文件SHA/JSON校验pass。
- [原件镜像](operator/selected_evidence_pull_20261007T071253Z/pulled)：计划/配置/训练历史/选点/完成及失败证据/日志。
- [只读证据入口](operator/pull_selected_vae_evidence.py)：复用现有Access类、精确env、独立VPN生命周期与有限SSH，不deploy、不sbatch、不清理源文件。
