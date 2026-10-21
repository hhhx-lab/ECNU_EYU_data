# Epoch004 重建图 1mm Diffusion ECNU 独立重提 R2

用户在旧 SegMamba/S5 清理验收后明确授权“现在空间充裕，进行重提！开始进行扩散模型训练”。本批次只修正 P0 轻量证据容量合同，原 R1 方案、失败目录、运行包和审计均保留；活动用户源码不改。

- 原方案 SHA：`fe538efa6c4d1c2fae4aed2003270a5dd785366061e8ddede0d71d17c87c5353`。
- P0 独立验收全部 1030 例、5150 NIfTI 的 SHA/finite/原生几何和源标签一致性。P0 只写轻量证据，预留 1 GiB，不写重建影像或训练 checkpoint。
- P1 仅处理 train823/val103；其 1mm 数据、缓存及四路 checkpoint 容量须另算，不能复用 P0 的 1 GiB 门。
- P2 四独立 A100 smoke、精确恢复和四模态生成自动/真实人工质量门通过后才可 P3。四模态各 200000 成功优化更新，不是 175 epoch，不复用 S2 权重。
- locked104 不评分；不拉大数据、不上传 Drive、不部署 S2、不安装依赖。

状态以 `operator/*/SUBMISSION_RECEIPT.json`、远端 `evidence/p0_native/VALIDATION.json` 和独立快照为准。P0 提交不等于训练已经开始。

所有新结果：`artifact_status=experimental_unvalidated`、`operator_approved=false`、`formal_gate_status=not_run_not_passed`。工程通过不证明 VAE 重建图或 1mm Diffusion 更优。
