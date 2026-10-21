# S2 Dataset264 Completion-Only nnU-Net Result Archive

归档时间：2026-07-20

远端来源：

- Host: `root@117.50.177.229:23`
- S2 root: `/cloud/cloud-ssd1/brats2026/s2`
- Fold result: `/cloud/cloud-ssd1/brats2026/s2/nnUNet_results/Dataset264_BraTS2026_MET_Completion/nnUNetTrainerBraTS2026RCCompletionFineTune__nnUNetPlans__3d_fullres/fold_0`

本地内容：

- `fold_0/checkpoint_final.pth`: S2 completion-only 最终 checkpoint。
- `fold_0/checkpoint_best.pth`: 本轮 best checkpoint。
- `fold_0/training_log_2026_7_19_09_02_20.txt`: 主训练日志。
- `fold_0/progress.png`, `fold_0/debug.json`: nnU-Net 训练进度与调试配置。
- `fold_0/validation/`: 103 个 fixed-split validation 预测，以及 `summary.json`。
- `logs/`: 缓存重建与云端训练 launcher 日志、PID 文件快照。
- `metadata/`: 完成门、warm-start 预处理门、`nnUNetPlans.json`、`dataset.json`、`completion_plans_audit.json`。
- `checksums/SHA256SUMS.txt`: 本归档实际结果文件的 SHA256 清单，不包含 checksum 文件自身。

完成状态：

- 训练完成到 `Epoch 199` 后执行 validation。
- 训练日志记录 `Mean Validation Dice: 0.5042405801751146`。
- launcher 日志记录 `Fixed-split training complete` 和 `S2_CLOUD_TRAIN_PASS`。
- 远端计划审计记录 plans 来自 Dataset263 baseline checkpoint 的 `init_args.plans`，目标 Dataset264 配置为 spacing `[1.0, 1.0, 1.0]`、patch size `[128, 128, 128]`。

本次未拉取：

- `nnUNet_preprocessed` 完整缓存，远端约 109 GB，本地只保存可复核元数据和完成门。
- `nnUNet_raw` 原始 4552 imagesTr / 1138 labelsTr。
- T2W Diffusion 权重；本目录只归档 S2 completion-only nnU-Net 结果。

已核对的远端/本地 SHA256：

- `checkpoint_final.pth`: `78eccc59f9217a529cafdd522733de9a1578f0e96d8765ee7c48731027824db5`
- `checkpoint_best.pth`: `6453a8a747abd5f838fff352733355d91bf8018cc733117b7503db88d717a68a`
- `validation/summary.json`: `33ae1eb0fb513178e00e0867ea7cc5bf1e7d07a8ca91b18995b699f5ebd3c103`
- `completion_plans_audit.json`: `5e7f136d5ffb433f473a53c932d75c1b751d513378ddaf4f5837418e1bc04ebc`
