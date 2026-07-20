# S2 工作区

S2 使用 nnU-Net v2 `3d_fullres` 和 RC-aware Trainer 训练 BraTS-MET 分割模型。

## 当前正式策略

- 不使用交叉验证，只训练一个固定模型。
- 默认读取 G2 patient-group real-only split：`823 train / 103 val / 104 test`。
- 默认使用 `Dataset263_BraTS2026_MET_RealOnly_Current`，与历史 Dataset260 隔离。
- validation 用于模型选择；locked test 只用于最终复核。
- 后续 real+synth 必须复用相同真实 validation/test，才能进行 paired 消融。
- nnU-Net 的 `fold_0` 只是 API/目录键，不代表五折。
- 官方无标签 validation 是另外 179 例，不属于上面的 103 例内部 validation；它只用于训练完成后的 Synapse 推理提交。

## Completion-only 增量实验

G1 V3 补全并经 G2 技术 QC 的 212 例 train completion 与 823 例真实 train 合并为 Dataset264；103 例真实 validation 和 104 例 locked test 保持不变。该实验从 Dataset263 real-only `checkpoint_final.pth` warm-start，不使用尚未完成的 Diffusion 在线增强。

ECNU 集群上 Dataset264 的 raw 链接目录、preprocessed 缓存和 results 必须放在 `/hpc_stor`；Slurm 会拒绝回退到使用率已达 99% 的 `/public`。运行入口：

```text
work_space/S2/slurm/01_prepare_completion_ecnu.slurm
work_space/S2/slurm/02_train_completion_v100_ecnu.slurm
```

赛期紧急且 `/hpc_stor` 权限尚未就绪时，可由操作者显式设置 `S2_ALLOW_PUBLIC_EMERGENCY=1`，仅允许写入独立的 `work_space/S2/data/ecnu_completion_emergency/`。该模式保留至少 1TB `/public` 空间并每 5 分钟检查；训练和内部验证成功后，自动删除可再生的 raw 链接和大体积预处理数组，保留 checkpoint、plans、真值与验证结果。

历史 Dataset260 `828/207/259` 模型已完成 1000 epochs 和 207 例验证推理，但只能作为历史结果，不用于当前严格消融。

服务器运行说明：

```text
work_space/S2/BraTS2026_S2_RC_v1.0/repository/docs/S2_服务器运行手册.md
```

官方 179 例推理会自动读取 `work_space/G1/data/raw/Validation/`，生成预测并打包为：

```text
work_space/S2/results/realonly_current_official_validation_submission/
S2_realonly_current_Task1_validation_179.zip
```
