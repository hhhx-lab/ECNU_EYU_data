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
