# S2 工作区

S2 使用 nnU-Net v2 `3d_fullres` 和 RC-aware Trainer 训练 BraTS-MET 分割模型。

## 当前正式策略

- 不使用交叉验证，只训练一个固定模型。
- 默认读取 G2 patient-group real-only split：`823 train / 103 val / 104 test`。
- 默认使用 `Dataset263_BraTS2026_MET_RealOnly_Current`，与历史 Dataset260 隔离。
- validation 用于模型选择；locked test 只用于最终复核。
- 后续 real+synth 必须复用相同真实 validation/test，才能进行 paired 消融。
- nnU-Net 的 `fold_0` 只是 API/目录键，不代表五折。

历史 Dataset260 `828/207/259` 模型已完成 1000 epochs 和 207 例验证推理，但只能作为历史结果，不用于当前严格消融。

服务器运行说明：

```text
work_space/S2/BraTS2026_S2_RC_v1.0/repository/docs/S2_服务器运行手册.md
```
