# G1 工作区

## 整体任务

详见 [task_assignment.md](task_assignment.md)。

## 当前进度

G1 当前保留两条正式产线：

| 产线 | 正式代码 | 状态 |
|---|---|---|
| 缺失 T2W 填补 V3 | `code/brats2025-latent-ensemble-synthesis-main-v3` | 服务器阶段任务运行中 |
| Diffusion augmentation V3 | `code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN` | Slurm/手册已改为 NYU Greene 入口（`*_v3_nyu.slurm`） |

两条线不可互换。缺失 T2W 线修复原病例；Diffusion augmentation 线生成新增 synthetic 病例，并由 G2 composer/QC 接收。

## 本周计划

1. 在 NYU Greene 完成 Diffusion augmentation 四模态训练与固定 val 评估。
2. 锁定四个 checkpoint、采样参数和生成 metadata。
3. 批量生成后交给 G2 composer 与 full-generation QC。

操作入口：

- [Diffusion 服务器手册（NYU）](docs/G1_diffusion_augmentation服务器训练手册.md)
- [G1-G2 总运行手册](docs/G1_G2_服务器训练推理QC运行手册.md)
- [slurm/README.md](code/BraTS_2023_2024_solutions-main%203/Segmentation_Tasks/GliGAN/slurm/README.md)
- [历史 ECNU 部署记录](docs/G1_Diffusion_V3_ECNU部署记录_2026-07-15.md)

## 提交记录

| 日期 | 内容 | 备注 |
| --- | --- | --- |
| 2026-07-15 | Diffusion V3 Slurm/手册切换为 NYU Greene（对齐 main2 `*_v2_nyu.slurm` 风格） | 入口：`slurm/*_v3_nyu.slurm` |
| 2026-07-15 | Diffusion augmentation V3 真实 GPU 回归、数据契约修复（ECNU 历史验证） | 详见历史 ECNU 部署记录 |
