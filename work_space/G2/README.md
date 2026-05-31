# G2 工作区

## 整体任务

详见 [task_assignment.md](task_assignment.md)。

G2 当前负责 Task 1 的数据生成、合成数据质量控制、数据增强实验设计与交付物整理。现阶段方案已经从“直接生成整例 MRI”收窄为“优先做真实病例背景上的局部病灶条件扩散生成/插入”，先把数据工程、质量控制、nnU-Net 数据转换、训练拆分和消融评估框架建好，等 G1 生成模型稳定后再接入合成样本。

## 当前进度

1. 已补充 G2 数据生成与质量控制主方案，明确扩散模型替代 GAN 的可行路径、阶段边界、输入输出契约、质量控制指标和消融评估方式。
2. 已新增“模型训练完成前可执行工作清单”，把不依赖 G1 模型权重的任务拆成可立即执行的数据盘点、manifest、标签检查、统计分析、nnU-Net 转换、拆分协议和报告模板。
3. 已将合成数据质量判断标准调整为“最终能否提升真实验证集分割效果”，FID、MS-SSIM 等生成质量指标只作为辅助，不作为主要决策依据。
4. 已明确第一阶段优先路线：局部 lesion-conditioned inpainting/insertion，保留真实脑背景，避免一开始承担整例多模态 MRI 生成的过高风险。
5. 已完成本机可执行的训练前数据准备，结果保存到 [results/](results/)；GPU 训练、在线 batch 生成、nnU-Net 预处理和大规模 synthetic NIfTI 生成暂缓到训练机执行。

## 本周计划

1. 盘点真实训练集、验证集、修正标签和路径结构，生成病例级 manifest。
2. 完成标签合法性检查、修正标签覆盖策略和病灶体积/类别统计。
3. 建立 real-only nnU-Net 数据目录、训练/验证拆分和基线实验配置。
4. 和 G1 对齐生成模型输出契约，明确每个合成样本必须携带的影像、标签、来源、生成参数和 QC 元数据。
5. 搭建合成样本 QC 表、人工抽查模板、消融实验记录表和阶段报告模板。

## 相关资料

- [docs/G2_数据生成与质量控制实施方案.md](docs/G2_数据生成与质量控制实施方案.md)
- [docs/G2_数据生成与质量控制实施方案.pdf](docs/G2_数据生成与质量控制实施方案.pdf)
- [docs/G1_G2_diffusion_output_contract.md](docs/G1_G2_diffusion_output_contract.md)
- [docs/G2_模型训练完成前可执行工作清单.md](docs/G2_模型训练完成前可执行工作清单.md)
- [docs/G2_数据生成与质量控制实现难度评估.md](docs/G2_数据生成与质量控制实现难度评估.md)
- [results/reports/G2_训练前数据准备与G1方案对接总结.md](results/reports/G2_训练前数据准备与G1方案对接总结.md)
- [results/reports/G2_训练前数据准备与G1方案对接总结.docx](results/reports/G2_训练前数据准备与G1方案对接总结.docx)
- [../../data_space/task1_2026/BraTS2026_Task1_中文.md](../../data_space/task1_2026/BraTS2026_Task1_中文.md)
- [../../data_space/task1_2026/datasets/数据集现状.md](../../data_space/task1_2026/datasets/数据集现状.md)
- [../../data_space/task1_2026/datasets/数据契约.md](../../data_space/task1_2026/datasets/数据契约.md)

## 提交记录

| 日期 | 内容 | 备注 |
| --- | --- | --- |
| 2026-05-31 | 完成 G2 训练前数据准备、结果产物、G1/S1/S2 总结 DOCX 与时间线更新 | 文档与小型 CSV/JSON 产物更新，尚未提交 |
