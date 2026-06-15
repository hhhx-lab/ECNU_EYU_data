# G2 工作区

## 整体任务

详见 [task_assignment.md](task_assignment.md)。

G2 当前负责 Task 1 的数据生成、G1 raw output 接收、合成数据质量控制、数据增强实验设计与交付物整理。当前主口径已经调整为：**G2 完全适配 G1 的 GliGAN-compatible diffusion 输出**，不要求 G1 第一阶段直接输出理想 `SYN-MET-*` 资产；G1 负责生成局部 ROI raw synthetic case，G2 负责标准化、记忆化、QC、nnU-Net 导出和数据报告。

## 当前进度

1. 已将 G2 主方案重写为 G1-first adapter 版：G2 主动接收 G1 legacy raw output，再补齐 manifest、QC、final ID 和 nnU-Net 导出。
2. 已将 G1-G2 契约重写为 raw output 对接契约，明确 source CSV、legacy suffix、label channel、run config、generation log、manifest 和 smoke test 标准。
3. 已将 synthetic QC 策略重写为 G1 raw output 适配版，覆盖 raw intake、文件完整性、NIfTI 几何、label 合法性、source 泄漏、ROI 插入一致性、扩散生成质量、多模态一致性、teacher、批次分布、nnU-Net integrity 和真实验证 fold 消融。
4. 已新增 `G2_G1适配执行清单.md`，把 G2 接收 G1 输出后的动作拆成可执行步骤。
5. 已补充 `g2_synthetic_raw_intake_qc.py`、`g2_materialize_nnunet_dataset.py`、`g2_official_mets_metrics_parser.py` 三个工程入口。
6. 已补充 `diffusion_quality_metrics_template.csv` 和 `synthetic_generation_manifest_template_g1.csv`，用于后续数据报告和 G1 smoke run 接收。
7. 已完成本机可执行的训练前数据准备，结果保存到 [results/](results/)；GPU 训练、在线 batch 生成、nnU-Net 预处理和大规模 synthetic NIfTI 生成暂缓到训练机执行。

## 本周计划

1. 盘点真实训练集、验证集、修正标签和路径结构，生成病例级 manifest。
2. 完成标签合法性检查、修正标签覆盖策略和病灶体积/类别统计。
3. 建立 real-only nnU-Net 数据目录、训练/验证拆分和基线实验配置。
4. 使用 G1 smoke raw output 跑通 G2 intake、legacy suffix 映射、manifest 补建和 QC。
5. 根据 rejected/needs_regeneration 表向 G1 回传下一轮模型或推理参数调整建议。

## 相关资料

- [docs/G2_数据生成与质量控制实施方案.md](docs/G2_数据生成与质量控制实施方案.md)
- [docs/G1_G2_diffusion_output_contract.md](docs/G1_G2_diffusion_output_contract.md)
- [docs/G2_G1适配执行清单.md](docs/G2_G1适配执行清单.md)
- [docs/G2_模型训练完成前可执行工作清单.md](docs/G2_模型训练完成前可执行工作清单.md)
- [results/qc/G2_synthetic_data_QC规则策略_v2.md](results/qc/G2_synthetic_data_QC规则策略_v2.md)
- [results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md](results/qc/G2_official_metrics_alignment_QC_strategy_2026-06-15.md)
- [results/qc/diffusion_quality_metrics_template.csv](results/qc/diffusion_quality_metrics_template.csv)
- [results/manifests/synthetic_generation_manifest_template_g1.csv](results/manifests/synthetic_generation_manifest_template_g1.csv)
- [code/g2_synthetic_raw_intake_qc.py](code/g2_synthetic_raw_intake_qc.py)
- [code/g2_materialize_nnunet_dataset.py](code/g2_materialize_nnunet_dataset.py)
- [code/g2_official_mets_metrics_parser.py](code/g2_official_mets_metrics_parser.py)
- [../../data_space/task1_2026/BraTS2026_Task1_中文.md](../../data_space/task1_2026/BraTS2026_Task1_中文.md)
- [../../data_space/task1_2026/datasets/数据集现状.md](../../data_space/task1_2026/datasets/数据集现状.md)
- [../../data_space/task1_2026/datasets/数据契约.md](../../data_space/task1_2026/datasets/数据契约.md)

## 提交记录

| 日期 | 内容 | 备注 |
| --- | --- | --- |
| 2026-06-13 | G2 主口径改为完全适配 G1 raw output，重写主方案、对接契约、QC 策略和扩散质量模板 | 尚未提交 |
| 2026-05-31 | 完成 G2 训练前数据准备、结果产物、G1/S1/S2 总结 DOCX 与时间线更新 | 文档与小型 CSV/JSON 产物更新 |
