# G2 数据生成与质量控制

## 角色

G2 接收 G1 生成结果，负责数据身份、标准化、质量控制、训练数据发布和增强价值报告。任务依据见 [task_assignment.md](task_assignment.md)。

当前两条 G1 产线必须分开处理：

| 产线 | 数据语义 | G2 处理 |
|---|---|---|
| Diffusion V2 | segmentation-conditioned augmentation | 先 composition，再作为新增 synthetic train case |
| Missing-T2W V3 | 缺失模态修复 | 保留真实病例 ID，只替换对应 T2W |

> 版本说明（2026-07-15）：表中的 `Diffusion V2` 是 G2 augmentation 接口名。当前 G1 生产模型代码已升级为 `work_space/G1/code/BraTS_2023_2024_solutions-main 3/Segmentation_Tasks/GliGAN`；G2 仍使用 `g2_v2_compose_augmentation.py` 接收其输出，不代表继续运行旧 `main 2` 模型。

## 唯一数据口径

| 文件 | 用途 |
|---|---|
| `results/manifests/nnunet_case_mapping_master.csv` | 全部 1295 例身份，包括 265 个 completion 目标 |
| `results/manifests/nnunet_case_mapping_realonly.csv` | 1030 个真实 T2W 病例，供 real-only baseline |
| `results/manifests/g1_v2_source_manifest.csv` | V2 source 表，只有 master train 且真实 T2W 的 823 例可生成 |
| `results/splits/splits_master_train_val_test.json` | 全部病例 patient-group split：1035/130/130 |
| `results/splits/splits_final_train_val_test.json` | real-only 派生 split：823/103/104，与 G1 V3 一致 |

同一 `BraTS-MET-xxxxx` patient group 不会跨 train、val、test。

## 正式入口

```text
code/g2_build_realonly_from_raw.py
code/g2_create_train_val_test_split.py
code/g2_v2_compose_augmentation.py
code/g2_v3_completion_intake.py
code/g2_v3_paired_quality.py
code/g2_s2_v3_teacher_eval.py
code/g2_synthetic_raw_intake_qc.py
code/g2_materialize_nnunet_dataset.py
code/g2_official_mets_metrics_parser.py
```

G1 V3 阶段 5 审批使用两个独立入口：

```text
slurm/01_g2_v3_paired_quality.slurm  # 真实/生成 T2W 影像与病灶 QC
slurm/02_g2_v3_s2_teacher.slurm      # 冻结 S2 对同一 103 例做成对 teacher 推理
```

teacher 输入固定为真实 `t1n/t1c/t2f` 加生成 `t2w`，checkpoint 固定为当前
Dataset263 real-only 模型。teacher 技术通过仍不等于自动批准阶段 6，还必须复核 paired-QC montage。
paired QC 同时强制读取同一 Stage 5 run 的 `spatial_audit.csv`；病例 ID
不一致、字段缺失或任一 foreground/lesion 体素逃出模型 FOV 都直接拒绝。

执行顺序见 [G2_G1适配执行清单.md](docs/G2_G1适配执行清单.md)，接口字段见 [G1_G2_diffusion_output_contract.md](docs/G1_G2_diffusion_output_contract.md)，本轮整改记录见 [2026-07-12_G2_V2_V3对接与QC整改.md](docs/2026-07-12_G2_V2_V3对接与QC整改.md)。

## Release 状态

| 状态 | 含义 |
|---|---|
| `rejected` | 技术硬门失败 |
| `pending_review` | 技术通过，等待 teacher/人工批准 |
| `accepted_for_training` | 可进入 train |
| `accepted_for_evaluation` | V3 val/test completion，仅用于固定评估 |

没有审批记录的病例不会自动进入训练；manifest/log 缺少逐病例唯一成功记录时会直接 rejected。

## 通道与标签

```text
0000=t1n
0001=t1c
0002=t2w
0003=t2f
labels={0,1,2,3,4}
```

这是 S1-S5 当前统一口径，不再保留旧通道兼容选项。

## 验证

```bash
python -m unittest discover -s work_space/G2/tests -v
```

当前 G2 共 49 项测试，使用临时小型 NIfTI，不依赖本机保存 40GB 数据，也不向仓库写入临时影像。
