# SegMamba BraTS 2026 实验指南

本项目包含 8 组实验，用于验证不同配置对脑转移瘤分割模型性能的影响。

---

## 目录

- [实验列表](#实验列表)
- [快速开始](#快速开始)
- [脚本说明](#脚本说明)
- [实验配置说明](#实验配置说明)

---

## 实验列表

| 实验 | 配置 | 说明 |
|------|------|------|
| 实验1 | `exp_ce_dice.yaml` | CE + Dice Loss（标准做法） |
| 实验2 | `exp_dice_focal.yaml` | Dice + Focal Loss（适用于小病灶多的情况） |
| 实验3 | `exp_ce_focal.yaml` | CE + Focal Loss（适用于类别极度不平衡） |
| 实验4 | `exp1_larger_patch.yaml` | 更大 Patch Size - 192³ |
| 实验5 | `exp2_adamw_cosine.yaml` | AdamW + Cosine 学习率调度 |
| 实验6 | `exp3_deeper_model.yaml` | 更深模型 depths [3,3,6,6] |
| 实验7 | `exp4_no_augmentation.yaml` | 无数据增强（消融实验） |
| 实验8 | `exp5_high_overlap_infer.yaml` | 高重叠推理（只影响推理，不用重新训练） |

---

## 快速开始

### 运行全部实验

```bash
# 一键运行所有实验（训练 + 预测 + 指标计算）
./run_all_experiments.sh

# 只运行指定实验
./run_all_experiments.sh exp_ce_dice
```

---

## 脚本说明

| 脚本 | 功能 | 主要参数 |
|------|------|----------|
| `run_all_experiments.sh` | 运行全部实验 | `--skip-train` 跳过训练, `--log-dir` 指定日志目录 |
| `train.sh` | 训练模型 | 实验名或 `--config` 配置文件 |
| `predict.sh` | 生成预测 | 实验名或 `--config` 配置文件 |
| `compute_metrics.sh` | 计算指标 | 实验名或 `--config` 配置文件 |

### 高级用法

```bash
# 覆盖配置参数
./train.sh exp_ce_dice --max_epoch 50 --batch_size 4

# 指定日志目录
./run_all_experiments.sh --log-dir ./my_custom_logs
```

---

## 实验配置说明

配置文件位于 `configs/` 目录下，每个实验可调整的参数包括：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data_dir` | 数据目录 | `./data/fullres/train` |
| `device` | 训练设备 | `cuda:0` |
| `roi_size` | Patch 大小 | `[128, 128, 128]` |
| `max_epoch` | 最大训练轮数 | `100` |
| `batch_size` | 批大小 | `2` |
| `learning_rate` | 学习率 | `0.0001` |
| `loss_type` | 损失函数类型 | `ce_dice` |

### 损失函数配置

```yaml
# CE + Dice (实验1)
loss_type: ce_dice

# Dice + Focal (实验2)
loss_type: dice_focal

# CE + Focal (实验3)
loss_type: ce_focal
```

### 推理配置

```yaml
# exp5_high_overlap_infer.yaml 中的特殊配置
inference:
  overlap: 0.75  # 默认 0.5，提高到 0.75
```

---

## 输出目录结构

```
.
├── logs/
│   ├── segmamba_<实验名>/
│   │   └── model/
│   │       └── best_model_<实验名>_<epoch>.pt
│   └── experiments/           # run_all_experiments.sh 的日志
│       └── <实验名>/
│           ├── train.log
│           ├── predict.log
│           └── metrics.log
├── prediction_results/
│   ├── segmamba_brats2026_<实验名>/
│   │   ├── case_001.nii.gz
│   │   ├── case_002.nii.gz
│   │   └── ...
│   └── result_metrics_<实验名>/
│       ├── segmamba_brats2026_<实验名>_segmentation.npy
│       └── segmamba_brats2026_<实验名>_detection.npy
```

---

## 评估指标说明

### 分割指标
- **DSC (Dice Similarity Coefficient)**: 分割重叠度，值越接近 1 越好
- **HD95**: 95% Hausdorff 距离，值越小越好
- **NSD (Normalized Surface Distance)**: 归一化表面距离，值越接近 1 越好

### 检测指标
- **F1**: 病灶检测 F1 分数（在不同 IoU 阈值下）
- **AUC**: F1 曲线的 AUC 值，综合评估检测性能


