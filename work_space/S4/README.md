# SAM2-UNet BraTS-MET 3D 分割训练说明

本文档说明 `work_space/S4/code` 中 `sam2unet_model.py`、`train_sam2unet_post.py` 和 `slurm/train_sam2unet_post.slurm` 的代码构成、作用，以及如何在服务器/Slurm 集群上启动训练、配置参数和检查结果。

当前项目推荐服务器根目录为 `/scratch/bf2260/ECNU_EYU_data`。S4 代码目录是：

```text
/scratch/bf2260/ECNU_EYU_data/work_space/S4/code
```

S4 现在支持两种模式：正式实验通过 `SAM2UNET_FIXED_SPLIT_ROOT` 读取 G2 `train/val/test` case-folder view；旧的 RC-stratified 自动划分只保留给独立探索 smoke。与 S1-S5 做严格比较时必须使用 G2 fixed split。

## 1. 项目简介

本项目用于 BraTS-MET 3D 医学图像分割。输入是每个病例的 4 个 MRI 模态：

- `t1n`
- `t1c`
- `t2w`
- `t2f`

标签使用 5 类整数：

| 标签 | 含义 |
| --- | --- |
| `0` | background |
| `1` | NETC |
| `2` | SNFH |
| `3` | ET |
| `4` | RC |

模型不是直接用一个普通 5 通道 head 做分类，而是拆成两个任务：

- 主任务 head：预测 `BG/NETC/SNFH/ET`，输出 `main_logits`，形状为 `(B, 4, D, H, W)`。
- RC 二分类 head：预测 `RC vs non-RC`，输出 `rc_logit`，形状为 `(B, 1, D, H, W)`。

推理时先对 `main_logits` 做 `argmax` 得到 `0..3` 的主任务预测，再对 `rc_logit` 做 `sigmoid`。当 RC 概率大于阈值 `rc_threshold` 时，该体素被覆盖为标签 `4`，最终输出仍是 `0..4` 的单个 label map。

## 2. 目录结构

```text
work_space/S4/code
├── sam2unet_model.py                 # SAM2-UNet 3D 模型定义
├── train_sam2unet_post.py            # 训练、验证、保存结果的主脚本
├── slurm/
│   └── train_sam2unet_post.slurm     # Slurm 提交脚本
└── README.md                         # 当前说明文档
```

每个病例目录应放在服务器上的训练数据根目录下。real-only smoke test 可先使用：

```text
/scratch/bf2260/ECNU_EYU_data/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training
```

一个完整病例目录示例：

```text
TrainingData/
└── BraTS-MET-00001-000/
    ├── BraTS-MET-00001-000-t1n.nii.gz
    ├── BraTS-MET-00001-000-t1c.nii.gz
    ├── BraTS-MET-00001-000-t2w.nii.gz
    ├── BraTS-MET-00001-000-t2f.nii.gz
    └── BraTS-MET-00001-000-seg.nii.gz
```

训练脚本会跳过缺文件的病例目录。若分割标签中出现 `0..4` 以外的值，也会跳过该病例，并在输出目录写入 `skipped_invalid_label_cases.json` 和 `skipped_invalid_label_cases.csv`。

## 3. `sam2unet_model.py` 代码说明

`sam2unet_model.py` 负责定义 3D SAM2-UNet 模型。它把 SAM2/Hiera 风格的层级视觉 Transformer 思路改成适合 3D 体数据的医学图像分割网络。

### 3.1 标签和类别常量

文件开头定义了类别常量：

- `BRATS_MAIN_CLASS_NAMES = ("background", "NETC", "SNFH", "ET")`
- `BRATS_MAIN_NUM_CLASSES = 4`
- `BRATS_MET_CLASS_NAMES = ("background", "NETC", "SNFH", "ET", "RC")`
- `BRATS_MET_NUM_CLASSES = 5`
- `BRATS_RC_LABEL = 4`

这些常量保证模型和训练脚本都使用同一套标签含义。

### 3.2 `WindowedAttention3D`

`WindowedAttention3D` 是 3D 窗口多头自注意力模块。它的作用是：

1. 把体数据 `(D, H, W)` 按 `window_size` 切成多个不重叠的 3D window。
2. 在每个 window 内部做 self-attention。
3. 使用可学习的 3D relative position bias。
4. 把 window 结果重新合并回原来的 3D 空间。

这样做的好处是避免在整个体积上直接做全局 attention。全局 attention 的显存和计算量会随体素数快速增长，而 window attention 只在局部窗口内计算，更适合 3D MRI patch。

主要参数：

- `dim`：输入通道数。
- `num_heads`：attention head 数量。
- `window_size`：窗口大小，例如 `(4, 4, 4)`。
- `qkv_bias`：QKV 线性层是否使用 bias。
- `attn_drop`、`proj_drop`：attention 和输出投影 dropout。

### 3.3 `AttentionLayer3D`

`AttentionLayer3D` 是一个 Transformer block，输入输出形状都是 `(B, C, D, H, W)`。内部流程是：

1. 将 3D 特征展平为 token 序列 `(B, D*H*W, C)`。
2. `LayerNorm -> WindowedAttention3D -> residual`。
3. `LayerNorm -> MLP -> residual`。
4. 再 reshape 回 `(B, C, D, H, W)`。

这个模块替代了 U-Mamba 中类似长程建模的位置，使模型使用 windowed self-attention 学习空间上下文。

### 3.4 UNet 编码器和解码器模块

模型包含几个基础 building block：

- `ConvBlock3D`：两层 3D convolution，配合 normalization、GELU、dropout 和 residual skip。
- `DownBlock`：编码器块，结构是 `ConvBlock3D -> optional AttentionLayer3D -> stride=2 downsample`，同时返回 skip feature。
- `UpBlock`：解码器块，结构是 `ConvTranspose3d upsample -> concat skip -> ConvBlock3D -> optional AttentionLayer3D`。
- `Bottleneck`：最深层特征处理块，包含 `ConvBlock3D` 和 `AttentionLayer3D`。

`SAM2UNet3D` 通过这些模块形成标准 UNet 风格结构：

```text
4-channel MRI input
  -> stem
  -> encoder stages with skip connections
  -> bottleneck
  -> decoder stages
  -> shared head_features
  -> main_head + rc_head
```

默认 `depths=4`、`feature_size=48` 时，encoder 通道大致为 `48, 96, 192, 384`。

### 3.5 `SAM2UNet3D` 双 head 设计

`SAM2UNet3D` 是主模型类。关键输入参数：

- `spatial_size`：输入 patch 尺寸，例如 `(96, 96, 96)`。
- `in_channels`：输入模态数，当前固定为 `4`。
- `out_channels`：最终标签类别数，当前必须是 `5`。
- `feature_size`：基础通道数，默认 `48`。
- `depths`：encoder/decoder 层级数，默认 `4`。
- `num_heads`：attention head 基础数量。
- `window_size`：3D attention window 大小，默认 `(4, 4, 4)`。
- `dropout_rate`：dropout，默认 `0.2`。
- `use_attention`：是否启用 attention。
- `return_dict`：是否返回字典形式输出。

模型最终输出两个 head：

```python
{
    "main_logits": main_logits,  # (B, 4, D, H, W)
    "rc_logit": rc_logit,        # (B, 1, D, H, W)
}
```

其中 `rc_head` 不只使用 decoder feature，还会拼接 `softmax(main_logits)` 作为上下文：

```text
decoder feature -> head_features -> main_logits -> softmax(main_logits)
task_features + main_probs -> rc_head -> rc_logit
```

这样设计的目的，是让 RC head 可以参考 `BG/NETC/SNFH/ET` 的主任务预测结果，学习更依赖上下文的 RC 区域。

### 3.6 推理输出和兼容接口

`SAM2UNet3D.logits_to_label_map(main_logits, rc_logit, rc_threshold)` 会把双 head 输出转成最终标签图：

1. `main_pred = argmax(main_logits)`，得到 `0..3`。
2. `rc_prob = sigmoid(rc_logit)`。
3. 对 `rc_prob > rc_threshold` 的位置赋值为 `4`。
4. 返回 long tensor，标签范围为 `0..4`。

`predict_label_map(x, rc_threshold=0.3)` 会直接完成 forward 和 label map 转换。

`to_legacy_logits(outputs)` 会把 `main_logits` 和 `rc_logit` 拼接成 5 通道 tensor，用于兼容旧代码路径。但当前训练脚本使用的是双 head 字典输出。

### 3.7 `create_sam2unet()` 和自测入口

`create_sam2unet()` 是工厂函数，会创建 `SAM2UNet3D` 并打印参数量和配置。

直接运行模型文件可做一次简单 forward 自测：

```bash
python sam2unet_model.py
```

预期会创建一个 `(1, 4, 96, 96, 96)` 的随机输入，并打印：

- `main_logits` shape
- `rc_logit` shape
- legacy 5 通道 logits shape
- 最终 prediction shape

## 4. `train_sam2unet_post.py` 代码说明

`train_sam2unet_post.py` 是训练主脚本，负责参数解析、数据扫描、划分训练/验证集、构建 Dataset/DataLoader、训练、全体积验证、保存 checkpoint 和指标文件。

### 4.1 参数和配置

脚本通过 `argparse` 读取命令行参数，同时大多数参数也支持环境变量。配置生成流程是：

1. `build_arg_parser()` 定义所有 CLI 参数。
2. 默认值优先读取对应 `SAM2UNET_*` 环境变量。
3. `build_config(args)` 检查 `train_dir` 和 `save_dir`。
4. 自动写入 `in_channels=4`、`out_channels=5`、`class_names=["NETC","SNFH","ET","RC"]`。
5. 在输出目录保存完整 `config.json`。

必须提供：

- `--train_dir` 或 `SAM2UNET_TRAIN_DIR`
- `--save_dir` 或 `SAM2UNET_SAVE_DIR`

### 4.2 数据检查和标签统计

`find_case_dirs(train_dir, limit=None)` 会扫描训练数据根目录，寻找完整病例。每个病例必须包含：

- `{case}-t1n.nii.gz`
- `{case}-t1c.nii.gz`
- `{case}-t2w.nii.gz`
- `{case}-t2f.nii.gz`
- `{case}-seg.nii.gz`

`scan_label_statistics(train_dir, save_dir, limit=None)` 会读取每个病例的 segmentation，统计：

- 每个标签 `0..4` 的 voxel 数量。
- 是否 RC-positive。
- RC bounding box。
- RC bbox size。
- RC center。

输出文件：

- `label_stats.json`
- `rc_case_list.csv`
- 如存在非法标签病例，还会写 `skipped_invalid_label_cases.json` 和 `skipped_invalid_label_cases.csv`。

### 4.3 RC 分层训练/验证划分

`rc_stratified_split(records, split_ratio, seed, save_dir)` 会分别对 RC-positive 和 RC-negative 病例做随机划分，再合并得到训练集和验证集。

默认：

- `split_ratio=0.8`
- `split_seed=2025`

输出文件：

- `train_val_split_rc_stratified.json`
- `train_val_split_rc_stratified.csv`

这样做的目的是让稀有 RC 病例尽量同时出现在训练和验证中。如果 RC-positive 病例太少，脚本会给出 warning。

### 4.4 Patch Dataset 和 crop 策略

训练使用 `BraTSPatchDataset`，每次读取一个病例，加载 4 个模态并逐模态 z-score normalize，然后裁剪为 `crop_size`。

主任务阶段 `phase="main"`：

- 70% 概率围绕任意前景区域 `seg > 0` 裁剪。
- 30% 概率随机裁剪。

RC 阶段 `phase="rc"`：

- 80% 概率从 RC-positive 病例中围绕 `seg == 4` 裁剪。
- 20% 概率围绕 `NETC/SNFH/ET` 等非 RC 前景区域裁剪 hard negative，并尽量避免裁到 RC。

如果原图小于 crop size，会先 zero padding。

验证使用 `BraTSFullVolumeDataset`，加载完整体数据，不裁剪，用于 sliding-window full-volume validation。

### 4.5 Loss 和指标

训练损失由 `Plan2Losses` 计算：

主任务：

- `MainDiceLoss`
- `CrossEntropyLoss(ignore_index=4)`

这里 `RC` 标签不会被当作背景训练，而是作为 `ignore_index=4` 忽略，避免主任务 head 学错 RC 区域。

RC 任务：

- `FocalBCELoss`
- `TverskyLoss`

RC target 是二值图：

```python
target_rc = (seg == 4)
```

训练过程中会用 `dice_score_volume()` 计算 `NETC/SNFH/ET/RC` 四个前景类 Dice 和平均 Dice。

### 4.6 训练流程

`train(config)` 的主要流程：

1. 选择 `cuda` 或 `cpu`。
2. 创建输出目录。
3. 扫描标签统计并写出统计文件。
4. 进行 RC 分层 train/val split。
5. 构建 main dataset、RC dataset 和 full-volume val dataset。
6. 创建 `SAM2UNet3D` 模型。
7. 创建 loss、AdamW optimizer、CosineAnnealingLR scheduler、GradScaler。
8. 如果 `latest_checkpoint.pth` 存在且没有指定 `--no_resume`，自动断点续训。
9. 每个 epoch 先跑 main phase。
10. 当 `epoch > warmup_epochs` 时，再跑 RC phase。
11. 每个 epoch 保存 `latest_checkpoint.pth` 和 `history.json`。
12. 每隔 `checkpoint_interval` 个 epoch 或最后一个 epoch，执行 full-volume validation。
13. 根据验证指标保存最佳模型。

默认 warmup：

- 前 `warmup_epochs=30` 个 epoch 只训练主任务。
- 第 `31` 个 epoch 开始增加 RC phase。

### 4.7 Optimizer 参数组

`build_optimizer()` 使用 AdamW，并拆成 3 个参数组：

| 参数组 | 默认学习率 |
| --- | --- |
| backbone/decoder/head_features | `lr=1e-4` |
| `main_head` | `main_head_lr=1e-4` |
| `rc_head` | `rc_head_lr=3e-4` |

RC head 默认学习率更高，因为 RC 是稀有目标，需要更强的专门学习。

### 4.8 Full-volume validation

验证使用 sliding window，不是 patch-level validation。`sliding_window_predict_probs()` 会：

1. 按 `crop_size` 和 `sliding_window_overlap` 切 full volume。
2. 对每个 patch 推理。
3. 聚合 `main_probs` 和 `rc_prob`。
4. 对多个 `rc_thresholds` 逐个生成最终预测。
5. 计算 `NETC/SNFH/ET/RC` Dice。

默认 RC 阈值扫描：

```text
0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50
```

模型选择分数：

```text
combined_score = 0.4 * mean(NETC, SNFH, ET Dice) + 0.6 * RC Dice
```

### 4.9 当前脚本的输出特点

当前 `train_sam2unet_post.py` 只保存指标、曲线和 checkpoint，不保存新的 NIfTI segmentation prediction images。

仓库中历史目录 `output/full_train1/predictions/*.nii.gz` 是旧输出形态，不是当前训练脚本的预期新增产物。

## 5. 直接运行训练脚本

### 5.1 安装环境

训练至少需要：

- Python 3
- PyTorch
- CUDA 版本的 PyTorch，若使用 GPU
- `nibabel`
- `numpy`
- `einops`
- `tqdm`
- `matplotlib`，用于保存 `validation_dice_by_epoch.png`

可参考：

```bash
pip install torch nibabel numpy einops tqdm matplotlib
```

服务器上建议使用已有 conda/venv/module 环境，不要在计算节点临时联网安装依赖。

### 5.2 Debug 小规模试跑

正式训练前建议先用少量病例和少量 epoch 跑通：

```bash
python train_sam2unet_post.py \
  --train_dir /scratch/bf2260/ECNU_EYU_data/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  --save_dir /scratch/bf2260/ECNU_EYU_data/work_space/S4/output/debug_post \
  --debug_case_limit 20 \
  --epochs 2 \
  --warmup_epochs 1 \
  --checkpoint_interval 1 \
  --crop_size 96,96,96 \
  --batch_size 1 \
  --num_workers 4
```

如果显存不足，可以先调小：

```bash
python train_sam2unet_post.py \
  --train_dir /scratch/bf2260/ECNU_EYU_data/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  --save_dir /scratch/bf2260/ECNU_EYU_data/work_space/S4/output/debug_small \
  --debug_case_limit 10 \
  --epochs 2 \
  --warmup_epochs 1 \
  --checkpoint_interval 1 \
  --crop_size 80,80,80 \
  --batch_size 1 \
  --sliding_window_batch_size 1
```

### 5.3 正式训练示例

```bash
python train_sam2unet_post.py \
  --train_dir /scratch/bf2260/ECNU_EYU_data/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training \
  --save_dir /scratch/bf2260/ECNU_EYU_data/work_space/S4/output/full_train_450 \
  --epochs 450 \
  --warmup_epochs 30 \
  --checkpoint_interval 10 \
  --crop_size 96,96,96 \
  --batch_size 1 \
  --accumulation_steps 1 \
  --num_workers 4 \
  --sliding_window_batch_size 1
```

断点续训默认开启。如果输出目录中已有 `latest_checkpoint.pth`，脚本会自动 resume。若想从头训练，换一个新的 `save_dir`，或加：

```bash
--no_resume
```

## 6. Slurm 运行说明

Slurm 脚本位于：

```text
slurm/train_sam2unet_post.slurm
```

提交前需要根据集群情况检查脚本顶部资源：

```bash
#SBATCH -J sam2unet_post
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
```

含义：

| Slurm 项 | 含义 |
| --- | --- |
| `-J sam2unet_post` | 作业名 |
| `-p gpu` | 分区名 |
| `-N 1` | 申请 1 个节点 |
| `-n 1` | 申请 1 个 task |
| `--gres=gpu:1` | 申请 1 张 GPU |
| `--cpus-per-task=8` | 每个 task 使用 8 个 CPU |
| `--mem=64G` | 内存 64 GB |
| `-t 72:00:00` | 最长运行 72 小时 |
| `-o %x-%j.out` | 标准输出文件 |
| `-e %x-%j.err` | 标准错误文件 |

当前脚本默认分区是 `gpu`。如果在华东师大超算八期集群上运行，可根据实际可用资源把 `#SBATCH -p gpu` 改为 `#SBATCH -p a100` 或 `#SBATCH -p v100`。如果集群要求账户，还需要增加：

```bash
#SBATCH --account=<your_account>
```

### 6.1 Slurm 脚本环境变量

脚本中的主要可配置变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `PROJECT_DIR` | `${SLURM_SUBMIT_DIR}` 或当前目录 | 项目根目录，必须包含两个 Python 文件 |
| `TRAIN_DIR` | `${PROJECT_DIR}/TrainingData` | 训练数据目录 |
| `SAVE_ROOT` | `${PROJECT_DIR}/output` | 输出根目录 |
| `RUN_MODE` | `full` | `debug` 或 `full` |
| `SAVE_DIR` | debug 时 `${SAVE_ROOT}/debug_post`，full 时 `${SAVE_ROOT}/full_train_450` | 本次运行输出目录 |
| `VENV_DIR` | 空 | virtualenv/venv 路径 |
| `CONDA_ENV` | 空 | conda/mamba 环境名 |
| `PYTHON_BIN` | `python` | 直接指定 Python 可执行文件 |
| `PYTHON_MODULE` | 空 | 需要 `module load` 的 Python 模块 |
| `CUDA_MODULE` | 空 | 需要 `module load` 的 CUDA 模块 |

环境选择三选一即可：

1. `VENV_DIR=/path/to/venv`
2. `CONDA_ENV=myenv`
3. `PYTHON_BIN=/path/to/python`

如果集群使用 module，可以设置：

```bash
PYTHON_MODULE=apps/envs/miniconda3/25.5.1
CUDA_MODULE=compiler/cuda/12.1
```

实际模块名以集群 `module avail` 为准。

### 6.2 Slurm debug 提交

先提交 debug 模式，确认数据、环境和 GPU 都正常：

```bash
sbatch --export=ALL,\
RUN_MODE=debug,\
PROJECT_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/S4/code,\
TRAIN_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training,\
SAVE_ROOT=/scratch/bf2260/ECNU_EYU_data/work_space/S4/output,\
CONDA_ENV=myenv \
slurm/train_sam2unet_post.slurm
```

debug 模式默认会设置：

| 变量 | debug 默认值 |
| --- | --- |
| `SAM2UNET_DEBUG_CASE_LIMIT` | `20` |
| `SAM2UNET_EPOCHS` | `2` |
| `SAM2UNET_WARMUP_EPOCHS` | `1` |
| `SAM2UNET_CHECKPOINT_INTERVAL` | `1` |
| `SAVE_DIR` | `${SAVE_ROOT}/debug_post` |

### 6.3 Slurm full 提交

正式训练：

```bash
sbatch --export=ALL,\
RUN_MODE=full,\
PROJECT_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/S4/code,\
TRAIN_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training,\
SAVE_ROOT=/scratch/bf2260/ECNU_EYU_data/work_space/S4/output,\
CONDA_ENV=myenv \
slurm/train_sam2unet_post.slurm
```

full 模式默认会设置：

| 变量 | full 默认值 |
| --- | --- |
| `SAM2UNET_EPOCHS` | `450` |
| `SAM2UNET_WARMUP_EPOCHS` | `30` |
| `SAM2UNET_CHECKPOINT_INTERVAL` | `10` |
| `SAVE_DIR` | `${SAVE_ROOT}/full_train_450` |

可以在提交时覆盖训练参数。例如把 crop 改成 `80,80,80`、把 full 训练改成 100 epoch。由于 `--export` 本身用逗号分隔变量，带逗号的值建议先在 shell 中导出，再提交：

```bash
export SAM2UNET_EPOCHS=100
export SAM2UNET_CROP_SIZE=80,80,80
export SAM2UNET_BATCH_SIZE=1
export SAM2UNET_SW_BATCH_SIZE=1

sbatch --export=ALL,\
RUN_MODE=full,\
PROJECT_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/S4/code,\
TRAIN_DIR=/scratch/bf2260/ECNU_EYU_data/work_space/G1/data/raw/MICCAI-LH-BraTS2025-MET-Challenge-Training,\
SAVE_ROOT=/scratch/bf2260/ECNU_EYU_data/work_space/S4/output,\
CONDA_ENV=myenv \
slurm/train_sam2unet_post.slurm
```

### 6.4 Slurm 作业检查

提交前可做 Slurm dry-run 检查：

```bash
sbatch --test-only slurm/train_sam2unet_post.slurm
```

如果需要指定分区测试，例如华东师大超算：

```bash
sbatch --test-only -p a100 -N 1 -n 1 --gres=gpu:1 --cpus-per-task=1 --mem=4G --time=00:01:00 --wrap=nvidia-smi
```

查看队列：

```bash
squeue -u $USER
```

查看历史和退出码：

```bash
sacct -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
```

取消作业：

```bash
scancel JOBID
```

Slurm 标准输出和错误默认写在提交目录：

```text
sam2unet_post-<jobid>.out
sam2unet_post-<jobid>.err
```

训练脚本自身日志还会写入：

```text
<SAVE_DIR>/logs/train-<jobid>.log
```

## 7. 参数表

所有 CLI 参数也可通过对应环境变量设置。CLI 显式传入的值优先于环境变量默认值。

| CLI 参数 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--train_dir` | `SAM2UNET_TRAIN_DIR` | 无 | 训练数据根目录，必须提供 |
| `--save_dir` | `SAM2UNET_SAVE_DIR` | 无 | 输出目录，必须提供 |
| `--epochs` | `SAM2UNET_EPOCHS` | `400` | 总训练 epoch 数；Slurm full 默认覆盖为 `450` |
| `--crop_size` | `SAM2UNET_CROP_SIZE` | `96,96,96` | 训练 patch 和验证 sliding window 大小 |
| `--batch_size` | `SAM2UNET_BATCH_SIZE` | `1` | DataLoader batch size |
| `--accumulation_steps` | `SAM2UNET_ACCUMULATION_STEPS` | `1` | 梯度累积步数 |
| `--num_workers` | `SAM2UNET_NUM_WORKERS` | `4` | DataLoader worker 数 |
| `--prefetch_factor` | `SAM2UNET_PREFETCH_FACTOR` | `2` | DataLoader worker 预取 batch 数 |
| `--checkpoint_interval` | `SAM2UNET_CHECKPOINT_INTERVAL` | `10` | 每隔多少 epoch 做 full-volume validation |
| `--debug_case_limit` | `SAM2UNET_DEBUG_CASE_LIMIT` | `None` | 只使用前 N 个完整病例，便于 debug |
| `--split_ratio` | `SAM2UNET_SPLIT_RATIO` | `0.8` | train/val 划分比例 |
| `--split_seed` | `SAM2UNET_SPLIT_SEED` | `2025` | train/val 划分随机种子 |
| `--warmup_epochs` | `SAM2UNET_WARMUP_EPOCHS` | `30` | 只训练 main phase 的 epoch 数 |
| `--rc_thresholds` | `SAM2UNET_RC_THRESHOLDS` | `0.15,0.20,0.25,0.30,0.35,0.40,0.50` | 验证时扫描的 RC 阈值 |
| `--lr` | `SAM2UNET_LR` | `1e-4` | backbone/decoder 基础学习率 |
| `--main_head_lr` | `SAM2UNET_MAIN_HEAD_LR` | `1e-4` | main head 学习率 |
| `--rc_head_lr` | `SAM2UNET_RC_HEAD_LR` | `3e-4` | RC head 学习率 |
| `--weight_decay` | `SAM2UNET_WEIGHT_DECAY` | `1e-5` | AdamW weight decay |
| `--main_loss_weight` | `SAM2UNET_MAIN_LOSS_WEIGHT` | `1.0` | main phase 中主任务 loss 权重 |
| `--rc_phase_main_loss_weight` | `SAM2UNET_RC_PHASE_MAIN_LOSS_WEIGHT` | `0.3` | RC phase 中 main loss 权重 |
| `--rc_loss_weight` | `SAM2UNET_RC_LOSS_WEIGHT` | `4.0` | RC phase 中 RC loss 权重 |
| `--rc_focal_alpha` | `SAM2UNET_RC_FOCAL_ALPHA` | `0.75` | Focal BCE alpha |
| `--rc_focal_gamma` | `SAM2UNET_RC_FOCAL_GAMMA` | `2.0` | Focal BCE gamma |
| `--rc_tversky_alpha` | `SAM2UNET_RC_TVERSKY_ALPHA` | `0.3` | Tversky false positive 权重 |
| `--rc_tversky_beta` | `SAM2UNET_RC_TVERSKY_BETA` | `0.7` | Tversky false negative 权重，默认更强调召回 |
| `--feature_size` | `SAM2UNET_FEATURE_SIZE` | `48` | 模型基础通道数 |
| `--depths` | `SAM2UNET_DEPTHS` | `4` | encoder/decoder 层级数 |
| `--num_heads` | `SAM2UNET_NUM_HEADS` | `4` | attention head 基础数量 |
| `--window_size` | `SAM2UNET_WINDOW_SIZE` | `4,4,4` | 3D window attention 窗口大小 |
| `--dropout_rate` | `SAM2UNET_DROPOUT_RATE` | `0.2` | 模型 dropout |
| `--no_attention` | 无 | `False` | 加上后关闭 attention，变成更接近纯 CNN 的 baseline |
| `--sliding_window_batch_size` | `SAM2UNET_SW_BATCH_SIZE` | `1` | 验证 sliding window 推理 batch size |
| `--sliding_window_overlap` | `SAM2UNET_SW_OVERLAP` | `0.5` | 验证 sliding window 重叠比例 |
| `--no_resume` | 无 | `False` | 加上后不从 `latest_checkpoint.pth` 断点续训 |

## 8. 预期结果文件

训练输出位于 `save_dir`。当前脚本预期生成：

| 文件 | 作用 |
| --- | --- |
| `config.json` | 本次训练的完整配置 |
| `label_stats.json` | 每个病例的标签统计、RC 信息、跳过病例信息 |
| `rc_case_list.csv` | 每个病例的 RC 阳性状态和 RC bbox/center |
| `skipped_invalid_label_cases.json` | 可选，记录含非法标签的跳过病例 |
| `skipped_invalid_label_cases.csv` | 可选，记录含非法标签的跳过病例 |
| `train_val_split_rc_stratified.json` | RC 分层 train/val 划分 |
| `train_val_split_rc_stratified.csv` | RC 分层 train/val 划分表格 |
| `latest_checkpoint.pth` | 最近一次 checkpoint，用于断点续训 |
| `best_main_model.pth` | 主任务 Dice 最好的 checkpoint |
| `best_rc_model.pth` | RC Dice 最好的 checkpoint |
| `best_combined_model.pth` | 综合分数最好的 checkpoint |
| `history.json` | 每个 epoch 的训练 loss、Dice 和学习率记录 |
| `validation_rc_threshold_sweep.csv` | 每次验证时各 RC 阈值下的 Dice |
| `validation_dice_summary.csv` | 当前验证中综合最优阈值的 Dice 摘要 |
| `validation_dice_summary.json` | 同上，JSON 格式 |
| `validation_dice_per_case.csv` | 最优 RC 阈值下每个验证病例的 Dice |
| `validation_dice_history.csv` | 每次验证的最佳阈值和 Dice 历史 |
| `validation_dice_by_epoch.png` | epoch-Dice 曲线图 |
| `logs/train-<jobid>.log` | Slurm 脚本 tee 出来的训练日志 |

Slurm 还会在提交目录生成：

| 文件 | 作用 |
| --- | --- |
| `sam2unet_post-<jobid>.out` | Slurm 标准输出 |
| `sam2unet_post-<jobid>.err` | Slurm 标准错误 |

注意：当前 `train_sam2unet_post.py` 不保存新的 NIfTI 预测图。历史目录 `output/full_train1/predictions/*.nii.gz` 来自旧版本输出，不作为当前脚本的预期结果。

## 9. 如何判断 debug 跑通

debug 作业正常时应看到：

1. Slurm `.out` 或 `logs/train-<jobid>.log` 中打印 `torch cuda: ...` 和 `cuda available: True`。
2. `label_stats.json` 和 `rc_case_list.csv` 被写出。
3. `train_val_split_rc_stratified.json` 和 `.csv` 被写出。
4. 训练进度条中 `loss`、`main_loss`、`rc_loss` 是有限数值，不是 `nan`。
5. `latest_checkpoint.pth` 被写出。
6. 因 debug 默认 `checkpoint_interval=1`，应生成 validation 相关 CSV/JSON 和 `validation_dice_by_epoch.png`。
7. 如果 debug 运行第二次且不加 `--no_resume`，日志中应出现 `Resuming from .../latest_checkpoint.pth`。

## 10. 常见问题

### 10.1 数据目录缺文件

报错或提示：

```text
No valid cases found ...
Missing TrainingData directory ...
Skipped incomplete case directories.
```

检查每个病例目录是否严格包含 4 个模态和 1 个 segmentation，文件名必须是 `{case}-{modality}.nii.gz` 和 `{case}-seg.nii.gz`。

### 10.2 显存不足

常见处理顺序：

1. 把 `--sliding_window_batch_size` 或 `SAM2UNET_SW_BATCH_SIZE` 改为 `1`。
2. 把 `--batch_size` 保持为 `1`。
3. 减小 `--crop_size`，例如 `96,96,96` 改成 `80,80,80`。
4. 增大 `--accumulation_steps` 来维持有效 batch。
5. 必要时加 `--no_attention` 做低显存 baseline。

### 10.3 没有 CUDA 或 GPU 不可见

检查日志：

```text
cuda available: False
Using device: cpu
```

可能原因：

- Slurm 没有申请 GPU。
- 忘记写 `#SBATCH --gres=gpu:1`。
- `CUDA_VISIBLE_DEVICES` 被错误设置。
- PyTorch 不是 CUDA 版本。
- CUDA module 没加载。

可先在作业里运行：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 10.4 conda、mamba 或 module 找不到

如果设置了 `CONDA_ENV`，脚本会尝试寻找 `conda` 或 `mamba`。如果都找不到，会提示：

```text
CONDA_ENV=... was set, but conda/mamba was not found.
Set PYTHON_MODULE, VENV_DIR, or PYTHON_BIN for this cluster.
```

解决方式：

- 用 `module avail` 找到正确的 Python/conda module。
- 设置 `PYTHON_MODULE`。
- 或直接设置 `VENV_DIR=/path/to/venv`。
- 或设置 `PYTHON_BIN=/path/to/python`。

### 10.5 断点续训

默认会从 `<save_dir>/latest_checkpoint.pth` 续训。若不想续训：

```bash
python train_sam2unet_post.py ... --no_resume
```

或使用新的 `save_dir`。

### 10.6 正式训练前必须先 debug

建议先跑：

```bash
RUN_MODE=debug
```

确认环境、数据、输出目录和 GPU 都没问题后，再跑：

```bash
RUN_MODE=full
```

这样可以避免长时间排队后才发现路径、环境或数据格式错误。
