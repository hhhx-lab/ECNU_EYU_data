# Tumour Diffusion — 从 seg 生成完整 MRI

训练条件扩散模型，输入分割标签（seg），输出 4 模态脑 MRI（t1c / t1n / t2w / t2f）。

---

## 0. 环境

```bash
conda create -n brats python=3.11 -y && conda activate brats
pip install torch monai nibabel numpy scipy matplotlib
```

---

## 1. 数据

每个病例一个文件夹，直接放在 `DataSet/` 下（无需子目录）：

```
DataSet/
├── BraTS-MET-00001-000/
│   ├── BraTS-MET-00001-000-t1c.nii.gz
│   ├── BraTS-MET-00001-000-t1n.nii.gz
│   ├── BraTS-MET-00001-000-t2w.nii.gz
│   ├── BraTS-MET-00001-000-t2f.nii.gz
│   └── BraTS-MET-00001-000-seg.nii.gz
├── BraTS-MET-00002-000/
│   └── ...
└── ...
```

> 支持 BraTS 2023 胶质瘤（`BraTS-GLI-*`）和 BraTS 2024 转移瘤（`BraTS-MET-*`）两种命名。转移瘤每个病例含多个病灶，胶质瘤通常为单病灶。

约束：肿瘤 bbox 各方向 ≤ `--crop_size` 体素（默认 64），超出的自动切分为重叠 64³ 瓦片（pre-tiling, stride=56, overlap 8）。每个连通域（病灶）独立成行（一条 CSV 行 = 一个病灶），支持 `--merge_dist` 合并邻近病灶。

---

## 2. 运行

**所有命令在 `Segmentation_Tasks/GliGAN/` 目录下执行。**

### Step 1: 创建 CSV 索引

```bash
python src/train/csv_creator.py \
    --dataset BRATS_2024 \
    --datadir DataSet \
    --logdir my_exp \
    --crop_size 64 \
    --merge_dist 16 \
    --val_patients "00002"
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--crop_size` | 64 | crop/pad 目标尺寸 |
| `--merge_dist` | 16 | 合并质心距离 < N 体素的病灶到同一 crop |
| `--val_patients` | "" | 逗号分隔的验证集患者 ID，该患者所有病灶归入 val |

CSV 每行代表一个病灶（而非一个患者），包含 `patient_id`, `lesion_id`, `n_voxels`, `split` 等列。`--val_patients` 指定的患者其所有病灶行 `split=val`，其余为 `split=train`。

`patient_id` 取目录名最后 9 位（`BraTS-MET-00001-000` → `00001-000`）。逗号分隔多个 ID 如 `"00001-000,00005-000"`。

CSV 创建时会检查每个病例是否同时存在 `t1c/t1n/t2w/t2f/seg`。缺任一必需文件的病例会被跳过并打印 `[WARN]`，不会写入 CSV，也不会进入训练。

### Step 2: 训练（每模态独立训练）

```bash
python src/train/tumour_main_diffusion.py \
    --dataset BRATS_2024 --modality t1c --logdir my_exp \
    --split train \
    --batch_size 16 --generator_type Unet_NnU \
    --crop_size 64 --small_lesion_weight 3.0 \
    --num_steps 100000 --noise_schedule edm --use_compile
```

> 每个 modality 需要独立训练，4 个模态都要跑：`t1c`、`t1n`、`t2w`、`t2f`。把 `--modality t1c` 依次替换为其他三个，`--logdir` 保持一致。

`--in_channels` 根据 dataset 自动检测，无需手动指定。
训练默认只读取 CSV 中 `split=train` 的行；不要把验证集混入训练。若只是做异常排查才使用 `--split all`。

断点续训：`--resume_iter <步数>`。

训练过程中会在 `Checkpoint/{logdir}/{modality}/loss_lists/loss_diffusion.log` 写入 loss 日志。另开终端实时监控：

```bash
# 文本模式（SSH 无图形界面可用），每 10 秒刷新
python scripts/watch_loss.py ../../Checkpoint/my_exp t1c --live

# 一次性查看
python scripts/watch_loss.py ../../Checkpoint/my_exp t1c

# 带图形（需要 X11 转发或本地桌面）
python scripts/watch_loss.py ../../Checkpoint/my_exp t1c --live --plot
```

也可以直接 `tail -f ../../Checkpoint/my_exp/t1c/loss_lists/loss_diffusion.log` 看原始数值。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--crop_size` | 64 | crop/pad 目标尺寸。须与 CSV 创建时的值一致 |
| `--split` | `train` | 训练读取的 CSV split。正式训练保持 `train` |
| `--small_lesion_weight` | 3.0 | 小病灶 loss 加权因子。loss 乘以 `1 + weight * clamp(27/n_voxels, 1)`。0=关闭 |

### Step 3: 生成（从 label 生成 4 模态 MRI）

```bash
python src/infer/generate_from_label.py \
    --label_path DataSet/BraTS-MET-00001-000/BraTS-MET-00001-000-seg.nii.gz \
    --diffusion_ckpt_dir ../../Checkpoint/my_exp \
    --dataset BRATS_2024 \
    --output_dir ./output \
    --generator_type Unet_NnU \
    --crop_size 64 --merge_dist 16 \
    --noise_schedule edm \
    --sampling_method edm_heun --sampling_steps 18 \
    --modality all --use_compile
```

输出：`./output/{casename}-t1c.nii.gz`, `-t1n.nii.gz`, `-t2w.nii.gz`, `-t2f.nii.gz`

**推理流程**：
1. 加载全脑 label → 连通域分析 (CC) 找出所有独立病灶
2. `--merge_dist` 合并邻近病灶（质心距离 < N 体素）
3. 每个病灶独立提取 crop（bbox + 10% margin → pad 到 `crop_size`³）
4. 逐病灶逐模态运行扩散生成
5. 3D Gaussian 加权融合拼接回全脑空间（重叠区域平滑过渡）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--crop_size` | 64 | crop/pad 目标尺寸。须与训练时一致 |
| `--merge_dist` | 16 | 合并邻近病灶的质心距离阈值（体素） |
| `--large_lesion_mode` | `resize` | 大病灶处理策略：`resize`（缩放）/ `skip`（跳过）/ `tile`（滑窗） |

### Step 4: 评估

```bash
# Patch-level 评估（每病灶独立 crop，默认）
python src/infer/evaluate_generation.py \
    --diffusion_ckpt_dir ../../Checkpoint/my_exp \
    --csv_path ../../Checkpoint/my_exp/my_exp.csv \
    --dataset BRATS_2024 \
    --output_dir ./eval_results \
    --generator_type Unet_NnU \
    --crop_size 64 \
    --split val \
    --noise_schedule edm \
    --sampling_method edm_heun --sampling_steps 18 --use_compile

# Whole-brain 评估（多病灶 Gaussian 融合全脑生成后评估，推荐转移瘤使用）
python src/infer/evaluate_generation.py \
    --diffusion_ckpt_dir ../../Checkpoint/my_exp \
    --csv_path ../../Checkpoint/my_exp/my_exp.csv \
    --dataset BRATS_2024 \
    --output_dir ./eval_results \
    --generator_type Unet_NnU \
    --crop_size 64 --evaluation_mode whole_brain \
    --split val \
    --noise_schedule edm \
    --sampling_method edm_heun --sampling_steps 18 --use_compile
```

输出 `./eval_results/metrics.json`，含 MSE、MAE、PSNR、SSIM。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--crop_size` | 64 | 须与训练一致 |
| `--split` | `val` | 评估读取的 CSV split。正式验证保持 `val`，全量诊断可用 `all` |
| `--evaluation_mode` | `patch` | `patch`：每病灶 crop 独立评估；`whole_brain`：多病灶全脑融合评估 |
| `--large_lesion_mode` | `resize` | 大病灶策略（whole_brain 模式）：`resize` / `skip` / `tile`（见 6.2 节） |
| `--max_cases` | 0 | 限制评估样本数（0=全部） |
| `--self_test` | False | 自比较模式：比对真实 scan 与自身（无需模型，验证预处理+指标正确性） |

**两种评估模式对比**：

| | patch | whole_brain |
|---|---|---|
| 评估粒度 | 每病灶 crop（CSV 每行一个指标） | 每患者全脑（每患者一个指标） |
| 预处理 | 每病灶独立 crop+pad（与训练一致） | 全脑归一化（minmax/zscore） |
| 指标计算 | 在 crop_size³ patch 上 | 在肿瘤 mask 区域（masked metrics） |
| 适用场景 | 快速 spot-check 单病灶质量 | 评估最终临床输出质量 |

---

## 3. 噪声调度（noise_schedule）

支持三条路径，通过 `--noise_schedule` 选择：

| `--noise_schedule` | 说明 | 配套 `--sampling_method` |
|---|---|---|
| `cosine` / `linear` / `sqrt` | 传统 β-schedule（DDPM） | `ddpm` / `ddim` |
| `edm` | EDM (Karras 2022), σ-parameterized | `edm_heun` |
| `lognsr` | logsnr (Kingma 2021), SNR-parameterized | `lognsr_ode` |

训练和推理的 `--noise_schedule` 必须一致。训练时 checkpoint 会保存 schedule 信息，推理时自动匹配。

### 可选：SNR shift

```bash
--snr_shift 0.5   # 正数=提高 SNR=训练更稳定，0=关闭（默认）
```

### 可选：CFG（Classifier-Free Guidance）

训练时加 `--p_uncond 0.2`（随机 20% 丢弃 condition），推理时加 `--cfg_weight 2.0`（增强 condition 强度）。默认关闭（p_uncond=0, cfg_weight=1.0）。

---

## 4. 参数速查

### 训练

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | (必填) | `BRATS_2023` / `BRATS_2024` / `BRATS_GOAT_2024` |
| `--modality` | `t1c` | `t1c` / `t1n` / `t2w` / `t2f` |
| `--batch_size` | 16 | |
| `--optim_lr` | 2e-4 | 学习率 |
| `--reg_weight` | 1e-5 | 权重衰减 |
| `--num_steps` | 100000 | 总迭代数 |
| `--n_steps` | 1000 | 扩散步数 T |
| `--noise_schedule` | `edm` | 噪声调度（见上表） |
| `--generator_type` | `Unet_NnU` | `Unet_NnU` / `PlainConvUNet` / `SwinUNETR` / `AttentionUnet` / `Unet` |
| `--feature_size` | 48 | SwinUNETR 特征维度 |
| `--normalization` | `minmax` | `minmax` / `zscore` |
| `--crop_size` | 64 | crop/pad 目标尺寸 |
| `--small_lesion_weight` | 3.0 | 小病灶加权因子。0=关闭，详见 §9 |
| `--small_lesion_threshold` | 27.0 | 病灶体素阈值，小于此值的病灶获得完整加权，详见 §9 |
| `--small_lesion_clamp` | 1.0 | threshold/n_voxels 的截断上限，1=保守，2~3=激进，详见 §9 |
| `--patient_balance_mode` | `none` | 患者级均衡：`none`/`divide`/`sqrt`，详见 §9 |
| `--use_compile` | (关闭) | 启用 torch.compile 加速（需 PyTorch ≥ 2.0，A100 受益最大） |
| `--p_uncond` | 0 | CFG 训练丢弃率 |
| `--snr_shift` | 0 | log-SNR 偏置（EDM/lognsr） |

> **关于 `--num_steps`**：100000 是从同任务 GAN 对标的经验值，未在扩散模型上做严格超参搜索。建议首次全量训练时设 150000~200000，每 10000 步自动存 checkpoint，训练完成后用评估脚本 (`evaluate_generation.py`) 对同一组 checkpoint 跑指标，画 PSNR/SSIM 随步数的曲线。拐点出现后即可确定最优步数，后续实验直接沿用，无需反复烧 GPU。

### 推理

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_steps` | 1000 | 必须与训练一致 |
| `--noise_schedule` | `edm` | 必须与训练一致 |
| `--sampling_method` | `edm_heun` | `edm_heun` / `lognsr_ode` / `ddpm` / `ddim` |
| `--sampling_steps` | 18 | 采样子步数（越小越快） |
| `--eta` | 0 | 随机性（0=确定性，1≈DDPM） |
| `--cfg_weight` | 1.0 | CFG 强度（>1 增强 condition） |
| `--modality` | `all` | `all` / `t1c` / `t1n` / `t2w` / `t2f` |
| `--crop_size` | 64 | crop/pad 目标尺寸（须与训练一致） |
| `--merge_dist` | 16 | 合并邻近病灶的距离阈值（体素） |
| `--large_lesion_mode` | `resize` | 大病灶策略：`resize` / `skip` / `tile`（见 6.2 节） |
| `--use_compile` | (关闭) | 启用 torch.compile 加速（需 PyTorch ≥ 2.0） |

### 4.1 传统 β-schedule（cosine / linear / sqrt）

对应 `--noise_schedule cosine`（推荐）或 `linear` / `sqrt`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_steps` | 1000 | 扩散总步数，T 越大噪声加得越细 |
| `--noise_schedule` | `cosine` | β 调度类型：`cosine` 在两端更平缓，`linear` 均匀递增 |
| `--sampling_method` | `ddpm` | `ddpm` 质量最好但慢；`ddim` 可大幅减步子 |
| `--sampling_steps` | 50 | DDIM 子步数，50~200 常用，越小越快 |
| `--eta` | 0 | DDIM 随机性，0=确定性（推荐），1≈DDPM 随机性 |

> **调参建议：** 先用默认 cosine + 1000 步训练，DDIM 50 步采样快速验证。`n_steps` 影响训练慢快和质量上限，不建议低于 500。

### 4.2 EDM（Karras 2022）

对应 `--noise_schedule edm`，配套 `--sampling_method edm_heun`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--sigma_data` | 0.5 | 数据标准差，影响 σ 归一化，不宜大幅改动 |
| `--sigma_max` | 50 | 最大噪声水平，越大多样性越高，典型 40~160 |
| `--sigma_min` | 0.002 | 最小噪声水平，越小细节越多，典型 0.001~0.01 |
| `--rho` | 7 | 训练 σ 采样密度，越大噪声大的步采样越多，典型 5~10 |
| `--snr_shift` | 0 | log-SNR 偏置，正数=训练更稳定但生成偏保守，典型 0.3~0.7 |
| `--sampling_steps` | 18 | Heun solver 步数，18~50 常用，远少于 DDPM |

> **调参建议：** EDM 的核心优势是采样高效（18 步≈DDPM 1000 步）。先调 `sigma_max` 和 `snr_shift`，`rho` 一般不动。想提升细节降低 `sigma_min`。

### 4.3 logsnr（Kingma 2021）

对应 `--noise_schedule lognsr`，配套 `--sampling_method lognsr_ode`。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gamma_max` | 10 | SNR 上限 (log scale)，越大=噪声越小=起始点越干净 |
| `--gamma_min` | -10 | SNR 下限，越小=噪声越大，典型 -15~-5 |
| `--snr_shift` | 0 | 全局 log-SNR 偏置，含义同 EDM |
| `--sampling_steps` | 50 | ODE solver 步数 |
| `--eta` | 0 | 随机性，0=确定性 ODE，>0 加噪声 |

> **调参建议：** `gamma_min` 和 `gamma_max` 控制 SNR 范围，类似 EDM 的 σ 范围。先保持默认范围，主要调 `snr_shift` 控制训练稳定性。

### 4.4 CFG（所有策略通用）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--p_uncond` | 0 | 训练时随机丢弃 condition 的概率，0.1~0.3 常用 |
| `--cfg_weight` | 1.0 | 推理时条件强度，1=正常，2~3 增加 condition 保真度 |

> **注意：** 训练和推理的 `--noise_schedule` / `--n_steps` 必须一致。`--generator_type` 推理时也必须与训练一致。

---

## 5. 文件结构

```
Segmentation_Tasks/GliGAN/
├── src/train/
│   ├── csv_creator.py               # Step 1: CSV 索引（支持连通域分析 + 多病灶）
│   └── tumour_main_diffusion.py     # Step 2: 训练（支持 --crop_size + 小病灶加权）
├── src/utils/
│   ├── crop_label.py                # 病灶裁剪（支持 target_size 参数）
│   ├── gaussian_noise_tumour.py     # 肿瘤区域噪声注入（支持 target_size）
│   ├── gaussian_noise_tumour_extended.py  # 扩展噪声（支持 target_size）
│   └── data_utils.py                # 数据加载（传递 crop_size 到各变换）
├── src/infer/
│   ├── generate_from_label.py       # Step 3: 生成（多病灶 CC + Gaussian 融合）
│   ├── evaluate_generation.py       # Step 4: 评估
│   └── diffusion_inference_utils.py # 采样逻辑
├── src/networks/
│   └── DiffusionNetwork.py          # 5 种 backbone + 噪声嵌入（crop_size → img_size/input_size）
└── model.py                         # 扩散数学（loss / 采样 / schedule）

Checkpoint/<exp_name>/
├── <exp_name>.csv
├── t1c/weights/diffusion_*.pt
├── t1n/weights/diffusion_*.pt
├── t2w/weights/diffusion_*.pt
└── t2f/weights/diffusion_*.pt
```

---

## 6. 切换为胶质瘤模式

默认配置面向转移瘤（64³ crop、Unet_NnU、小病灶加权）。如果处理胶质瘤（单发大病灶），调整以下参数：

| 参数 | 默认（转移瘤） | 胶质瘤 |
|---|---|---|
| `--crop_size` | 64 | 96 |
| `--generator_type` | `Unet_NnU` | `SwinUNETR` |
| `--batch_size` | 4 | 2 |
| `--small_lesion_weight` | 3.0 | 0 |
| `--merge_dist` | 16 | 不需要（单病灶场景） |

```bash
# CSV 创建（胶质瘤）
python src/train/csv_creator.py \
    --dataset BRATS_2024 --datadir DataSet --logdir my_exp \
    --crop_size 96

# 训练（胶质瘤）
python src/train/tumour_main_diffusion.py \
    --dataset BRATS_2024 --modality t1c --logdir my_exp \
    --batch_size 2 --generator_type SwinUNETR --crop_size 96 \
    --small_lesion_weight 0 --num_steps 100000 --noise_schedule edm --use_compile

# 推理（胶质瘤，单病灶无 CC 分析 / 无 merge）
python src/infer/generate_from_label.py \
    --label_path ... --diffusion_ckpt_dir ... \
    --dataset BRATS_2024 --output_dir ./output \
    --generator_type SwinUNETR --crop_size 96 \
    --noise_schedule edm --sampling_method edm_heun --sampling_steps 18 --use_compile
```

### crop_size 选择参考

| crop_size | 可用 backbone | 训练速度* | 推理速度* | 适用场景 |
|---|---|---|---|---|
| 64 | Unet_NnU / PlainConvUNet / SwinUNETR / AttentionUnet / Unet | 3.4× | 3.4× | 转移瘤（默认） |
| 96 | 全部 5 种 | 1× | 1× | 胶质瘤（单发大病灶） |
| 48 | Unet_NnU / PlainConvUNet / AttentionUnet / Unet | 7.7× | 7.7× | 极小转移瘤（SwinUNETR 不可用） |

> *相对于 96³ 的速度比。SwinUNETR 需要 img_size ≥ 64（attention 下采样 16× 后 ≥ 4³ token）。48³ 仅限纯 CNN backbone。
> PlainConvUNet 比 Unet_NnU 快 30-40%（无残差连接），极端追求速度时可替换。

### 大病灶处理策略 (`--large_lesion_mode`)

当病灶的 bbox（含 margin）在任何维度超过 `--crop_size` 时，推理阶段提供三种处理策略。仅在推理 (`generate_from_label.py`) 时生效，训练不涉及。

| 策略 | CLI 值 | 原理 | 适用场景 |
|---|---|---|---|
| **缩放** | `resize`（默认） | 病灶标签等比例缩小到 crop_size 内 → 生成 → 升采样回原始尺寸 | 通用场景，速度快，质量略低于 tile |
| **跳过** | `skip` | 直接忽略大病灶，不生成对应区域（全脑背景保持 0） | 仅需分析小病灶，或大病灶占极少数 |
| **滑窗** | `tile` | 将大病灶区域切分为多个重叠的 crop_size³ 窗口，逐窗口独立生成，Gaussian 加权拼接 | 追求大病灶最高质量，但耗时显著增加 |

**三种策略的输出差异**：

- `resize`：所有病灶都生成，但大病灶的细节经过缩放-升采样两个插值过程，可能略有平滑
- `skip`：大病灶区域在全脑生成结果中为纯黑（背景值），不影响小病灶区域的生成质量
- `tile`：大病灶以原始分辨率生成，细节保留最好，但一个大病灶可能拆成 8~27 个 tile，每 tile 一次扩散采样，耗时成倍增长

**策略间的独立性**：

三种策略在代码中是互斥的三路分支，`skip` 跳过时不加载模型、不执行生成；`tile` 使用独立的 `_tile_generate_lesion()` 函数，不依赖 `extract_single_crop` 的缩放逻辑；`resize` 保持原有路径。切换策略只需改 `--large_lesion_mode` 参数，不会相互干扰。

**Tile 模式技术细节**：

- 滑窗步长 = crop_size / 2（50% 重叠）
- 每 tile 独立 Gaussian weighted blending（sigma = crop_size / 3）
- 边缘 tile 自动 padding 到 crop_size³
- 支持 EDM/logsnr/DDPM 所有采样方式

**使用示例**：

```bash
# 默认 resize（兼容现有流程）
python src/infer/generate_from_label.py ... --large_lesion_mode resize

# 跳过大病灶（快速预览小病灶质量）
python src/infer/generate_from_label.py ... --large_lesion_mode skip

# 滑窗高质量生成（大病灶较多时推荐）
python src/infer/generate_from_label.py ... --large_lesion_mode tile
```

### 训练时 Pre-tiling（超大病灶自动切分）

当病灶 bbox 任一维度 > `--crop_size` 时，CSV 创建阶段 (`csv_creator.py`) 会自动将其切分为多个重叠的 crop_size³ 瓦片，避免训练时缩放丢失空间细节。

**切分策略**：
- stride = crop_size - 8（默认 56），相邻瓦片重叠 8 体素
- 每个瓦片独立计算 `n_voxels`（从该瓦片内的实际肿瘤体素统计）
- 空瓦片（无肿瘤体素）自动丢弃
- 每个瓦片写为独立 CSV 行，训练时与普通病灶完全一致

**适用性**：
- 转移瘤数据集 ~5.5% 的病灶 > 64³，pre-tiling 后 CSV 行数增加 ~24%
- 训练总时间不变（step-based，num_steps 固定）
- 推理不受影响（推理使用独立的全脑滑窗逻辑）
- 与合并策略互斥：`merge_nearby_lesions` 中合并后超出 crop_size 的病灶不会合并（在合并阶段已过滤）

### 自适应随机 Crop 偏移

训练时每个 crop 的中心位置会在安全范围内随机偏移，打破病灶在 crop 窗口中的固定位置，防止模型学习"肿瘤总是在窗口正中间"的虚假先验。

**偏移范围**：
```
margin = min(8, max(0, (crop_size - bbox_size) // 2 - 2))
```
- 小病灶（padding ≥ 17）：偏移 ±8 vox
- 中等病灶（padding ≈ 8）：偏移 ±6 vox
- 大病灶（padding ≈ 4）：偏移 ±2 vox
- 超大病灶（padding ≤ 1）：不偏移（pre-tiling 后瓦片内恢复标准偏移范围）

**关键约束**：
- scan 和 label 在同一个窗口内同步移动，始终对齐
- 偏移 clamp 到全脑边界内
- 每次迭代独立抽签，偏移不累积
- 仅训练时生效，推理不受影响（推理是确定性全脑滑窗）

---

## 7. EDM 调参手册

EDM (Karras 2022) 在连续 σ 空间上工作，没有离散时间步的概念。训练时从 log-normal 分布采样 σ，对干净图像加噪后让模型预测去噪结果；推理时用 Heun 2nd-order ODE solver 从高 σ 逐步降噪到低 σ。

### 7.1 原理速览

```
训练: x_noisy = x_0 + σ·ε    (σ ~ logNormal(-1.2, 1.2²))
      模型预测 clean x_0，用 preconditioning 权重做 MSE loss

推理: 从 σ_max 出发，Heun ODE 18 步递推到 σ_min ≈ 0
      每步: Euler 预估 → Heun 校正 (trapezoidal rule)
```

Preconditioning 系数（Eq. 8-11）保证网络输入/输出始终接近单位方差，使训练在不同 σ 下都稳定。

---

### 7.2 参数详解

#### `--sigma_data`（默认 0.5）

**含义**：数据像素标准差估计，决定 preconditioning 的"拐点"位置。

Preconditioning 的三个系数都依赖 `sigma_data`：

```
c_in   = 1 / √(σ² + σ_data²)    → 缩放噪声输入到 ~unit variance
c_out  = σ·σ_data / √(σ² + σ_data²) → 缩放模型输出
c_skip = σ_data² / (σ² + σ_data²)   → skip connection 权重
```

- **σ >> σ_data 时**：模型主要预测干净信号（c_skip≈0, c_out≈σ_data）
- **σ << σ_data 时**：模型主要预测残差/噪声（c_skip≈1, c_out≈σ/σ_data→0），skip connection 主导
- **调大**：更重视大 σ（结构）阶段，但可能忽略细节
- **调小**：更重视小 σ（细节）阶段，但大尺度结构可能变差

> 对 minmax 归一化（0~1），0.5 是合理默认值。如果换成 z-score 归一化，数据 std≈1，建议调到 1.0。

---

#### `--sigma_max`（默认 50）

**含义**：训练中的最大噪声水平，也是推理采样的起始 σ。

- **控制生成多样性**：σ_max 越大，初始噪声越强、探索空间越大、生成多样性越高
- **与控制训练难度**：σ_max 越大，训练时极端噪声的样本越多，任务越难

| σ_max | 效果 |
|---|---|
| 40~60 | 训练更快收敛，生成更保守，适合小肿瘤/转移瘤 |
| 50（默认） | 偏向稳定，适合小/中肿瘤（转移瘤场景优化） |
| 70~100 | 更多样性，大肿瘤/胶质瘤场景推荐 |
| 100~160 | 最大多样性，需要更多训练迭代 |

**调试信号**：
- 生成图像过于模糊/相似 → 增大 σ_max
- 训练 loss 迟迟不降 → 减小 σ_max
- 生成图像有大尺度伪影/畸变 → 减小 σ_max

---

#### `--sigma_min`（默认 0.002）

**含义**：训练中的最小噪声水平，也是推理采样的目标 σ。

- **控制细节锐度**：σ_min 越小，最终去噪更彻底，细节更锐利
- **过小有风险**：模型可能从未见过如此小的 σ，产生高频伪影（棋盘格/椒盐噪声）

| σ_min | 效果 |
|---|---|
| 0.001 | 极锐利，但可能引入噪声伪影 |
| 0.002（默认） | 平衡点 |
| 0.005~0.01 | 更平滑，细节稍弱但伪影少 |

**调试信号**：
- 生成图像有高频噪声/棋盘格 → 增大 σ_min
- 生成图像过于模糊、缺失纹理 → 减小 σ_min
- 评估 SSIM 高但视觉模糊 → 减小 σ_min

---

#### `--rho`（默认 7）

**含义**：推理时 σ 步长分布的弯曲度（Eq. 5）。

```
t_i = (σ_max^(1/ρ) + i/(N-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
```

- **仅影响推理采样**，不影响训练（训练 σ 采样是 log-normal，与此无关）
- **ρ 越大**：步长在 σ 大的阶段更密集（结构去噪更精细），σ 小的阶段步长更大
- **ρ 越小**：步长分布更均匀

| ρ | 效果 |
|---|---|
| 3~5 | 均匀分布，低 σ 区细节更好 |
| 7（默认） | 偏重高 σ 区，推荐 |
| 9~10 | 更偏重高 σ 区 |

> 一般不动。如果生成的大体结构正确但细节不足，可以降至 5 试试。

---

#### `--snr_shift`（默认 0，关闭）

**含义**：全局 log-SNR 偏置。正值 = 训练时看到的 σ 整体偏小 = 去噪任务变简单 = 正则化效应。

实现方式：`ln_sigma = ln_sigma - snr_shift`（训练和推理同时应用）

- **0**：不启用，标准 EDM
- **0.3~0.5**：适度正则，训练更稳定，推荐先试 0.3
- **0.7~1.0**：强正则，生成保守、多样性降低

**调试信号**：
- 训练 loss 震荡剧烈、不收敛 → 加 0.3~0.5
- 生成图像过于平滑、缺少变化 → 减小或关闭
- 小数据集（< 50 例）训练 → 加 0.5 防过拟合

---

#### `--sampling_steps`（默认 18）

**含义**：Heun ODE solver 的步数。每步包含一次 Euler 预估 + 一次 Heun 校正，所以实际前向次数 ≈ 2×步数。

| 步数 | 推理速度 | 质量 |
|---|---|---|
| 10~12 | 快 | 可接受，适合快速预览 |
| 18（默认） | 标准 | 接近收敛，推荐 |
| 30~50 | 慢 | 略优于 18，边际收益递减 |
| 50+ | 很慢 | 几乎无额外提升 |

> 18 步是 EDM 论文推荐的甜点。日常调试用 10~12 步快速看结果，正式评估用 18~30 步。

---

#### ⚠️ 训练 σ 采样分布（不暴露 CLI，硬编码于 `model.py:325`）

训练时 σ 从 log-normal 采样：`ln(σ) ~ N(P_mean, P_std²)`。默认 `P_mean=-1.2, P_std=1.2`（EDM 论文 Table 1），覆盖 σ≈0.003~150，通常无需改动。

| 调整 | 效果 | 适用场景 |
|---|---|---|
| 增大 `P_std` | σ 分布更宽，极端 σ 采样更多 | 训练覆盖不足 |
| 减小 `P_std` | σ 集中在中等水平 | 训练不稳定 |
| 增大 `P_mean` | 整体 σ 更大，偏重结构去噪 | 结构差但细节好 |
| 减小 `P_mean` | 整体 σ 更小，偏重细节去噪 | 细节差但结构好 |

---

#### ⚠️ SDE churn（不暴露 CLI，硬编码于 `model.py:465`）

采样时注入 Langevin SDE 噪声，打破确定性 ODE 以增加多样性。默认 `S_churn=0`（关闭），同一 label 永远生成相同结果。

| 参数 | 默认 | 说明 |
|---|---|---|
| `S_churn` | 0 | churn 强度，5~20 增加多样性，需同步增加采样步数 |
| `S_min` | 0 | 只在 σ ≥ S_min 时加噪（避免在干净阶段扰动） |
| `S_max` | ∞ | 只在 σ ≤ S_max 时加噪 |
| `S_noise` | 1.0 | 噪声缩放，一般不动 |

> 如需从同一 label 生成多种合理 MRI，在 `diffusion_inference_utils.py` 调用 `sample_edm()` 处加 `S_churn=10`，同步提高 `sampling_steps` 至 30~50。平时保持 0。

---

### 7.3 CFG 参数（可选）

CFG 让模型在"有条件"和"无条件"之间插值，增强生成结果对输入 label 的保真度。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--p_uncond` | 0 | 训练时随机丢弃 condition 的概率，0.1~0.2 推荐 |
| `--cfg_weight` | 1.0 | 推理时条件强度：pred = pred_uncond + w·(pred_cond - pred_uncond) |

**开启方式**：
- 训练加 `--p_uncond 0.1`（10% 概率丢 condition）
- 推理加 `--cfg_weight 2.0`（w>1 增强 condition，w=1 等价于不开启）

**调试信号**：
- 生成图像与 label 不一致、肿瘤位置偏移 → 增大 cfg_weight（2.0~3.0）
- 生成图像过于僵硬、缺少合理变化 → 降低 cfg_weight（1.2~1.5）
- 训练 p_uncond 太大（>0.3）→ 模型见 condition 太少，生成质量下降

---

### 7.4 调参顺序建议

按优先级从高到低排列。**每个阶段只调一个参数**，确认效果后再调下一个，避免多个参数同时改动导致无法归因。

---

**1. 先确认 `sigma_data` 与归一化方式匹配**

minmax（0~1）→ 0.5；zscore（μ=0, σ=1）→ 1.0。这个不对，其余参数调了也白调。

> 如果不确定归一化方式，检查训练日志中 `Normalization:` 的输出，或临时跑 1 个 batch 打印 `real_labels.min(), real_labels.max(), real_labels.std()` 确认。

---

**2. 看 loss 曲线 —— 判断训练是否健康**

先用默认参数跑 ~5000 步，观察 loss 形态：

| 现象 | 曲线特征 | 处理 |
|---|---|---|
| 正常 | 整体下降，前半段降得快、后半段缓慢收敛 | 保持默认，进入第 3 步 |
| Loss 不降（发散前期） | 前 500 步纹丝不动或反而上升 | 降 `sigma_max`（50→40），降 `optim_lr`（2e-4→1e-4） |
| Loss 剧烈震荡 | 相邻 step 上下跳动 > 50%，毛刺密集 | 加 `snr_shift 0.3~0.5`，增大 `batch_size`（若显存允许） |
| Loss 过早平坦 | ~2000 步就基本不降了，维持在较高水平 | 增大 `sigma_max`（50→80）；若仍无效可增大 `P_std`（1.2→1.5，需改源码，非必要不动） |
| Loss 下降但出现尖刺 | 整体趋势向下，但每隔几百步出现一个高峰 | 正常现象，模型遇到了大 σ 采样点；若频繁出现可降 `sigma_max` |
| Loss 缓慢下降到很低的数值（<0.01）| 可能过拟合；检查训练集 vs 验证集 loss 差 | 加 `snr_shift 0.3` 正则化，或开 `p_uncond 0.1` |
| 中段停滞 | 5000~20000 步 loss 几乎不变 | 调 `sigma_max` 或 `sigma_min` 扩展噪声范围；若仍无效可增大 `P_std`（需改源码，非必要不动） |

> **判断标准**：EDM loss 是加权的 MSE，数值本身取决于 σ 采样分布，不要跨不同 `sigma_max` 配置比绝对值。核心看 **趋势**：是否持续下降？是否平滑？

---

**3. 看生成图像 —— 逐类问题定位**

**建议做法**：固定 1~2 个 case，用 `--sampling_steps 12` 快速采样，对比真实图像逐切片检查。

#### 3.1 细节问题

| 现象 | 典型表现 | 处理 |
|---|---|---|
| 整体模糊，边缘不锐 | 肿瘤边界模糊，灰白质对比度低 | 降 `sigma_min`（0.002→0.001），或增大 `sampling_steps`（18→30） |
| 高频伪影 | 棋盘格、椒盐噪声，背景有细密杂点 | 升 `sigma_min`（0.002→0.005） |
| 纹理过平滑 | 脑沟回不清晰，白质内部均匀无纹理 | 降 `sigma_min`，同时略微升 `sigma_max`（50→70）增加训练动态范围 |
| 细节正常但对比度不足 | 结构对但灰度范围偏窄 | 检查归一化方式，或微调 `sigma_data`（0.5→0.6） |

#### 3.2 结构问题

| 现象 | 典型表现 | 处理 |
|---|---|---|
| 大尺度结构失真 | 脑室移位、中线偏移、整体形态异常 | 降 `sigma_max`（50→40），或加 `snr_shift 0.3` |
| 多样性差 | 不同 case 生成结果高度相似 | 升 `sigma_max`（50→80）；若仍无效可开 SDE churn（`S_churn=10`，需改源码，非必要不动） |
| 多样性过强导致不稳定 | 同一 label 多次生成差异巨大 | 降 `sigma_max`，或开启 CFG 约束（`cfg_weight 1.5~2.0`） |
| 肿瘤形状不符合 label | 生成的肿瘤大小/形状与 seg 不匹配 | 先确认 label 是否正确传入模型（检查 cond 通道数），再开 CFG |

#### 3.3 模态特定问题

| 现象 | 典型表现 | 处理 |
|---|---|---|
| 某模态质量明显差于其他 | 如 t2f 模糊但 t1c 正常 | 该模态单独训练时调整参数，不要四个模态用同一套 |
| t1c/t1n 对比度不足 | 灰白质分界不清 | 优先降 `sigma_min`（0.002→0.001） |
| t2w/t2f 噪声多 | 背景区域有颗粒感 | 优先升 `sigma_min`（0.002→0.005），或加 `snr_shift` |
| 增强肿瘤区（t1c）不够亮 | 增强区与周围组织对比弱 | 该模态增大 `sigma_max`（50→80），CFG `weight 2.0~2.5` |

> **快速迭代技巧**：调参阶段用 `PlainConvUNet`（无残差连接，速度最快），参数确定后再换 `Unet_NnU` 跑全量。

---

**4. 看评估指标趋势 —— 量化确认调参方向**

每 5000~10000 步保存一次 checkpoint 并跑评估（`sampling_steps=12` 快速版），观察指标变化：

| 指标信号 | 含义 | 调参方向 |
|---|---|---|
| PSNR 升 + SSIM 升 | 质量在改善，继续训练 | 保持 |
| PSNR 升 + SSIM 平/降 | 像素匹配好了但结构感知差 | 降 `sigma_min` 提细节 |
| PSNR 平 + SSIM 升 | 结构好了但逐像素噪声大 | 升 `sigma_min` 降伪影 |
| PSNR 和 SSIM 都停滞 | 当前参数下已收敛 | 扩展噪声范围或调整采样步数 |
| 训练集指标远好于测试集 | 过拟合 | 加 `snr_shift 0.3`，或开 `p_uncond 0.1` |
| MAE/MSE 下降但 PSNR 不升 | 背景区域也被改了 | 确认肿瘤区域 mask 是否正确，检查 bbox 裁剪逻辑 |

---

**5. 参数间的联动关系**

- `sigma_max ↑` → 需要更多训练步数才能收敛（建议同步增加 `num_steps`）
- `sigma_min ↓` → 需要更多采样步数才能稳定去噪（建议同步增加 `sampling_steps`）
- `snr_shift ↑` → 训练更稳定但生成偏保守 → 可以适当增大 `sigma_max` 补偿多样性
- 开启 CFG（`p_uncond > 0`）→ 训练需要更多步数（模型同时学 conditional 和 unconditional）
- 开启 SDE churn → 需增加 `sampling_steps`（30~50），否则噪声来不及被去干净

---

**6. 完整调参工作流**

```
第 1 轮：默认参数跑 ~5000 步，看 loss 趋势是否健康
    ↓ 健康 → 第 2 轮，不健康 → 按第 2 节调
第 2 轮：跑 ~20000 步，采样 1~2 个 case（sampling_steps=12），对照真实图
    ↓ 结构 OK，细节差 → 调 sigma_min
    ↓ 细节 OK，结构差 → 调 sigma_max / snr_shift
    ↓ 都差 → 先调 sigma_max 稳住结构，再调 sigma_min
第 3 轮：确认方向后跑 ~50000 步，评估指标，微调
    ↓ PSNR/SSIM 满意 → 最终轮
第 4 轮：最终配置全量训练（建议 150000~200000 步，首次跑后根据 checkpoint 指标曲线确认收敛步数），sampling_steps=18~30 正式评估
```

> 第 1~2 轮用 `--batch_size 1 --generator_type PlainConvUNet` 快速迭代，第 3 轮起换 `Unet_NnU`。

---

### 7.5 推荐配置

**入门（稳定优先）**：
```bash
--noise_schedule edm --sigma_max 50 --sigma_min 0.005 --snr_shift 0.5
```

**标准（EDM 论文对齐）**：
```bash
--noise_schedule edm --sigma_max 50 --sigma_min 0.002 --snr_shift 0
```

**高细节（大样本量时）**：
```bash
--noise_schedule edm --sigma_max 100 --sigma_min 0.001 --sampling_steps 30
```

**高保真（配合 CFG）**：
```bash
# 训练
--noise_schedule edm --p_uncond 0.1
# 推理
--noise_schedule edm --cfg_weight 2.0 --sampling_steps 18
```

---

## 8. logsnr 调参手册

logsnr (Kingma 2021) 在连续 log-SNR 空间上工作，用均匀分布采样 γ，对干净图像加噪后让模型预测噪声。与 EDM 的核心区别：**logsnr 预测噪声 ε（无 preconditioning），EDM 预测干净 x₀（带 preconditioning）**。

### 8.1 原理速览

```
训练: γ ~ Uniform(γ_min, γ_max)
      ᾱ = sigmoid(γ),  σ = √(1-ᾱ)
      x_noisy = √ᾱ·x_0 + σ·ε
      模型预测 ε，loss = MSE(ε_pred, ε)（无加权）

推理: 从 γ_min（最嘈杂）出发，ODE 均匀递推到 γ_max（最干净）
      dx/dγ = -(1/2)·e^(-γ)·(x - D_θ(x,γ))
      每步: 用 ε_pred 反推 x̂₀，再向 γ_t+1 推进
```

**与 EDM 的关键对比**：

| | EDM | logsnr |
|---|---|---|
| 参数空间 | σ（噪声标准差，无界） | γ（log-SNR，有界） |
| σ 范围 | σ_max=80 → σ_min=0.002 | γ_min=-10(σ≈1) → γ_max=10(σ≈0.007) |
| 模型预测目标 | 干净 x₀ | 噪声 ε |
| Preconditioning | c_in/c_out/c_skip | 无 |
| Loss 加权 | λ(σ)·MSE | 普通 MSE |
| 训练 σ 分布 | log-normal (偏重大 σ) | Uniform in γ (均匀) |
| 推理步长分布 | ρ-弯曲 (偏重大 σ) | 均匀分布 |

### 8.2 参数详解

#### `--gamma_max`（默认 10）

**含义**：log-SNR 上限，对应最干净状态。γ_max 越大，训练末端越接近完全无噪。

γ → ᾱ → σ 的换算：

| γ_max | ᾱ = sigmoid(γ) | σ = √(1-ᾱ) | 效果 |
|---|---|---|---|
| 5 | 0.993 | 0.083 | 末端仍有明显噪声，生成偏模糊 |
| 7 | 0.999 | 0.031 | 较干净，细节一般 |
| 10（默认） | 0.99995 | 0.007 | 很干净，细节好 |
| 12 | 0.999994 | 0.002 | 极干净，接近 EDM σ_min=0.002 |

- **调大**：终点更干净，细节更锐利，但训练需覆盖更宽的 γ 范围
- **调小**：训练范围窄、收敛快，但生成可能偏模糊
- **注意**：γ_max 需要和 γ_min 配合，两者的差 Δγ 决定训练覆盖的总动态范围

**调试信号**：
- 生成图像整体模糊、细节丢失 → 增大 γ_max（7→10）
- 训练长时间不收敛 → 减小 γ_max 缩窄范围（10→7）

---

#### `--gamma_min`（默认 -10）

**含义**：log-SNR 下限，对应最嘈杂状态。γ_min 越小，训练的起始噪声越强。

| γ_min | ᾱ = sigmoid(γ) | σ = √(1-ᾱ) | 效果 |
|---|---|---|---|
| -5 | 0.0067 | 0.997 | 噪声较轻，训练简单 |
| -7 | 0.0009 | 0.9995 | 中等噪声 |
| -10（默认） | 0.000045 | 0.99998 | 几乎纯噪声，标准设置 |
| -12 | 6e-6 | 0.999997 | 接近纯噪声 |
| -15 | 3e-7 | ~1.0 | 完全纯噪声 |

- **调小（更负）**：起始噪声更强、采样多样性更高，但训练更难
- **调大（更接近 0）**：训练更简单，但初始噪声弱、多样性降低

**调试信号**：
- 生成多样性差、样本间过于相似 → 减小 γ_min（-10→-12）
- 训练 loss 降不下来 → 增大 γ_min（-10→-7）
- 生成图像有大尺度伪影 → 增大 γ_min

---

#### `--snr_shift`（默认 0，关闭）

**含义**：全局 γ 偏置。正值 = 训练时所有 γ 整体上移 = 每个样本的 SNR 更高 = 去噪任务变简单。

实现方式：`γ = γ + snr_shift`（训练和推理同时应用）

- **0**：不启用，标准 logsnr
- **0.3~0.5**：适度正则，训练更稳定，推荐先试 0.3
- **0.7~1.0**：强正则，生成保守

**与 EDM snr_shift 的区别**：logsnr 的 γ 范围有界（±10），而 EDM 的 ln(σ) 无界（σ 可达 80+）。所以 logsnr 下 snr_shift 的敏感度更高：shift=1 会让 γ_min 从 -10 变到 -9（σ 从 0.99998 变到 0.99988），影响比 EDM 更温和。

**调试信号**：
- 训练 loss 震荡 → 加 0.3~0.5
- 小数据集训练 → 加 0.3~0.5 防过拟合
- 生成过于保守、缺少变化 → 减小或关闭

---

#### `--sampling_steps`（默认 50）

**含义**：ODE solver 在 γ_min → γ_max 之间的均匀步数。

| 步数 | 推理速度 | 质量 |
|---|---|---|
| 20~30 | 快 | 可接受，适合调试 |
| 50（默认） | 标准 | 接近收敛 |
| 100~200 | 慢 | 略优于 50，边际收益递减 |

> logsnr 通常需要比 EDM 更多的步数（50 vs 18），因为 logsnr 没有 EDM 的 Heun 二阶校正和 ρ 步长优化，且 γ 范围被均匀离散化而非重点分配。

**调试信号**：
- 生成质量随步数增加持续提升 → 步数不足，加大到 100
- 50 步与 200 步结果几乎无差异 → 已达收敛，保持 50

---

#### `--eta`（默认 0）

**含义**：采样随机性。0 = 确定性 ODE（给定 label 总是生成同一张图），>0 = 注入 SDE 噪声，增加随机性。

| η | 效果 |
|---|---|
| 0（默认） | 确定性 ODE，可复现，推荐 |
| 0.1~0.3 | 轻微随机性，样本间有微小变化 |
| 0.5~0.8 | 明显随机性，多样性增加但可能引入伪影 |
| 1.0 | 近似 DDPM 随机性，速度慢 |

> 大多数情况下保持 0。如果需要从同一个 label 生成多种合理的 MRI，设 0.2~0.5。

---

### 8.3 CFG 参数（可选，与 EDM 通用）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--p_uncond` | 0 | 训练时随机丢弃 condition 的概率，0.1~0.2 推荐 |
| `--cfg_weight` | 1.0 | 推理时条件强度：pred = pred_uncond + w·(pred_cond - pred_uncond) |

logsnr 下 CFG 的开启方式和效果与 EDM 完全一致，详见 7.3 节。

---

### 8.4 调参顺序建议

按优先级从高到低排列。**每个阶段只调一个参数**，确认效果后再调下一个，避免多个参数同时改动导致无法归因。

logsnr 比 EDM 参数更少（无 preconditioning、无 ρ、无 SDE churn），调参面更窄，适合快速验证。但 logsnr 没有 preconditioning 的保护，参数选择对训练稳定性的影响更直接。

---

**1. 先确认 γ 范围是否匹配任务难度**

默认 γ_min=-10, γ_max=10 覆盖 σ≈0.007~1.0，对大多数脑 MRI 任务充足。

| 任务特征 | 调整建议 |
|---|---|
| 小肿瘤（bbox < 20³ 体素）、对比度高 | 缩窄到 [-7, 7] 加速收敛 |
| 大肿瘤、跨模态变化剧烈 | 保持 [-10, 10] |
| 数据噪声大、标注不精确 | 缩窄到 [-8, 8] 减少极端噪声的干扰 |
| 需要极锐利细节 | 扩展到 [-10, 12] |

> Δγ = γ_max - γ_min 决定训练覆盖的动态范围。Δγ 越大训练越难但覆盖越全。默认 20 已较宽，优先从中间调 snr_shift 而非继续扩大范围。

---

**2. 看 loss 曲线 —— 判断训练是否健康**

logsnr loss 是普通 MSE（无加权），数值不受 γ 分布影响，可以跨参数配置对比。

先用默认参数跑 ~5000 步，观察 loss 形态：

| 现象 | 曲线特征 | 处理 |
|---|---|---|
| 正常 | 整体下降，前半段降得快、后半段缓慢收敛 | 保持默认，进入第 3 步 |
| Loss 不降（发散前期） | 前 500 步纹丝不动或反而上升 | 缩窄 γ 范围（[-10,10]→[-7,7]），或降 `optim_lr`（2e-4→1e-4） |
| Loss 剧烈震荡 | 相邻 step 上下跳动 > 50%，毛刺密集 | 加 `snr_shift 0.3~0.5`，增大 `batch_size`（若显存允许） |
| Loss 过早平坦 | ~2000 步就基本不降了，维持在较高水平 | 扩展 γ 范围（[-10,10]→[-12,12]），优先扩展 `gamma_min`（更负）增大噪声覆盖 |
| Loss 下降但出现尖刺 | 整体趋势向下，但每隔几百步出现一个高峰 | 正常，高 γ（干净）样本梯度大；若频繁出现可缩小 `gamma_max`（10→8） |
| Loss 缓慢下降到很低（<0.005） | 可能过拟合；检查训练集 vs 验证集 loss 差 | 加 `snr_shift 0.3` 正则化，或开 `p_uncond 0.1` |
| 中段停滞 | 10000~30000 步 loss 几乎不变 | 扩展 γ 范围（优先 `gamma_min` 更负），或检查 `optim_lr` 是否需衰减 |

> **判断标准**：logsnr 的 MSE 通常比 EDM 的加权 MSE 数值更小，不要跨 schedule 比绝对值。核心看 **趋势** 和 **平滑度**。

---

**3. 看生成图像 —— 逐类问题定位**

**建议做法**：固定 1~2 个 case，用 `--sampling_steps 30`（logsnr 需要比 EDM 更多步数做快速预览）快速采样，对比真实图像逐切片检查。

#### 3.1 细节问题

| 现象 | 典型表现 | 处理 |
|---|---|---|
| 整体模糊，边缘不锐 | 肿瘤边界模糊，灰白质对比度低 | 升 `gamma_max`（10→12），或增加 `sampling_steps`（50→80） |
| 高频伪影 | 棋盘格、椒盐噪声，背景有细密杂点 | 降 `gamma_max`（10→7） |
| 纹理过平滑 | 脑沟回不清晰，白质内部均匀无纹理 | 升 `gamma_max`（10→12），同时扩展 `gamma_min`（-10→-12）增加动态范围 |
| 细节正常但对比度不足 | 结构对但灰度范围偏窄 | logsnr 无 sigma_data 可调，检查归一化方式，或升 `gamma_max` |

#### 3.2 结构问题

| 现象 | 典型表现 | 处理 |
|---|---|---|
| 大尺度结构失真 | 脑室移位、中线偏移、整体形态异常 | 升 `gamma_min`（-10→-7），或加 `snr_shift 0.3` |
| 多样性差 | 不同 case 生成结果高度相似 | 降 `gamma_min`（-10→-12），或设 `eta 0.2~0.5` |
| 多样性过强导致不稳定 | 同一 label 多次生成差异巨大 | 升 `gamma_min`，或开启 CFG 约束（`cfg_weight 1.5~2.0`） |
| 肿瘤形状不符合 label | 生成的肿瘤大小/形状与 seg 不匹配 | 确认 label 通道数正确，再开 CFG |

#### 3.3 模态特定问题

| 现象 | 典型表现 | 处理 |
|---|---|---|
| 某模态质量明显差于其他 | 如 t2f 模糊但 t1c 正常 | 该模态单独训练时调整参数 |
| t1c/t1n 对比度不足 | 灰白质分界不清 | 优先升 `gamma_max`（10→12） |
| t2w/t2f 噪声多 | 背景区域有颗粒感 | 优先降 `gamma_max`（10→7），或加 `snr_shift` |
| 增强肿瘤区（t1c）不够亮 | 增强区与周围组织对比弱 | 该模态开 CFG（`cfg_weight 2.0~2.5`） |

> **快速迭代技巧**：调参阶段用 `PlainConvUNet`（无残差连接，速度最快），参数确定后再换 `Unet_NnU` 跑全量。

---

**4. 看评估指标趋势 —— 量化确认调参方向**

每 5000~10000 步保存一次 checkpoint 并跑评估（`sampling_steps=30` 快速版），观察指标变化：

| 指标信号 | 含义 | 调参方向 |
|---|---|---|
| PSNR 升 + SSIM 升 | 质量在改善，继续训练 | 保持 |
| PSNR 升 + SSIM 平/降 | 像素匹配好了但结构感知差 | 升 `gamma_max` 提细节 |
| PSNR 平 + SSIM 升 | 结构好了但逐像素噪声大 | 降 `gamma_max` 减伪影，或加 `snr_shift` |
| PSNR 和 SSIM 都停滞 | 当前参数下已收敛 | 扩展 γ 范围或增大 `sampling_steps` |
| 训练集指标远好于测试集 | 过拟合 | 加 `snr_shift 0.3`，或开 `p_uncond 0.1` |
| MAE/MSE 下降但 PSNR 不升 | 背景区域也被改了 | 确认 bbox 裁剪逻辑和肿瘤 mask |

---

**5. 参数间的联动关系**

- `gamma_max ↑` → 模型需处理更干净的状态 → 可能需要更多训练步数
- `gamma_min ↓` → 初始噪声更强 → 可能需要更多采样步数来去干净
- `snr_shift ↑` → 训练更稳定但生成偏保守 → 可适当降 `gamma_min` 补偿多样性
- 开启 CFG（`p_uncond > 0`）→ 训练需要更多步数
- `eta > 0` → 增加了随机性 → 需增加 `sampling_steps`（60~100）补偿，否则噪声残留

---

**6. 完整调参工作流**

```
第 1 轮：默认参数跑 ~5000 步，看 loss 趋势是否健康
    ↓ 健康 → 第 2 轮，不健康 → 按第 2 节调
第 2 轮：跑 ~20000 步，采样 1~2 个 case（sampling_steps=30），对照真实图
    ↓ 结构 OK，细节差 → 调 gamma_max
    ↓ 细节 OK，结构差 → 调 gamma_min / snr_shift
    ↓ 都差 → 先调 gamma_min 稳住结构，再调 gamma_max
第 3 轮：确认方向后跑 ~50000 步，评估指标，微调
    ↓ PSNR/SSIM 满意 → 最终轮
第 4 轮：最终配置全量训练（建议 150000~200000 步，首次跑后根据 checkpoint 指标曲线确认收敛步数），sampling_steps=50~80、η=0 正式评估
```

> 第 1~2 轮用 `--batch_size 1 --generator_type PlainConvUNet` 快速迭代，第 3 轮起换 `Unet_NnU`。如需多样性，第 4 轮再设 `eta 0.2~0.5`。

---

### 8.5 推荐配置

**入门（稳定优先）**：
```bash
--noise_schedule lognsr --gamma_min -7 --gamma_max 7 --snr_shift 0.5
```

**标准（Kingma 论文对齐）**：
```bash
--noise_schedule lognsr --gamma_min -10 --gamma_max 10 --snr_shift 0
```

**高细节（大样本量时）**：
```bash
--noise_schedule lognsr --gamma_min -12 --gamma_max 12 --sampling_steps 100
```

**高保真（配合 CFG）**：
```bash
# 训练
--noise_schedule lognsr --p_uncond 0.1
# 推理
--noise_schedule lognsr --cfg_weight 2.0 --sampling_steps 50
```

### 8.6 EDM vs logsnr 选型建议

| 场景 | 推荐 |
|---|---|
| 追求生成质量上限 | EDM（preconditioning 通常带来更好细节） |
| 追求推理速度 | EDM（18 步 vs 50 步） |
| 调试/理解扩散过程 | logsnr（公式更简单，无 preconditioning 黑盒） |
| 小数据集（< 50 例） | logsnr（更简单的 loss 可能更好训） |
| 需要控制随机性 | 两者均可（EDM: S_churn, logsnr: eta） |
| 与其他 DDPM 工作对比 | logsnr（预测噪声，与 DDPM 一致） |

---

## 9. Loss 加权方案

### 9.1 问题背景

转移瘤场景下存在两级不平衡：

| 层级 | 问题 | 后果 |
|---|---|---|
| **Crop 级** | 病灶体素悬殊（42% 病灶 < 27mm³，同时存在数千体素的大病灶） | 大病灶主导 loss，小病灶被淹没 |
| **患者级** | 病灶数分布不均（avg 15/患者，少至 1、多至 50+） | 多病灶患者贡献更多梯度，模型偏向该患者的特征分布 |

训练时每个 batch 是随机采样的 crops（一行 CSV = 一个病灶），上述两级不平衡叠加：**一个 50 病灶患者的微小病灶，被采样的概率远低于一个单病灶患者的大病灶，但它们的 loss 之和差异巨大**。

### 9.2 公式

完整 loss 计算为四层结构：

```
Layer 1 (基础 loss):
    base_loss = weighted_MSE(denoised, x_0)     ← EDM 的 λ(σ)·MSE 或 lognsr 的 MSE

Layer 2 (per-crop 小病灶加权):
    per_crop_factor = 1 + small_lesion_weight × clamp(threshold / n_voxels, 0, clamp_max)

Layer 3 (患者级均衡):
    patient_scale = f(patient_n_crops)
        f = 1               (none)
        f = patient_n_crops (divide)
        f = √(patient_n_crops) (sqrt)

Layer 4 (最终 per-sample 加权):
    sample_weight = per_crop_factor / patient_scale
    final_loss = mean(sample_weight × base_loss_per_sample)
```

**逐项解释**：

**Layer 1 — base_loss**：噪声调度决定的逐体素 MSE。EDM 带 λ(σ) 权重（大 σ 时 weight 更大），logsnr 无权重（普通 MSE）。这一步已经给不同噪声水平的样本做了差异化的贡献。

**Layer 2 — per_crop_factor**：针对小病灶的乘性放大因子。

```
per_crop_factor = 1 + small_lesion_weight × clamp(threshold / n_voxels, 0, clamp_max)
```

- `n_voxels`：取 64³ crop 窗口内**crop 中心 (32,32,32) 所在连通分量 (CC) 的体素数**（训练时实时计算，非 CSV 写入值）。如果中心体素为肿瘤，直接取该 CC；如果中心落在背景（例如病灶被残片推出中心），取离心最近的 CC。这样可以防止大残片侵入时误将残片当作"主角"，导致小病灶的 loss 权重被错误压低。`effective_n_voxels` 字段由 `GaussianNoiseTumour` 变换在每次迭代时在线计算，不依赖 CSV 预处理
- `threshold / n_voxels`：比值。小病灶 n_voxels << threshold → 比值大；大病灶 n_voxels >> threshold → 比值趋近 0
- `clamp(·, 0, clamp_max)`：截断。防止极小病灶（1~2 体素）的比值爆炸
- `small_lesion_weight`：整体缩放。决定了"小病灶比大病灶重要多少倍"

**典型取值效果**（threshold=27, clamp=1.0）：

| n_voxels | threshold/n_voxels | per_crop_factor (w=3.0) | 相当于 |
|---|---|---|---|
| 5 | 5.4 → clamp 1.0 | 1 + 3.0 × 1.0 = **4.0** | 4× 权重 |
| 13 | 2.08 → clamp 1.0 | 1 + 3.0 × 1.0 = **4.0** | 4× 权重 |
| 27 | 1.0 | 1 + 3.0 × 1.0 = **4.0** | 4× 权重（阈值点） |
| 54 | 0.5 | 1 + 3.0 × 0.5 = **2.5** | 2.5× 权重 |
| 135 | 0.2 | 1 + 3.0 × 0.2 = **1.6** | 1.6× 权重 |
| 500 | 0.054 | 1 + 3.0 × 0.054 = **1.16** | 接近 1× |
| 2000 | 0.0135 | 1 + 3.0 × 0.0135 = **1.04** | 几乎 1× |

> **n_voxels 的来源**：训练时由 `GaussianNoiseTumour` 对 64³ crop 窗口内的 label 做连通域分析 (`ndimage.label`)，取 crop 中心 (32,32,32) 所在 CC 的体素数作为 `effective_n_voxels`（中心为背景时取最近 CC）。CSV 列 `n_voxels` 仅在 `effective_n_voxels` 不可用时作为 fallback。这样做的好处是：
> - 大残片侵入时，残片的 CC 通常不包含 crop 中心，不会被选为 `effective_n_voxels`，小病灶的 loss 权重不会被压低
> - 多个小病灶合并为一个 crop 时，中心所在的病灶被识别为"主角"，权重反映其真实大小
> - 开销极低（64³ 的 `ndimage.label` + 中心查找 < 1ms），占单步训练的 < 1%

**Layer 3 — patient_scale**：对拥有多个 crop 的患者做降权，防止多病灶患者主导梯度。

| mode | 效果 |
|---|---|
| `none` | 不做患者级均衡。适合病灶数分布均匀、或关注 crop 级质量 |
| `divide` | loss 除以 `patient_n_crops`。完全均等：每个患者的总 loss 贡献相同（无论有多少病灶） |
| `sqrt` | loss 除以 `√(patient_n_crops)`。折中：多病灶患者仍贡献略多，但不至于主导 |

**Layer 4 — 合并**：`sample_weight = per_crop_factor / patient_scale`。

- 小病灶且患者病灶少的 crop：高 per_crop_factor，低 patient_scale → 最高权重
- 大病灶且患者病灶多的 crop：低 per_crop_factor，高 patient_scale → 最低权重
- 两个方向的加权互不抵消（一个是体素大小，一个是患者级数量）

### 9.3 参数表

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `--small_lesion_weight` | 3.0 | 0~10 | 小病灶 loss 加权因子。0=关闭所有加权（Layer 2~4 均无效） |
| `--small_lesion_threshold` | 27.0 | 10~100 | 体素阈值。病灶 < 此值获得完整加权，> 此值逐渐衰减 |
| `--small_lesion_clamp` | 1.0 | 1.0~3.0 | threshold/n_voxels 的截断上限。1=保守（max 4×），2=中等（max 7×），3=激进（max 10×） |
| `--patient_balance_mode` | `none` | `none` / `divide` / `sqrt` | 患者级均衡策略 |

### 9.4 调参顺序

按优先级从高到低。每个阶段只调一个参数。

---

**1. 先确认 `patient_balance_mode`**

这是全局策略选择，影响所有样本的梯度贡献分布。

| 场景 | 推荐 |
|---|---|
| 病灶数分布均匀（CV < 0.5），关注单病灶质量 | `none` |
| 病灶数分布不均，希望每个患者贡献均等 | `divide` |
| 大多数情况（不确定时） | `sqrt`（折中，先试这个） |

> **判断方法**：检查 CSV 的 `patient_n_crops` 分布。`df.groupby('patient_id').size().describe()` 看 std/mean 比值。如果某患者有 50 个病灶而平均只有 5 个，强烈建议 `sqrt` 或 `divide`。

---

**2. 设 `small_lesion_weight`**

决定了小病灶在多大程度上被关注。

| 场景 | 推荐 weight |
|---|---|
| 胶质瘤（全部大病灶，n_voxels > 500） | 0（关闭） |
| 转移瘤，病灶大小均匀（CV < 1.0） | 1.0~2.0（轻度） |
| 转移瘤，小病灶较多（30~50% < threshold） | 3.0（默认） |
| 转移瘤，以极小病灶为主（> 50% < threshold） | 4.0~5.0 |
| 评估发现小病灶 SSIM 远低于大病灶 | 5.0~8.0 |

> **判断方法**：检查 CSV 的 `n_voxels` 分布。`df['n_voxels'].describe()` 看 25%/50%/75% 分位数。如果中位数接近 threshold，用 3.0；如果 75% 分位数仍远小于 threshold，用 4.0+。

---

**3. 设 `small_lesion_threshold`**

决定了"什么是小病灶"的分界线。

| 场景 | 推荐 threshold |
|---|---|
| 默认 | 27（~3×3×3 体素 = 3mm isotropic） |
| 病灶整体偏大，只关注 < 10 体素的极小病灶 | 10~15 |
| 病灶整体偏小，中位数 > 50 | 50~100 |
| 需要更平滑的衰减曲线（非分段式） | 增大到中位数附近 |

> `threshold` 是 `per_crop_factor` 中的拐点：n_voxels = threshold 时，ratio = 1.0，factor 达最大值。n_voxels > threshold 后 factor 逐渐衰减。阈值越大，"小病灶"的定义越宽。

---

**4. 最后调 `small_lesion_clamp`**

控制了"最小病灶能获得多少倍权重"的上限。

| clamp | max per_crop_factor (w=3.0) | 适用场景 |
|---|---|---|
| 1.0（默认） | 1 + 3.0×1.0 = 4.0× | 保守，防止 1~2 体素的噪声点主导 loss |
| 1.5 | 1 + 3.0×1.5 = 5.5× | 中等偏保守 |
| 2.0 | 1 + 3.0×2.0 = 7.0× | 主动关注极小病灶 |
| 3.0 | 1 + 3.0×3.0 = 10.0× | 激进，仅用于确认极小病灶被淹没时 |

> `clamp` 只影响 n_voxels < threshold 的"极小病灶"区间。对 n_voxels ≥ threshold 的病灶，factor 已经 < 4.0，不受 clamp 约束。所以 clamp 主要防止 1~5 体素的超小病灶（ratio > 5）获得天文数字的权重。

### 9.5 参数联动关系

- `patient_balance_mode=divide` 且 `small_lesion_weight=3.0`：多病灶患者的小病灶被双重压缩（per_crop_factor 提升但 patient_scale 压制）。如果该患者确实有很多微小病灶，考虑 `sqrt` 替代 `divide`
- `small_lesion_weight ↑` 同时 `clamp ↑`：小病灶权重急剧增大。建议先从 weight 调起，clamp 保持 1.0
- `threshold ↑` 同时 `clamp ↑`：更多病灶落入 clamp 区间，权重分布更均匀。大 threshold + 大 clamp = 几乎所有病灶带高权重，等价于不用该方案
- 开启 CFG（`p_uncond > 0`）：p_uncond 丢弃 condition 后，该样本的 n_voxels 和 patient_n_crops 仍然有效（这些值读自 CSV，不依赖 condition）

### 9.6 配置示例

**转移瘤标准**（默认，推荐起步配置）：
```bash
--small_lesion_weight 3.0 \
--small_lesion_threshold 27.0 \
--small_lesion_clamp 1.0 \
--patient_balance_mode sqrt
```

**转移瘤激进（极小病灶为主，愿意冒训练不稳定风险）**：
```bash
--small_lesion_weight 6.0 \
--small_lesion_threshold 27.0 \
--small_lesion_clamp 2.0 \
--patient_balance_mode divide
```

**混合病灶（大小病灶各半，病灶数分布均匀）**：
```bash
--small_lesion_weight 2.0 \
--small_lesion_threshold 50.0 \
--small_lesion_clamp 1.0 \
--patient_balance_mode none
```

**胶质瘤（全部大病灶，关闭加权）**：
```bash
--small_lesion_weight 0 \
--patient_balance_mode none
```

**仅患者均衡，不做小病灶加权（病灶大小均匀但数量不均）**：
```bash
--small_lesion_weight 0 \
--patient_balance_mode sqrt
```

### 9.7 监控与调试

训练时观察以下信号判断加权是否合理：

| 信号 | 含义 | 处理 |
|---|---|---|
| Loss 整体偏高且不降 | 小病灶权重过大，高噪声主导 | 降 `small_lesion_weight`（3→1.5），或降 `clamp` |
| Loss 下降但小病灶生成质量差 | 小病灶被大病灶淹没 | 升 `small_lesion_weight`（3→5），或降 `threshold` |
| 多病灶患者生成质量差于单病灶患者 | 患者级不平衡 | `none` → `sqrt` → `divide`（逐级加大均衡力度） |
| 极小病灶（< 5 体素）loss 剧烈振荡 | clamp 不足，ratio 过大导致梯度爆炸 | 降 `clamp`（2→1）或降 `weight` |
| 关闭加权后 loss 反而更稳定但小病灶质量下降 | 加权机制在起作用但幅度偏大 | 保持加权开启但降参数：weight 3→2, clamp 2→1 |
