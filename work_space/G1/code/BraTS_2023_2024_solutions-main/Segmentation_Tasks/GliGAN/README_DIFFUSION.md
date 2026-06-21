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

优先直接读取共享原始根 `work_space/G1/data/raw/`。`csv_creator.py` 会递归扫描 `BraTS-MET-*` 病例，自动跳过缺 `t2w`、缺其他必需模态或 ROI 超过 96 的病例，不需要人工拆目录。只有当你想在本地或服务器上做一份轻量镜像缓存时，才使用 `prepare_diffusion_dataset.py` 生成 `work_space/G1/data/diffusion_cache/`。旧 `BraTS-GLI-*` 数据不要放进来，也不要把历史 smoke-test 结果混进来。

### 1.1 直接读 raw root

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
TASK1_ROOT="$PROJECT_ROOT/work_space/G1/data/raw"

python src/train/csv_creator.py \
  --dataset BRATS_2026 \
  --datadir "$TASK1_ROOT" \
  --logdir my_exp \
  --require_met True
```

这会递归搜索 `BraTS-MET-*` 病例目录，只把 `t1c/t1n/t2w/t2f/seg` 齐全且 96 ROI 可用的病例写进 CSV；缺 `t2w` 的病例会自动跳过，并写入 `../../Checkpoint/my_exp/my_exp_skipped.csv`。

### 1.2 可选缓存镜像

如果服务器训练希望使用本地缓存目录，必须先从当前 G2 manifest 重新生成。脚本会在重建前自动清空旧内容，确保不会混入历史缓存：

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
python scripts/prepare_diffusion_dataset.py \
  --train-manifest "$PROJECT_ROOT/work_space/G2/results/manifests/real_train_manifest.csv" \
  --fake-t2w-manifest "$PROJECT_ROOT/work_space/G2/results/qc/official_fake_t2w_cases_by_gzip_header_2026-06-15.csv" \
  --dataset-root "$PROJECT_ROOT/work_space/G1/data/diffusion_cache" \
  --mode symlink
```

默认只会选：

1. `final_qc_pass=True`
2. 病例 ID 必须是 `BraTS-MET-*`
3. 四模态和 `seg` 都齐全
4. 不在 G2 标记的 265 个 fake/broken `t2w` 病例里

`work_space/G1/data/diffusion_cache/` 只应由这条脚本链路生成。每次重建都会先清空旧内容，避免历史测试数据、上一次 smoke run 或其他残留被误读进本轮训练。

最终会得到当前可用的完整 MET 病例集合，具体数量以 G2 manifest 为准。

---

## 2. 运行

**所有命令在 `Segmentation_Tasks/GliGAN/` 目录下执行。**

### Step 1: 创建 CSV 索引

```bash
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
TASK1_ROOT="$PROJECT_ROOT/work_space/G1/data/raw"

python src/train/csv_creator.py \
    --dataset BRATS_2026 \
    --datadir "$TASK1_ROOT" \
    --logdir my_exp \
    --require_met True
```

### Step 2: 训练（每模态独立训练）

```bash
python src/train/tumour_main_diffusion.py \
    --dataset BRATS_2026 --modality t1c --logdir my_exp \
    --batch_size 2 --generator_type SwinUNETR \
    --num_steps 100000 --noise_schedule edm
```

> 每个 modality 需要独立训练，4 个模态都要跑：`t1c`、`t1n`、`t2w`、`t2f`。把 `--modality t1c` 依次替换为其他三个，`--logdir` 保持一致。

`--in_channels` 根据 dataset 自动检测（BRATS_2026=5，旧版 BRATS_2023=4），无需手动指定。

断点续训：`--resume_iter <步数>`。

### Step 3: 生成（从 label 生成 4 模态 MRI）

```bash
python src/infer/generate_from_label.py \
  --label_path "$TASK1_ROOT/BraTS-MET-00001-000/BraTS-MET-00001-000-seg.nii.gz" \
  --diffusion_ckpt_dir ../../Checkpoint/my_exp \
    --dataset BRATS_2026 \
    --output_dir ./output \
    --generator_type SwinUNETR \
    --noise_schedule edm \
  --sampling_method edm_heun --sampling_steps 18 \
  --modality all
```

输出：`./output/{casename}/{casename}-t1c.nii.gz`, `./output/{casename}/{casename}-t1n.nii.gz`, `./output/{casename}/{casename}-t2w.nii.gz`, `./output/{casename}/{casename}-t2f.nii.gz`, `./output/{casename}/{casename}-seg.nii.gz`

### Step 4: 评估

```bash
python src/infer/evaluate_generation.py \
    --diffusion_ckpt_dir ../../Checkpoint/my_exp \
    --csv_path ../../Checkpoint/my_exp/my_exp.csv \
    --dataset BRATS_2026 \
    --output_dir ./eval_results \
    --generator_type SwinUNETR \
    --noise_schedule edm \
    --sampling_method edm_heun --sampling_steps 18
```

输出 `./eval_results/metrics.json`，含 MSE、MAE、PSNR、SSIM。

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
| `--dataset` | (必填) | `BRATS_2023` / `BRATS_2026` / `BRATS_GOAT_2024` |
| `--modality` | `t1c` | `t1c` / `t1n` / `t2w` / `t2f` |
| `--batch_size` | 2 | |
| `--optim_lr` | 2e-4 | 学习率 |
| `--reg_weight` | 1e-5 | 权重衰减 |
| `--num_steps` | 100000 | 总迭代数 |
| `--n_steps` | 1000 | 扩散步数 T |
| `--noise_schedule` | `cosine` | 噪声调度（见上表） |
| `--generator_type` | `SwinUNETR` | `SwinUNETR` / `AttentionUnet` / `Unet` / `Unet_NnU` / `PlainConvUNet` |
| `--feature_size` | 48 | SwinUNETR 特征维度 |
| `--normalization` | `minmax` | `minmax` / `zscore` |
| `--p_uncond` | 0 | CFG 训练丢弃率 |
| `--snr_shift` | 0 | log-SNR 偏置（EDM/lognsr） |

### 推理

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_steps` | 1000 | 必须与训练一致 |
| `--noise_schedule` | `cosine` | 必须与训练一致 |
| `--sampling_method` | `ddpm` | `ddpm` / `ddim` / `edm_heun` / `lognsr_ode` |
| `--sampling_steps` | 50 | 采样子步数（越小越快） |
| `--eta` | 0 | 随机性（0=确定性，1≈DDPM） |
| `--cfg_weight` | 1.0 | CFG 强度（>1 增强 condition） |
| `--modality` | `all` | `all` / `t1c` / `t1n` / `t2w` / `t2f` |

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
| `--sigma_max` | 80 | 最大噪声水平，越大多样性越高，典型 50~160 |
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
│   ├── csv_creator.py               # Step 1: CSV 索引
│   └── tumour_main_diffusion.py     # Step 2: 训练
├── src/infer/
│   ├── generate_from_label.py       # Step 3: 生成
│   ├── evaluate_generation.py       # Step 4: 评估
│   └── diffusion_inference_utils.py # 采样逻辑
├── src/networks/
│   └── DiffusionNetwork.py          # 5 种 backbone + 噪声嵌入
└── model.py                         # 扩散数学（loss / 采样 / schedule）

Checkpoint/<exp_name>/
├── <exp_name>.csv
├── t1c/weights/diffusion_*.pt
├── t1n/weights/diffusion_*.pt
├── t2w/weights/diffusion_*.pt
└── t2f/weights/diffusion_*.pt
```
