# 基于 EDM 的条件扩散模型用于脑肿瘤 MRI 生成 / EDM-Based Conditional Diffusion Model for Brain Tumor MRI Generation

---

## 中文版

### 1. 模型概述

本研究构建了一个基于 **EDM（Elucidating the Design Space of Diffusion Models, Karras et al. 2022）** 框架的条件扩散模型，用于从脑肿瘤分割标签生成逼真的多模态 MRI 图像。模型以病灶级 64³ 各向同性体素块为处理单元，以分割标签（4 类肿瘤成分的二值掩膜）为空间条件，逐模态独立学习从纯噪声到目标 MRI 的迭代去噪映射。生成结果用于数据增强（MET-AUG 流程），以扩充 nnUNet 分割模型的训练集。

### 2. 扩散框架

采用 EDM 的连续时间 σ-参数化扩散形式。在前向过程中，对干净图像 x₀ 注入高斯噪声：

$$\mathbf{x}_{\text{noisy}} = \mathbf{x}_0 + \sigma \cdot \boldsymbol{\varepsilon}, \quad \sigma \sim \text{LogNormal}(P_{\text{mean}}, P_{\text{std}}^2)$$

其中 σ 为连续噪声水平，P_mean=−1.2，P_std=1.2（EDM 论文默认值）。模型以 data-prediction 方式训练，直接预测干净图像 x₀。为稳定不同 σ 下的训练，采用 EDM 预条件（preconditioning）策略：

$$c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{out}} = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}$$

网络输出 F_θ 与跳跃连接的组合形成最终去噪预测：

$$D_\theta(\mathbf{x}_{\text{noisy}}, \sigma) = c_{\text{skip}} \cdot \mathbf{x}_{\text{noisy}} + c_{\text{out}} \cdot F_\theta(c_{\text{in}} \cdot \mathbf{x}_{\text{noisy}}, \log\sigma)$$

训练损失为 λ(σ) 加权的均方误差：

$$\mathcal{L} = \mathbb{E}_{\sigma, \mathbf{x}_0, \boldsymbol{\varepsilon}} \left[ \lambda(\sigma) \cdot \| D_\theta(\mathbf{x}_0 + \sigma\boldsymbol{\varepsilon}, \sigma) - \mathbf{x}_0 \|_2^2 \right], \quad \lambda(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$$

推理阶段采用 Heun 二阶 ODE 求解器，从 σ_max 出发沿确定性概率流 ODE 递推至 σ_min。步长分布由 EDM 的 ρ-加权离散化策略确定：

$$t_i = \left( \sigma_{\text{max}}^{1/\rho} + \frac{i}{N-1} \cdot (\sigma_{\text{min}}^{1/\rho} - \sigma_{\text{max}}^{1/\rho}) \right)^\rho$$

每步执行一次 Euler 预估和一次 Heun 梯形校正，共 2N 次网络前向（NFE）。

**表 1. EDM 噪声调度关键参数.**

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| σ_data | 0.5 (minmax) / 1.0 (z-score) | 数据标准差估计，决定预条件拐点 |
| σ_max | 50 | 最大噪声水平，推理起始 σ |
| σ_min | 0.002 | 最小噪声水平，推理目标 σ |
| ρ | 7 | 推理 σ 步长分布弯曲度 |
| 采样步数 N | 18 | Heun solver 步数（36 NFE） |

### 3. 网络架构

模型采用时间条件 3D U-Net 作为去噪骨干网络（`TimeConditionedUNet_NnU`）。架构基于 MONAI 的 3D UNet 实现，并借鉴 nnU-Net 的参数自动规划策略——根据输入 patch 尺寸和 GPU 显存容量自适应确定网络深度与特征图宽度。

**表 2. 默认网络架构参数（输入 64³，≥16 GB GPU）.**

| 特征 | 配置 |
|:---|:---|
| 编码器-解码器阶段数 | 5（4 次下采样 + 瓶颈层） |
| 各阶段特征图数 | (48, 96, 192, 384, 768) |
| 各阶段步长 | (2, 2, 2, 2, 1) |
| 卷积核大小 | 3×3×3 |
| 归一化 | InstanceNorm3d（eps=1×10⁻⁵, affine=True） |
| 激活函数 | LeakyReLU（负斜率 1×10⁻²） |
| 参数初始化 | He 初始化（kaiming_normal_） |
| 总参数量 | 约 2700 万 |

**时间嵌入**：对于 EDM/lognsr 等连续时间调度，采用 Gaussian 随机傅里叶特征（ContinuousNoiseEmbedding）。将 log σ 通过 64 个随机高斯频率（μ=4, σ=4）映射为 128 维傅里叶特征向量，经两层 MLP（128→256→256，SiLU 激活）变换后投影为 8 通道的空间广播张量，与输入图像和条件标签沿通道维拼接。对于离散时间步调度（DDPM/DDIM），采用标准正弦位置嵌入。

**条件注入**：条件模态为分割标签的二值掩膜（4 通道：NETC、SNFH、ET、RC）。去噪网络的输入由三部分通道拼接组成：含噪目标图像（1 通道）+ 条件标签（4 通道）+ 时间嵌入通道（8 通道）= 13 通道。这种拼接式的条件注入方式不需要修改骨干网络内部结构，可实现即插即用。

### 4. 输入与输出

**输入（条件）**：分割标签 `label_cube` — 形状为 [1, 4, 64, 64, 64] 的二值张量，4 个通道分别对应非增强肿瘤核心（NETC）、囊性/坏死成分（SNFH）、增强肿瘤（ET）、放射学一致区（RC），以肿瘤区域连通域为中心提取，含 10% 边缘扩展。

**输出（生成目标）**：单模态 MRI `scan_cube` — 形状为 [1, 1, 64, 64, 64] 的灰度张量，目标模态为 t1c、t1n、t2w、t2f 中的一种。四个模态各需独立训练一个模型。

**归一化方案**：

| 方案 | 值域 | σ_data | padding 值 | 适用场景 |
|:---|:---|:---|:---|:---|
| minmax | [-1, 1] | 0.5 | -1 | 独立评估、旧版 checkpoint 兼容 |
| z-score | ~N(0, 1)（脑掩码内） | 1.0 | 0 | S2 nnUNet 即时增强（推荐） |

两种方案训练的 checkpoint 不兼容，训练与推理必须使用相同的归一化方式和 σ_data。

### 5. 训练配置

**表 3. 训练超参数.**

| 参数 | 取值 |
|:---|:---|
| 优化器 | AdamW（β₁=0.5, β₂=0.999） |
| 学习率 | 2 × 10⁻⁴ |
| 权重衰减 | 1 × 10⁻⁵ |
| 批大小 | 16 |
| 总迭代步数 | 100,000 |
| 自动混合精度 | 启用（AMP, CUDA 设备） |
| 梯度裁剪 | 最大范数 12 |
| 小病灶加权因子 | 3.0 |
| 小病灶体素阈值 | 27 |
| 患者级均衡 | √(n_crops) |
| EMA | 不启用（与 GAN 对比时保持训练/推理一致） |

**小病灶加权**：针对转移灶体素悬殊问题（42% 病灶 < 27 体素），引入两级加权方案：

$$\text{sample\_weight} = \frac{1 + w_s \cdot \text{clamp}(T_s / n_{\text{voxels}}, 0, c_{\text{max}})}{\sqrt{N_{\text{crops}}^{\text{patient}}}}$$

其中 w_s=3.0 为小病灶加权因子，T_s=27 为体素阈值，c_max=1 为截断上限。分子为 crop 级小病灶放大因子（小病灶最高 4× 权重），分母为患者级均衡因子（多病灶患者降权，防止单个多发病例主导梯度）。

**训练策略**：每模态独立训练，4 个模态并行执行。每次迭代随机采样一个 batch 的病灶 crops，通过 `GaussianNoiseTumour` 变换根据 EDM log-normal 分布采样 σ 并向干净图像加噪。模型以前述 13 通道拼接作为输入，在 λ(σ) 加权的 MSE 损失约束下学习预测去噪图像。每 10,000 步自动保存 checkpoint。

### 6. 数据预处理

**病灶提取与索引**：`csv_creator.py` 对全脑分割标签执行 3D 连通域分析，将每个独立病灶记录为单独的 CSV 行。支持 `--merge_dist` 参数（默认 16 体素）合并质心距离小于阈值的邻近病灶至同一 crop。

**超大病灶预分片**：当病灶边界框（含 10% 边缘）在任一维度超过 64 体素时，自动将其切分为多个重叠的 64³ 瓦片（步长 56，重叠 8 体素）。每个瓦片独立写入 CSV 行，空瓦片（无肿瘤体素）自动丢弃。约 5.5% 的转移灶需预分片，CSV 行数增加约 24%。

**自适应随机裁剪偏移**：训练时每次迭代在安全范围内随机偏移 crop 中心位置，打破"肿瘤永远在窗口正中心"的虚假先验。偏移范围为 ±min(8, (crop_size − bbox_size) / 2 − 2) 体素，小病灶偏移更大（±8 vox），大病灶偏移更小（±2 vox）。

### 7. 推理流程

全脑生成流程包含以下步骤：

1. **连通域分析**：对输入全脑分割标签做 3D 连通域分析，识别所有独立病灶及其边界框。
2. **邻近合并**：质心距离 < 16 体素的病灶合并至同一 crop。
3. **逐病灶提取**：每病灶以边界框 +10% 边缘提取，pad 至 64³。
4. **大病灶处理**：超出 64³ 的病灶可采用缩放（resize）、跳过（skip）或滑窗分片（tile）三种策略。缩放为默认策略，滑窗步长 32，每 tile 独立 Gaussian 加权融合。
5. **扩散采样**：每病灶每模态运行 EDM Heun ODE 求解器（18 步），从 σ=50 递推至 σ≈0.002。初始噪声独立随机采样。
6. **全脑融合**：所有病灶的生成结果通过 3D Gaussian 加权融合拼回全脑空间（σ_gaussian = crop_size / 3），重叠区域平滑过渡。背景填充相应归一化模式的 padding 值（minmax=-1, zscore=0）。

**采样确定性**：默认 S_churn=0（无 Langevin 噪声注入），同一标签必定生成同一结果，保证可复现性。可通过开启 SDE churn（S_churn=5~20，需同步增加采样步数至 30~50）实现从同一标签生成多种合理 MRI。

### 8. 与其他扩散框架的兼容性

模型同时支持三种噪声调度框架：

| 框架 | noise_schedule | 预测目标 | 特征 |
|:---|:---|:---|:---|
| EDM (Karras 2022) | `edm` | 干净图像 x₀ | σ-参数化，预条件权重，Heun 2 阶 ODE，18 步 |
| Variational Diffusion (Kingma 2021) | `lognsr` | 噪声 ε | log-SNR 参数化，均匀 γ 采样，ODE solver，50 步 |
| DDPM/DDIM (Ho et al. 2020) | `cosine` / `linear` / `sqrt` | 噪声 ε | 离散时间步 (T=1000)，DDPM 采样或 DDIM 加速 |

训练和推理的 noise_schedule 必须一致，checkpoint 会保存 schedule 配置并在推理时自动匹配。

---

## English Version

### 1. Model Overview

We construct a conditional diffusion model based on the **EDM (Elucidating the Design Space of Diffusion Models, Karras et al. 2022)** framework for generating realistic multi-modal MRI images from brain tumor segmentation labels. The model operates on lesion-level 64³ isotropic voxel patches, takes segmentation labels (binary masks of 4 tumor sub-regions) as spatial conditions, and learns the iterative denoising mapping from pure noise to target MRI independently per modality. Generated outputs are used for data augmentation (MET-AUG pipeline) to expand the training set of the nnUNet segmentation model.

### 2. Diffusion Framework

We adopt EDM's continuous-time σ-parameterized diffusion formulation. In the forward process, Gaussian noise is injected into clean images x₀:

$$\mathbf{x}_{\text{noisy}} = \mathbf{x}_0 + \sigma \cdot \boldsymbol{\varepsilon}, \quad \sigma \sim \text{LogNormal}(P_{\text{mean}}, P_{\text{std}}^2)$$

where σ is the continuous noise level, P_mean=−1.2, P_std=1.2 (EDM paper defaults). The model is trained in data-prediction mode, directly predicting the clean image x₀. To stabilize training across different σ levels, EDM preprocessing is applied:

$$c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{out}} = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}$$

The network output F_θ combines with the skip connection to form the final denoised prediction:

$$D_\theta(\mathbf{x}_{\text{noisy}}, \sigma) = c_{\text{skip}} \cdot \mathbf{x}_{\text{noisy}} + c_{\text{out}} \cdot F_\theta(c_{\text{in}} \cdot \mathbf{x}_{\text{noisy}}, \log\sigma)$$

The training loss is λ(σ)-weighted mean squared error:

$$\mathcal{L} = \mathbb{E}_{\sigma, \mathbf{x}_0, \boldsymbol{\varepsilon}} \left[ \lambda(\sigma) \cdot \| D_\theta(\mathbf{x}_0 + \sigma\boldsymbol{\varepsilon}, \sigma) - \mathbf{x}_0 \|_2^2 \right], \quad \lambda(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$$

At inference, a Heun 2nd-order ODE solver iteratively integrates from σ_max to σ_min along the deterministic probability flow ODE. The step size distribution follows EDM's ρ-weighted discretization:

$$t_i = \left( \sigma_{\text{max}}^{1/\rho} + \frac{i}{N-1} \cdot (\sigma_{\text{min}}^{1/\rho} - \sigma_{\text{max}}^{1/\rho}) \right)^\rho$$

Each step performs one Euler prediction and one Heun trapezoidal correction, totaling 2N network forward evaluations (NFE).

**Table 1. Key EDM Noise Schedule Parameters.**

| Parameter | Default | Description |
|:---|:---|:---|
| σ_data | 0.5 (minmax) / 1.0 (z-score) | Data standard deviation estimate; determines preconditioning inflection point |
| σ_max | 50 | Maximum noise level; starting σ for inference |
| σ_min | 0.002 | Minimum noise level; target σ for inference |
| ρ | 7 | Curvature of inference σ step size distribution |
| Sampling steps N | 18 | Heun solver steps (36 NFE) |

### 3. Network Architecture

The model employs a time-conditioned 3D U-Net as the denoising backbone (`TimeConditionedUNet_NnU`). The architecture builds upon MONAI's 3D UNet implementation and adopts nnU-Net's heuristic parameter planner — adaptively determining network depth and feature map width based on input patch size and available GPU memory.

**Table 2. Default Network Architecture Parameters (64³ input, ≥16 GB GPU).**

| Feature | Configuration |
|:---|:---|
| Encoder-decoder stages | 5 (4 downsampling + bottleneck) |
| Feature maps per stage | (48, 96, 192, 384, 768) |
| Stride per stage | (2, 2, 2, 2, 1) |
| Convolution kernel size | 3×3×3 |
| Normalization | InstanceNorm3d (eps=1×10⁻⁵, affine=True) |
| Activation | LeakyReLU (negative slope 1×10⁻²) |
| Weight initialization | He initialization (kaiming_normal_) |
| Total parameters | ≈ 27M |

**Time embedding**: For continuous-time schedules (EDM/lognsr), Gaussian random Fourier features are employed (ContinuousNoiseEmbedding). The scalar log σ is mapped through 64 random Gaussian frequencies (μ=4, σ=4) into a 128-dimensional Fourier feature vector, transformed by a two-layer MLP (128→256→256, SiLU activation), and projected into 8-channel spatial broadcast tensors. These are concatenated channel-wise with the input image and condition label. For discrete timestep schedules (DDPM/DDIM), standard sinusoidal position embeddings are used.

**Condition injection**: The condition signal consists of binary segmentation masks (4 channels: NETC, SNFH, ET, RC). The denoising network input is formed by channel-wise concatenation of three components: noisy target image (1 channel) + condition labels (4 channels) + time embedding channels (8 channels) = 13 channels. This concatenation-based conditioning requires no modification to the backbone internals, enabling plug-and-play deployment.

### 4. Input and Output

**Input (condition)**: Segmentation label `label_cube` — a binary tensor of shape [1, 4, 64, 64, 64] with 4 channels corresponding to non-enhancing tumor core (NETC), cystic/necrotic components (SNFH), enhancing tumor (ET), and radiologically consistent zone (RC). Extracted around the tumor connected component centroid with a 10% margin expansion.

**Output (generation target)**: Single-modality MRI `scan_cube` — a grayscale tensor of shape [1, 1, 64, 64, 64], targeting one of four modalities: t1c, t1n, t2w, or t2f. Each modality requires an independently trained model.

**Normalization schemes**:

| Scheme | Value Range | σ_data | Padding Value | Use Case |
|:---|:---|:---|:---|:---|
| minmax | [-1, 1] | 0.5 | -1 | Standalone evaluation, legacy checkpoint compatibility |
| z-score | ~N(0, 1) (brain-masked) | 1.0 | 0 | S2 nnUNet online augmentation (recommended) |

Checkpoints trained under different normalization schemes are incompatible; training and inference must use the same normalization type and σ_data.

### 5. Training Configuration

**Table 3. Training Hyperparameters.**

| Parameter | Value |
|:---|:---|
| Optimizer | AdamW (β₁=0.5, β₂=0.999) |
| Learning rate | 2 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 16 |
| Total iterations | 100,000 |
| Automatic mixed precision | Enabled (AMP, CUDA devices) |
| Gradient clipping | Max norm 12 |
| Small lesion weight factor | 3.0 |
| Small lesion voxel threshold | 27 |
| Patient-level balancing | √(n_crops) |
| EMA | Disabled (maintains train/inference consistency for GAN comparison) |

**Small lesion weighting**: To address the extreme voxel count disparity in metastasis (42% of lesions < 27 voxels), a two-level weighting scheme is introduced:

$$\text{sample\_weight} = \frac{1 + w_s \cdot \text{clamp}(T_s / n_{\text{voxels}}, 0, c_{\text{max}})}{\sqrt{N_{\text{crops}}^{\text{patient}}}}$$

where w_s=3.0 is the small lesion weight factor, T_s=27 is the voxel threshold, and c_max=1 is the clamp ceiling. The numerator provides a crop-level amplification factor for small lesions (up to 4× weight), while the denominator applies patient-level balancing (down-weighting multi-lesion patients to prevent a single multi-metastatic case from dominating gradients).

**Training strategy**: Each modality is trained independently in parallel. At each iteration, a batch of lesion crops is randomly sampled. The `GaussianNoiseTumour` transform samples σ from the EDM log-normal distribution and adds noise to clean images. The model receives a 13-channel concatenated input and learns to predict the denoised image under λ(σ)-weighted MSE loss. Checkpoints are automatically saved every 10,000 steps.

### 6. Data Preprocessing

**Lesion extraction and indexing**: `csv_creator.py` performs 3D connected component analysis on whole-brain segmentation labels, recording each independent lesion as a separate CSV row. The `--merge_dist` parameter (default 16 voxels) merges nearby lesions whose centroid distance falls below the threshold into a single crop.

**Pre-tiling for oversized lesions**: When a lesion bounding box (with 10% margin) exceeds 64 voxels in any dimension, it is automatically split into multiple overlapping 64³ tiles (stride 56, overlap 8 voxels). Each tile is written as an independent CSV row; empty tiles (zero tumor voxels) are discarded. Approximately 5.5% of metastases require pre-tiling, increasing CSV row count by ~24%.

**Adaptive random crop shifting**: During training, the crop center is randomly shifted within a safe margin at each iteration, preventing the model from learning the spurious prior that "the tumor is always exactly at the window center." The shift range is ±min(8, (crop_size − bbox_size) / 2 − 2) voxels; small lesions experience larger shifts (±8 vox), while large lesions experience smaller shifts (±2 vox).

### 7. Inference Pipeline

The whole-brain generation pipeline consists of the following steps:

1. **Connected component analysis**: 3D CC analysis on the input whole-brain segmentation label identifies all independent lesions and their bounding boxes.
2. **Proximity merging**: Lesions with centroid distance < 16 voxels are merged into a single crop.
3. **Per-lesion extraction**: Each lesion is cropped with bounding box + 10% margin and padded to 64³.
4. **Large lesion handling**: Lesions exceeding 64³ adopt one of three strategies: resize (default), skip, or tile (sliding window with stride 32, independent Gaussian-weighted blending per tile).
5. **Diffusion sampling**: EDM Heun ODE solver (18 steps) is applied per lesion per modality, integrating from σ=50 to σ≈0.002. Initial noise is independently randomly sampled.
6. **Whole-brain fusion**: All lesion generations are fused back into whole-brain space via 3D Gaussian-weighted blending (σ_gaussian = crop_size / 3), with smooth transitions in overlapping regions. Background is filled with the padding value of the respective normalization scheme (minmax=-1, zscore=0).

**Sampling determinism**: Default S_churn=0 (no Langevin noise injection) ensures identical labels always produce identical outputs, guaranteeing reproducibility. Stochastic generation from a single label can be enabled by setting S_churn=5~20 (requiring increased sampling steps to 30~50).

### 8. Compatibility with Other Diffusion Frameworks

The model simultaneously supports three noise schedule frameworks:

| Framework | noise_schedule | Prediction Target | Characteristics |
|:---|:---|:---|:---|
| EDM (Karras 2022) | `edm` | Clean image x₀ | σ-parameterized, preconditioned loss, Heun 2nd-order ODE, 18 steps |
| Variational Diffusion (Kingma 2021) | `lognsr` | Noise ε | log-SNR parameterized, uniform γ sampling, ODE solver, 50 steps |
| DDPM/DDIM (Ho et al. 2020) | `cosine` / `linear` / `sqrt` | Noise ε | Discrete timesteps (T=1000), DDPM or DDIM accelerated sampling |

The noise_schedule must be identical between training and inference; the checkpoint stores the schedule configuration and auto-matches at inference time.

---

**References**:

- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. *Advances in Neural Information Processing Systems*, 35, 26565–26577.
- Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). Variational diffusion models. *Advances in Neural Information Processing Systems*, 34, 21696–21707.
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840–6851.
- Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations (ICLR)*.
