"""
Shared diffusion inference utilities — replaces GAN one-shot generation with
multi-step DDPM sampling.  3D data-processing / post-processing is not touched.
"""
import torch
import numpy as np
import sys
import os

# Import diffusion utilities from the repo-root model.py
import importlib.util


def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_diffusion_utils = _import_from_path(
    "diffusion_utils",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "model.py"),
)


# ===========================================================================
# DDIM 本地辅助函数 — 与 model.py 中 _get_ddim_timesteps / _ddim_step 逻辑一致
# 因该模块通过 importlib 引入 model.py，直接在此处定义避免跨层依赖
# ===========================================================================

def _get_ddim_timesteps_local(n_steps, sampling_steps):
    """从 n_steps 中均匀抽取 sampling_steps 个时间步，返回倒序列表。"""
    step = n_steps // sampling_steps
    timesteps = list(range(0, n_steps, step))
    return timesteps[::-1]


def _ddim_step_local(x_t, t, t_prev, noise_pred, alphas_bar, eta, device):
    """
    DDIM 单步去噪 — Song et al. 2021, Equation 12.
    x̂_0 = (x_t − √(1−ᾱ_t))·ε_θ) / √(ᾱ_t)
    σ_t = η · √((1−ᾱ_{t−1})/(1−ᾱ_t)) · √(1 − ᾱ_t/ᾱ_{t−1})
    x_{t−1} = √(ᾱ_{t−1}) · x̂_0 + √(1−ᾱ_{t−1} − σ²) · ε_θ + σ_t · z
    """
    alpha_bar_t    = alphas_bar[t].to(device)
    alpha_bar_prev = alphas_bar[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

    # 预测干净的 x_0
    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t + 1e-8)

    # 随机性系数 σ_t — η=0 为确定性 DDIM, η=1 退化为 DDPM
    sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)) * \
            torch.sqrt(1.0 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-8))

    # DDIM 更新
    direction  = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)) * noise_pred
    noise_term = sigma * torch.randn_like(x_t) if eta > 1e-8 else 0.0

    return torch.sqrt(alpha_bar_prev) * x0_pred + direction + noise_term


# ===========================================================================
# 扩散系数构建
# ===========================================================================

def make_diffusion_coefficients(n_steps=1000, beta_schedule="cosine", device="cpu",
                                noise_schedule="cosine", sigma_data=0.5, sigma_max=80.0,
                                sigma_min=0.002, rho=7.0, gamma_max=10.0, gamma_min=-10.0,
                                snr_shift=0.0):
    """Build unified noise schedule config + legacy beta/alpha tensors.

    Returns NoiseScheduleConfig with all tensors moved to device.
    For legacy schedules, .betas, .alphas_bar_sqrt, .one_minus_alphas_bar_sqrt,
    .alphas_bar are populated. For EDM/logsnr, those are None.
    """
    schedule_cfg = _diffusion_utils.make_noise_schedule(
        schedule=noise_schedule, n_steps=n_steps,
        beta_start=1e-4, beta_end=2e-2,
        sigma_data=sigma_data, sigma_max=sigma_max, sigma_min=sigma_min, rho=rho,
        gamma_max=gamma_max, gamma_min=gamma_min,
        snr_shift=snr_shift,
    )
    # Move tensors to device
    if schedule_cfg.betas is not None:
        schedule_cfg.betas = schedule_cfg.betas.to(device)
    if schedule_cfg.alphas_bar_sqrt is not None:
        schedule_cfg.alphas_bar_sqrt = schedule_cfg.alphas_bar_sqrt.to(device)
    if schedule_cfg.one_minus_alphas_bar_sqrt is not None:
        schedule_cfg.one_minus_alphas_bar_sqrt = schedule_cfg.one_minus_alphas_bar_sqrt.to(device)
    if schedule_cfg.alphas_bar is not None:
        schedule_cfg.alphas_bar = schedule_cfg.alphas_bar.to(device)
    return schedule_cfg


@torch.no_grad()
def sample_label_diffusion(model, out_channels, spatial_size, n_steps, betas,
                           alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device,
                           method='ddpm', sampling_steps=None, eta=0.0, alphas_bar=None):
    """
    Sample a label map from pure noise.
    model: LabelDenoiser3D (forward: model(x, t))
    Returns: [1, out_channels, *spatial_size] numpy array

    参数:
        method:         'ddpm' (默认, 原 DDPM 采样) 或 'ddim' (DDIM 加速采样)
        sampling_steps: DDIM 子序列步数 (None 则使用全部 n_steps)
        eta:            DDIM 随机性 (0=确定性, 1≈DDPM; method='ddpm' 时忽略)
        alphas_bar:     DDIM 所需累积 α (method='ddpm' 时忽略, 可传入 None)
    """
    model.eval()

    # ============================ DDPM 分支 — 原逻辑不变 ============================
    if method == 'ddpm':
        shape = (1, out_channels, *spatial_size)
        x_t = torch.randn(shape, device=device)
        for t in reversed(range(n_steps)):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            noise_pred = model(x_t, t_tensor)
            if t > 0:
                beta_t = betas[t]
                alpha_t = 1.0 - beta_t
                alpha_t_sqrt = torch.sqrt(torch.tensor(alpha_t, device=device))
                noise = torch.randn_like(x_t)
                x_t = (1.0 / alpha_t_sqrt) * (
                    x_t - (1.0 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred
                ) + torch.sqrt(torch.tensor(beta_t, device=device)) * noise
            else:
                alpha_t = 1.0 - betas[t]
                alpha_t_sqrt = torch.sqrt(torch.tensor(alpha_t, device=device))
                x_t = (1.0 / alpha_t_sqrt) * (
                    x_t - (1.0 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred
                )
        return x_t.squeeze(0).cpu().numpy()

    # ============================ DDIM 分支 — 新增加速采样 ============================
    elif method == 'ddim':
        # 若未传入 alphas_bar，从 betas 实时计算
        if alphas_bar is None:
            alphas = 1.0 - betas
            alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        steps = sampling_steps if sampling_steps is not None else n_steps
        timesteps = _get_ddim_timesteps_local(n_steps, steps)
        shape = (1, out_channels, *spatial_size)
        x_t = torch.randn(shape, device=device)
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            noise_pred = model(x_t, t_tensor)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x_t = _ddim_step_local(x_t, t, t_prev, noise_pred, alphas_bar, eta, device)
        return x_t.squeeze(0).cpu().numpy()

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'ddpm' or 'ddim'.")


@torch.no_grad()
def sample_tumour_diffusion_inpaint(model, noisy_scan, label_cond, n_steps, betas,
                                   alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device,
                                   method='ddpm', sampling_steps=None, eta=0.0, alphas_bar=None):
    """
    Inpainting-style diffusion sampling for tumour generation.

    Starts from the noisy scan (Gaussian noise in tumour region) and runs
    reverse diffusion. At each step the known (non-tumour) region is replaced
    with the appropriately-noised version of the original scan.

    model:  TimeConditioned*  (forward: model(x, t, cond))
    noisy_scan:   [1, 1, 96, 96, 96]  — scan with Gaussian noise in tumour
    label_cond:   [1, C_label, 96, 96, 96]  — multi-channel tumour label
    Returns:      [1, 1, 96, 96, 96]  — reconstructed clean scan

    参数:
        method:         'ddpm' (默认, 原 DDPM 采样) 或 'ddim' (DDIM 加速采样)
        sampling_steps: DDIM 子序列步数 (None 则使用全部 n_steps)
        eta:            DDIM 随机性 (0=确定性, 1≈DDPM; method='ddpm' 时忽略)
        alphas_bar:     DDIM 所需累积 α (method='ddpm' 时忽略, 可传入 None)
    """
    model.eval()

    # Tumour mask: any non-zero label channel → tumour
    tumour_mask = (label_cond.sum(dim=1, keepdim=True) > 0).float()  # [1, 1, 96, 96, 96]
    known_mask = 1.0 - tumour_mask
    batch_size = noisy_scan.shape[0]

    # ============================ DDPM 分支 — 原逻辑一字未改 ============================
    if method == 'ddpm':
        x_t = noisy_scan.clone()
        for t in reversed(range(n_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(x_t, t_tensor, label_cond)
            if t > 0:
                beta_t = betas[t]
                alpha_t = 1.0 - beta_t
                alpha_t_sqrt = torch.sqrt(torch.tensor(alpha_t, device=device))
                noise = torch.randn_like(x_t)
                x_next = (1.0 / alpha_t_sqrt) * (
                    x_t - (1.0 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred
                ) + torch.sqrt(torch.tensor(beta_t, device=device)) * noise
            else:
                alpha_t = 1.0 - betas[t]
                alpha_t_sqrt = torch.sqrt(torch.tensor(alpha_t, device=device))
                x_next = (1.0 / alpha_t_sqrt) * (
                    x_t - (1.0 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred
                )
            # Inpainting: 已知区域替换为对应时刻的加噪原始扫描
            if t > 0:
                known_noise = torch.randn_like(x_t)
                known_noisy = (
                    alphas_bar_sqrt[t] * noisy_scan
                    + one_minus_alphas_bar_sqrt[t] * known_noise
                )
            else:
                known_noisy = noisy_scan
            x_t = tumour_mask * x_next + known_mask * known_noisy
        return x_t

    # ============================ DDIM 分支 — 新增加速 inpainting 采样 ============================
    elif method == 'ddim':
        if alphas_bar is None:
            alphas = 1.0 - betas
            alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        steps = sampling_steps if sampling_steps is not None else n_steps
        timesteps = _get_ddim_timesteps_local(n_steps, steps)
        x_t = noisy_scan.clone()
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(x_t, t_tensor, label_cond)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            # DDIM 单步去噪: x_t → x_{t_prev}（可一次跳过多个中间步）
            x_next = _ddim_step_local(x_t, t, t_prev, noise_pred, alphas_bar, eta, device)
            # Inpainting: 已知区域在时刻 t_prev 的加噪版本（与原 DDPM inpainting 逻辑一致）
            if t_prev >= 0:
                alpha_bar_prev = alphas_bar[t_prev].to(device)
                known_noise = torch.randn_like(x_t)
                known_noisy = (
                    torch.sqrt(alpha_bar_prev) * noisy_scan
                    + torch.sqrt(1.0 - alpha_bar_prev) * known_noise
                )
            else:
                known_noisy = noisy_scan  # t_prev=-1 → 最后一步，恢复干净扫描
            x_t = tumour_mask * x_next + known_mask * known_noisy
        return x_t

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'ddpm' or 'ddim'.")


@torch.no_grad()
def sample_tumour_diffusion_full(model, label_cond, spatial_size, n_steps, betas,
                                  alphas_bar_sqrt, one_minus_alphas_bar_sqrt, device,
                                  method='ddpm', sampling_steps=None, eta=0.0, alphas_bar=None,
                                  noise_schedule_cfg=None, cfg_weight=1.0):
    """
    Full 3D scan generation from a label (no inpainting).
    Starts from pure Gaussian noise, conditioned on label_cond, denoises to
    produce a complete brain MRI scan.

    model:      TimeConditioned*  (forward: model(x, t, cond))
    label_cond: [1, C_label, *spatial_size] — multi-channel tumour label
    Returns:    [1, 1, *spatial_size] — generated clean scan

    Parameters:
        method:            'ddpm', 'ddim', 'edm_heun', or 'lognsr_ode'
        sampling_steps:    sub-step count (DDIM, EDM, logsnr)
        eta:               DDIM/lognsr stochasticity
        alphas_bar:        cumulative α (DDIM)
        noise_schedule_cfg: NoiseScheduleConfig for EDM/logsnr modes
        cfg_weight:        CFG weight (1.0=normal, >1=stronger conditioning)
    """
    model.eval()
    batch_size = label_cond.shape[0]
    shape_unbatched = (1, *spatial_size)
    # Precompute zero condition for CFG (DDPM/DDIM local branches)
    zero_cond = torch.zeros_like(label_cond) if (cfg_weight != 1.0) else None

    # ============================ EDM Heun branch ============================
    if method == 'edm_heun':
        if noise_schedule_cfg is None or noise_schedule_cfg.sigma_data is None:
            raise ValueError("noise_schedule_cfg with EDM params required for method='edm_heun'")
        num_steps = sampling_steps if sampling_steps else 18
        return _diffusion_utils.sample_edm(
            model=model, shape=shape_unbatched, cond=label_cond,
            schedule_cfg=noise_schedule_cfg, device=device,
            num_steps=num_steps, cfg_weight=cfg_weight)

    # ============================ logsnr ODE branch ============================
    if method == 'lognsr_ode':
        if noise_schedule_cfg is None or noise_schedule_cfg.gamma_max is None:
            raise ValueError("noise_schedule_cfg with logsnr params required for method='lognsr_ode'")
        num_steps = sampling_steps if sampling_steps else 50
        return _diffusion_utils.sample_lognsr(
            model=model, shape=shape_unbatched, cond=label_cond,
            schedule_cfg=noise_schedule_cfg, device=device,
            num_steps=num_steps, eta=eta, cfg_weight=cfg_weight)

    # ============================ DDPM branch ============================
    if method == 'ddpm':
        x_t = torch.randn((batch_size, 1, *spatial_size), device=device)
        for t in reversed(range(n_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if zero_cond is not None:
                eps_uncond = model(x_t, t_tensor, zero_cond)
                eps_cond = model(x_t, t_tensor, label_cond)
                noise_pred = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
            else:
                noise_pred = model(x_t, t_tensor, label_cond)
            if t > 0:
                beta_t = betas[t]
                alpha_t = 1.0 - beta_t
                alpha_t_sqrt = torch.sqrt(torch.tensor(alpha_t, device=device))
                noise = torch.randn_like(x_t)
                x_t = (1.0 / alpha_t_sqrt) * (
                    x_t - (1.0 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred
                ) + torch.sqrt(torch.tensor(beta_t, device=device)) * noise
            else:
                alpha_t = 1.0 - betas[t]
                alpha_t_sqrt = torch.sqrt(torch.tensor(alpha_t, device=device))
                x_t = (1.0 / alpha_t_sqrt) * (
                    x_t - (1.0 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred
                )
        return x_t

    # ============================ DDIM branch ============================
    elif method == 'ddim':
        if alphas_bar is None:
            alphas = 1.0 - betas
            alphas_bar = torch.cumprod(alphas, dim=0).to(device)
        steps = sampling_steps if sampling_steps is not None else n_steps
        timesteps = _get_ddim_timesteps_local(n_steps, steps)
        x_t = torch.randn((batch_size, 1, *spatial_size), device=device)
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if zero_cond is not None:
                eps_uncond = model(x_t, t_tensor, zero_cond)
                eps_cond = model(x_t, t_tensor, label_cond)
                noise_pred = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
            else:
                noise_pred = model(x_t, t_tensor, label_cond)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x_t = _ddim_step_local(x_t, t, t_prev, noise_pred, alphas_bar, eta, device)
        return x_t

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'ddpm', 'ddim', 'edm_heun', or 'lognsr_ode'.")


# ---------------------------------------------------------------------------
# 归一化桥接工具：z-score 值域 ↔ [-1, 1] 扩散模型值域
# 原 GAN 版本用 rescale_array(arr, -1, 1) 将原始扫描值映射到 [-1,1]，
# on-the-fly 版本将在 nnU-Net 的 z-score 数据上执行相同操作，逻辑完全一致。
# ---------------------------------------------------------------------------


def rescale_to_mm1(arr):
    """
    将任意值域的 numpy 数组 min-max 归一化到 [-1, 1]。
    等价于原 GAN 推理中的 rescale_array(arr, minv=-1, maxv=1)。

    参数:
        arr: np.ndarray，任意形状
    返回:
        (normed_arr, arr_min, arr_max)：归一化后的数组、原始最小值、原始最大值
        （保留 arr_min/arr_max 用于后续 rescale_from_mm1 逆变换）
    """
    arr_min = np.amin(arr)
    arr_max = np.amax(arr)
    if arr_min == arr_max:
        return arr, arr_min, arr_max
    norm = (arr - arr_min) / (arr_max - arr_min)
    return (norm * 2.0) - 1.0, arr_min, arr_max


def rescale_from_mm1(arr_normed, arr_min, arr_max):
    """
    将 [-1, 1] 归一化的数组逆变换回原始值域 [arr_min, arr_max]。
    这是 rescale_to_mm1 的逆操作。

    参数:
        arr_normed: 在 [-1,1] 范围内的 numpy 数组
        arr_min:    原始数组的最小值
        arr_max:    原始数组的最大值
    返回:
        np.ndarray: 恢复到原始值域的数组
    """
    if arr_min == arr_max:
        return arr_normed
    norm = (arr_normed + 1.0) / 2.0  # [-1,1] → [0,1]
    return norm * (arr_max - arr_min) + arr_min


# ---------------------------------------------------------------------------
# 向量化后处理函数
# 原 GAN 版本的后处理使用 Python 三重 for 循环遍历 96³ 体素（约 88 万次迭代），
# 单次调用耗时约 0.5 秒。在 on-the-fly 场景下每个 batch 可能调用多次，
# 因此用 numpy 布尔索引向量化替代，将单次调用降至 <0.01 秒。
# ---------------------------------------------------------------------------


def correct_background_vec(healthy_crop_pad, imgs_recon):
    """
    向量化版本：确保生成的肿瘤图像中，脑外区域和超出 [-1,1] 的值被正确裁剪。

    原函数位置: main_random_label_random_dataset_generator_multiprocess.py:202
    原函数使用三重 for 循环，这里用 numpy 布尔索引替代。

    参数:
        healthy_crop_pad: (96,96,96) numpy 数组，健康扫描的 96³ 区域
                          （已归一化到 [-1,1]，脑外区域值为 -1）
        imgs_recon:       (96,96,96) numpy 数组，扩散模型生成的肿瘤扫描
    返回:
        np.ndarray: 修正后的 (96,96,96) 数组
    """
    imgs_corrected = np.copy(imgs_recon)

    # 脑外区域（healthy_crop_pad == -1）或值 < -1 的体素 → 设为 -1
    outside_brain = (healthy_crop_pad == -1) | (imgs_corrected < -1)
    imgs_corrected[outside_brain] = -1

    # 值 > 1 的体素 → clamp 到 1
    imgs_corrected[imgs_corrected > 1] = 1

    return imgs_corrected


def correct_label_vec(label, healthy_scan, original_label):
    """
    向量化版本：确保标签不在脑外，且不与原有肿瘤重叠。

    原函数位置: main_random_label_random_dataset_generator_multiprocess.py:188
    原函数使用三重 for 循环，这里用 numpy 布尔索引替代。

    参数:
        label:           (96,96,96) 要修正的标签
        healthy_scan:    (96,96,96) 健康扫描（归一化后，脑外值为 -1 或 0）
        original_label:  (96,96,96) 原有肿瘤标签
    返回:
        np.ndarray: 修正后的 (96,96,96) 标签
    """
    corrected = np.copy(label)
    # 脑外区域 → 清零
    corrected[healthy_scan == 0] = 0
    corrected[healthy_scan == -1] = 0
    # 与原有肿瘤重叠的区域 → 清零
    corrected[original_label != 0] = 0
    return corrected


def get_inten_coord_vec(healthy_scan_crop, original_label_crop, noise):
    """
    向量化版本：在 96³ 立方体的 6 个面上搜索"干净"参照点（脑内、非肿瘤、无噪声）。

    这些参照点用于 linear_interpolation 的强度校正：
    找到生成图像中应该与原始扫描强度一致的体素，用于拟合线性回归。

    原函数位置: main_random_label_random_dataset_generator_multiprocess.py:259
    原函数使用嵌套 for 循环遍历 6 个面，这里用 numpy 向量化替代。

    参数:
        healthy_scan_crop:  (96,96,96) 健康扫描的 96³ 区域
        original_label_crop:(96,96,96) 原有标签
        noise:              (96,96,96) 噪声掩码（肿瘤区域非零）
    返回:
        (untouch_x, untouch_y, untouch_z): 三个一维 numpy 数组，参照点的坐标
    """
    # 构建"干净"掩码：脑内 且 无原有肿瘤 且 无噪声
    is_brain = (healthy_scan_crop != 0)
    is_clean_label = (original_label_crop == 0)
    is_clean_noise = (noise == 0)
    clean_mask = is_brain & is_clean_label & is_clean_noise  # (96,96,96)

    untouch_x = []
    untouch_y = []
    untouch_z = []

    # 沿 6 个面采样，通过 clean_mask 直接过滤
    # y=0 面
    face_mask = np.zeros_like(clean_mask, dtype=bool)
    face_mask[:, 0, :] = clean_mask[:, 0, :]
    coords = np.where(face_mask)
    untouch_x.extend(coords[0].tolist())
    untouch_y.extend(coords[1].tolist())
    untouch_z.extend(coords[2].tolist())

    # y=95 面
    face_mask = np.zeros_like(clean_mask, dtype=bool)
    face_mask[:, 95, :] = clean_mask[:, 95, :]
    coords = np.where(face_mask)
    untouch_x.extend(coords[0].tolist())
    untouch_y.extend(coords[1].tolist())
    untouch_z.extend(coords[2].tolist())

    # z=0 面
    face_mask = np.zeros_like(clean_mask, dtype=bool)
    face_mask[:, :, 0] = clean_mask[:, :, 0]
    coords = np.where(face_mask)
    untouch_x.extend(coords[0].tolist())
    untouch_y.extend(coords[1].tolist())
    untouch_z.extend(coords[2].tolist())

    # z=95 面
    face_mask = np.zeros_like(clean_mask, dtype=bool)
    face_mask[:, :, 95] = clean_mask[:, :, 95]
    coords = np.where(face_mask)
    untouch_x.extend(coords[0].tolist())
    untouch_y.extend(coords[1].tolist())
    untouch_z.extend(coords[2].tolist())

    # x=0 面
    face_mask = np.zeros_like(clean_mask, dtype=bool)
    face_mask[0, :, :] = clean_mask[0, :, :]
    coords = np.where(face_mask)
    untouch_x.extend(coords[0].tolist())
    untouch_y.extend(coords[1].tolist())
    untouch_z.extend(coords[2].tolist())

    # x=95 面
    face_mask = np.zeros_like(clean_mask, dtype=bool)
    face_mask[95, :, :] = clean_mask[95, :, :]
    coords = np.where(face_mask)
    untouch_x.extend(coords[0].tolist())
    untouch_y.extend(coords[1].tolist())
    untouch_z.extend(coords[2].tolist())

    if len(untouch_x) == 0:
        # 如果没有找到任何参照点（极端情况），使用角落的脑内体素
        return np.array([0]), np.array([0]), np.array([0])

    return np.array(untouch_x), np.array(untouch_y), np.array(untouch_z)


def linear_interpolation_vec(final_recons, healthy_scan_crop,
                              untouch_x_axis, untouch_y_axis, untouch_z_axis):
    """
    向量化版本：用线性回归校正生成图像的强度值，使其与周围脑组织匹配。

    原理：在参照点（脑内非肿瘤区域）上拟合一条线性回归线，
    将生成图像的强度值映射到原始扫描的强度值。

    原函数位置: main_random_label_random_dataset_generator_multiprocess.py:219
    原函数使用 for 循环做多项式拟合和像素级修正，这里用 numpy 向量化。

    参数:
        final_recons:      (96,96,96) 扩散模型生成的肿瘤扫描
        healthy_scan_crop: (96,96,96) 原始健康扫描
        untouch_x_axis:    参照点 x 坐标
        untouch_y_axis:    参照点 y 坐标
        untouch_z_axis:    参照点 z 坐标
    返回:
        np.ndarray: 强度校正后的 (96,96,96) 数组
    """
    # 添加锚点：(-1, 0) 表示背景值为 -1 时对应的原始值应为 0
    x_vals = [-1.0]
    y_vals = [0.0]

    # 在参照点收集原始扫描强度值 (y) 和生成图像强度值 (x)
    y_vals.extend(healthy_scan_crop[untouch_x_axis, untouch_y_axis, untouch_z_axis].tolist())
    x_vals.extend(final_recons[untouch_x_axis, untouch_y_axis, untouch_z_axis].tolist())

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # 线性回归：y = m * x + b
    coefficients = np.polyfit(x_vals, y_vals, 1)
    best_m, best_b = coefficients

    # 对整个生成图像应用线性校正
    inten_scan = best_m * final_recons + best_b

    # 脑外区域和负值区域 → 设为 0
    inten_scan[healthy_scan_crop == 0] = 0
    inten_scan[inten_scan < 0] = 0

    return inten_scan


def add_gaussian_noise_tumour_vec(scan, label):
    """
    向量化版本：在肿瘤区域添加高斯噪声，然后将扫描重缩放到 [-1, 1]。

    原函数位置: main_random_label_random_dataset_generator_multiprocess.py:165
    原函数使用三重 for 循环遍历 96³ 体素并逐个调用 torch.randn，
    这里用 numpy 向量化替代（np.random.randn 一次性生成所有噪声）。

    参数:
        scan:  (96,96,96) numpy 数组，归一化到 [-1,1] 的健康扫描
        label: (96,96,96) numpy 数组，整型标签（非零=肿瘤区域）
    返回:
        scan_noisy: (96,96,96) 肿瘤区域被高斯噪声替换后的扫描
        noise:      (96,96,96) 噪声掩码（肿瘤区域=1，其他=0）
    """
    scan_noisy = np.copy(scan)

    # 在肿瘤区域（label != 0）生成高斯噪声，其他区域为 0
    tumour_mask = (label != 0)
    noise = np.zeros_like(scan, dtype=np.float32)
    noise[tumour_mask] = np.random.randn(np.sum(tumour_mask)).astype(np.float32)

    # 将噪声复制到扫描中（非背景区域的肿瘤位置）
    copy_mask = (noise != 0) & (scan_noisy != -1)
    scan_noisy[copy_mask] = noise[copy_mask]

    # 重缩放到 [-1, 1]（与原函数 rescale_gaussian_noise 一致）
    # 找出 scan_noisy 中的次大值（排除 1000 标记值），但向量化版本中不再使用 1000 标记
    # 直接用非噪声区域的值域进行 min-max 归一化
    non_tumour_vals = scan_noisy[~tumour_mask]
    if len(non_tumour_vals) > 0:
        mina = non_tumour_vals.min()
        maxa = non_tumour_vals.max()
    else:
        mina = scan_noisy.min()
        maxa = scan_noisy.max()
    if maxa > mina:
        scan_noisy = (scan_noisy - mina) / (maxa - mina) * 2.0 - 1.0

    # 构建噪声掩码（肿瘤区域=1，用于 get_inten_coord 查找参照点）
    noise_mask = tumour_mask.astype(np.float32)

    return scan_noisy, noise_mask


def load_diffusion_model(args, model_path, network_type="tumour"):
    """Load a trained diffusion model checkpoint.

    Returns:
        model, n_steps, noise_schedule (str), schedule_config (dict or None)
    """
    if network_type == "tumour":
        from src.networks.DiffusionNetwork import get_diffusion_network

        model = get_diffusion_network(args)
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(ckpt["state_dict"])
        n_steps = ckpt.get("n_steps", 1000)
        noise_schedule = ckpt.get("noise_schedule", ckpt.get("beta_schedule", "cosine"))
        schedule_config = ckpt.get("schedule_config", None)
    elif network_type == "label":
        from src.networks.DiffusionNetwork import LabelDenoiser3D

        noise_mode = getattr(args, "noise_embedding_mode", "discrete")
        model = LabelDenoiser3D(in_channels=args.out_channels_label,
                                noise_embedding_mode=noise_mode)
        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(ckpt["state_dict"])
        n_steps = ckpt.get("n_steps", 1000)
        noise_schedule = ckpt.get("noise_schedule", ckpt.get("beta_schedule", "cosine"))
        schedule_config = ckpt.get("schedule_config", None)
    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    model.eval()
    return model, n_steps, noise_schedule, schedule_config
