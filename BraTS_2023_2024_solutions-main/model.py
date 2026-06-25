# import
import math
import torch
import torch.nn as nn
import os
import numpy as np
import random
from copy import deepcopy
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
from scipy import stats
from scipy.stats import rankdata
from dataclasses import dataclass
# from MINE import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# diffusion_loss_fn is used when sampling_model='ddpm' in diffusion_crt.py
def diffusion_loss_fn(model, batch_y, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                      n_steps, device, p_uncond=0.0):
    batch_size = batch_y.shape[0]
    t = torch.randint(0, n_steps, size=(batch_size,))
    t = t.unsqueeze(-1)
    a = alphas_bar_sqrt[t].to(device).view(-1, 1, 1, 1, 1)
    aml = one_minus_alphas_bar_sqrt[t].to(device).view(-1, 1, 1, 1, 1)
    e = torch.randn_like(batch_y).to(device)
    y_t = batch_y * a + aml * e
    # CFG: randomly drop condition
    cond = batch_x
    if p_uncond > 0.0 and cond is not None:
        mask = (torch.rand(batch_size, device=device) >= p_uncond).float()
        cond = cond * mask.view(-1, 1, 1, 1, 1)
    output = model(y_t, t.squeeze(-1).to(device), cond)
    return (e - output).square().mean()

# score_loss_fn is used when sampling_model='score' in diffusion_crt.py
def score_loss_fn(model, y_0, x, n_steps,device):
    batch_size = y_0.shape[0]
    t = torch.randint(1, n_steps, size=(batch_size,), device=device) # t_min=1/n_step
    t_input = t.unsqueeze(-1)  # priorly set T=1000/100=10
    t_compute=t.unsqueeze(-1) /100
    epsilon=torch.normal(0,1,(batch_size,1)).to(device)
    y_input=torch.exp(-t_compute/2)*y_0+torch.sqrt(1-torch.exp(-t_compute))*epsilon
    output = model(x, y_input,  t_input.squeeze(-1))
    loss=(output+(y_input-torch.exp(-t_compute/2)*y_0)/(1-torch.exp(-t_compute))).square().mean()
    return loss
    

# make_beta_schedule is used when sampling_model='ddpm' in diffusion_crt.py
def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-4, end=2e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas




# ===========================================================================
# DDIM: 从总步数中均匀抽取子序列时间步，用于加速采样
# 例如 n_steps=1000, sampling_steps=20 → [950, 900, 850, ..., 50, 0]
# ===========================================================================
def _get_ddim_timesteps(n_steps, sampling_steps):
    """从 n_steps 中均匀抽取 sampling_steps 个时间步，返回倒序列表。"""
    step = n_steps // sampling_steps
    timesteps = list(range(0, n_steps, step))          # 正序: [0, 50, 100, ..., 950]
    return timesteps[::-1]                              # 倒序: [950, ..., 100, 50, 0]


# ===========================================================================
# DDIM: 单步更新（Song et al. 2021, Equation 12）
#
# x̂_0 = (x_t − √(1−ᾱ_t) · ε_θ) / √(ᾱ_t)
# σ_t = η · √((1−ᾱ_{t−1})/(1−ᾱ_t)) · √(1 − ᾱ_t/ᾱ_{t−1})
# x_{t−1} = √(ᾱ_{t−1}) · x̂_0 + √(1−ᾱ_{t−1} − σ_t²) · ε_θ + σ_t · z
#
# eta=0 → 确定性 DDIM
# eta=1 → 退化为 DDPM
# ===========================================================================
def _ddim_step(x_t, t, t_prev, noise_pred, alphas_bar, eta, device):
    """
    DDIM 单步去噪：从 x_t (时刻 t) 一步跳到 x_{t_prev} (时刻 t_prev)。

    参数:
        x_t:         当前时刻的样本张量
        t, t_prev:   当前/目标时间步索引（t_prev < t, t_prev=-1 表示最后一步）
        noise_pred:  模型预测的噪声 ε_θ(x_t, t)
        alphas_bar:  累积 α 乘积序列 ᾱ_t, shape [n_steps]
        eta:         0.0=确定性 DDIM, 1.0≈退化为 DDPM
        device:      计算设备
    """
    alpha_bar_t     = alphas_bar[t].to(device)
    alpha_bar_prev  = alphas_bar[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

    # 预测 x_0
    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t + 1e-8)

    # 随机性系数 σ_t — 完全由 η 控制
    sigma = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)) * \
            torch.sqrt(1.0 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-8))

    # DDIM 采样
    direction  = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)) * noise_pred
    noise_term = sigma * torch.randn_like(x_t) if eta > 1e-8 else 0.0

    return torch.sqrt(alpha_bar_prev) * x0_pred + direction + noise_term


# ===========================================================================
# DDIM: 完整加速采样循环
# ===========================================================================
def _sample_ddim(model, num_samples, input_dim, cond, alphas_bar, num_steps,
                 device, sampling_steps=50, eta=0.0, cfg_weight=1.0, zero_cond=None):
    """
    DDIM 加速采样循环 — 在子序列时间步上迭代，大幅减少采样步数。

    参数:
        alphas_bar:  累积 α 乘积 ᾱ_t, shape [n_steps]（非 sqrt 版本）
        sampling_steps: 子序列长度，默认 50 步
        eta:         0.0=确定性, >0 引入随机性
        cfg_weight:  CFG 权重 (1.0=正常)
        zero_cond:   预计算的全零条件 tensor (CFG 用)
    """
    traj = []
    timesteps = _get_ddim_timesteps(num_steps, sampling_steps)
    x_t = torch.randn(num_samples, input_dim).to(device)
    traj.append(x_t)

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
        if zero_cond is not None:
            eps_uncond = model(x_t, t_tensor, zero_cond)
            eps_cond = model(x_t, t_tensor, cond)
            noise_pred = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
        else:
            noise_pred = model(x_t, t_tensor, cond)

        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        x_t = _ddim_step(x_t, t, t_prev, noise_pred, alphas_bar, eta, device)
        traj.append(x_t)

    return traj


# ===========================================================================
# 统一的扩散采样入口 — 支持 DDPM / DDIM 两种采样算法
# 原 DDPM 逻辑完整保留，DDIM 为纯增量代码
# ===========================================================================
def sample_from_diff(model, num_samples, input_dim, cond,
                     alphas_bar_sqrt, one_minus_alphas_bar_sqrt, betas,
                     num_steps, device,
                     method='ddpm', sampling_steps=None, eta=0.0, cfg_weight=1.0):
    """
    扩散模型反向采样入口。

    参数:
        method:         'ddpm' — 原始 DDPM 采样（默认，1000 步）
                        'ddim' — DDIM 加速采样（确定性/随机，可指定步数）
        sampling_steps: DDIM 模式的子序列步数（None 则使用 num_steps）
        eta:            DDIM 随机性系数（0=确定性, 0<η≤1 随机; method='ddpm' 时忽略）
        cfg_weight:     CFG 权重 (1.0=正常, >1=增强条件约束)
    """
    # Precompute zero condition once for CFG
    zero_cond = torch.zeros_like(cond) if (cfg_weight != 1.0 and cond is not None) else None

    # ======================== DDPM 分支 ========================
    if method == 'ddpm':
        traj = []
        x_t = torch.randn(num_samples, input_dim).to(device)
        traj.append(x_t)
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            if zero_cond is not None:
                eps_uncond = model(x_t, t_tensor, zero_cond)
                eps_cond = model(x_t, t_tensor, cond)
                noise_pred = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
            else:
                noise_pred = model(x_t, t_tensor, cond)
            if t > 0:
                beta_t = betas[t]
                alpha_t = 1 - beta_t
                alpha_t_sqrt = torch.sqrt(alpha_t)
                noise = torch.randn_like(x_t).to(device)
                x_t = (1 / alpha_t_sqrt) * (x_t - (1 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred) + torch.sqrt(beta_t) * noise
                traj.append(x_t)
            else:
                alpha_t = 1 - betas[t]
                alpha_t_sqrt = torch.sqrt(alpha_t)
                x_t = (1 / alpha_t_sqrt) * (x_t - (1 - alpha_t) / one_minus_alphas_bar_sqrt[t] * noise_pred)
                traj.append(x_t)
        return traj

    # ======================== DDIM 分支 ========================
    elif method == 'ddim':
        steps = sampling_steps if sampling_steps is not None else num_steps
        alphas = 1.0 - betas
        alphas_bar_full = torch.cumprod(alphas, dim=0)
        return _sample_ddim(model, num_samples, input_dim, cond,
                            alphas_bar_full, num_steps, device, steps, eta,
                            cfg_weight, zero_cond)

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'ddpm' or 'ddim'.")

# ===========================================================================
# Unified noise-schedule factory: EDM / logsnr / legacy beta schedules
# ===========================================================================

@dataclass
class NoiseScheduleConfig:
    """Unified container returned by make_noise_schedule."""
    name: str                     # 'edm' | 'lognsr' | 'cosine' | 'linear' | ...
    # Legacy beta-schedule fields (None for edm/lognsr)
    betas: torch.Tensor = None
    alphas_bar_sqrt: torch.Tensor = None
    one_minus_alphas_bar_sqrt: torch.Tensor = None
    alphas_bar: torch.Tensor = None
    # EDM fields
    sigma_data: float = None
    sigma_min: float = None
    sigma_max: float = None
    rho: float = None
    # logsnr fields
    gamma_max: float = None
    gamma_min: float = None
    # Global SNR shift: positive → higher SNR → easier task (0=off, default)
    snr_shift: float = 0.0
    # Total steps (may differ from n_steps for edm/lognsr which are continuous)
    n_steps: int = 1000


def make_noise_schedule(
    schedule="cosine",
    n_steps=1000,
    # Legacy beta params
    beta_start=1e-4,
    beta_end=2e-2,
    # EDM params
    sigma_data=0.5,
    sigma_max=80.0,
    sigma_min=0.002,
    rho=7.0,
    # logsnr params
    gamma_max=10.0,
    gamma_min=-10.0,
    # Global SNR shift (applies to edm / logsnr)
    snr_shift=0.0,
):
    """
    Unified factory: returns NoiseScheduleConfig for any schedule type.

    Schedule types:
      Legacy beta: 'linear', 'const', 'quad', 'jsd', 'sigmoid',
                    'cosine','cosine_reverse', 'cosine_anneal'
      EDM:          'edm'
      logsnr:       'lognsr'
    """
    if schedule in ("edm", "lognsr"):
        # ---- Continuous-noise-family: betas / alphas not applicable ----
        if schedule == "edm":
            return NoiseScheduleConfig(
                name="edm", n_steps=n_steps,
                sigma_data=sigma_data, sigma_min=sigma_min,
                sigma_max=sigma_max, rho=rho,
                snr_shift=snr_shift,
            )
        else:
            return NoiseScheduleConfig(
                name="lognsr", n_steps=n_steps,
                gamma_max=gamma_max, gamma_min=gamma_min,
                snr_shift=snr_shift,
            )
    else:
        # ---- Legacy beta-family (unchanged) ----
        betas = make_beta_schedule(schedule=schedule, num_timesteps=n_steps,
                                    start=beta_start, end=beta_end)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = torch.sqrt(alphas_bar)
        one_minus_alphas_bar_sqrt = torch.sqrt(1.0 - alphas_bar)
        return NoiseScheduleConfig(
            name=schedule, n_steps=n_steps,
            betas=betas, alphas_bar_sqrt=alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt,
            alphas_bar=alphas_bar,
        )


# ===========================================================================
# EDM loss (Karras et al. 2022) — preconditioned denoising
# ===========================================================================

def edm_loss_fn(model, x_0, cond, schedule_cfg: NoiseScheduleConfig, device,
                P_mean=-1.2, P_std=1.2, p_uncond=0.0):
    """
    EDM preconditioned denoising loss.

    Samples σ ~ log-normal(P_mean, P_std), then:
      x_noisy = x_0 + σ · ε
      D_θ(x_noisy, σ, cond) predicts clean x_0
      λ(σ)·MSE with EDM preconditioning weights.

    Args:
        model:    network with forward(x, t, cond); t = log(σ) as float
        x_0:      clean target [B, C, D, H, W]
        cond:     condition tensor [B, C_cond, D, H, W]
        schedule_cfg: NoiseScheduleConfig with sigma_data
        device:   torch device
        P_mean, P_std: log-normal sampling params (EDM defaults -1.2, 1.2)
    Returns:
        scalar loss
    """
    batch_size = x_0.shape[0]
    sigma_data = schedule_cfg.sigma_data

    # Sample σ from log-normal distribution (EDM Table 1)
    ln_sigma = torch.randn(batch_size, device=device) * P_std + P_mean
    # Apply SNR shift: positive shift → lower σ → higher SNR → easier task
    if schedule_cfg.snr_shift != 0.0:
        ln_sigma = ln_sigma - schedule_cfg.snr_shift
    sigma = torch.exp(ln_sigma)  # [B]
    sigma = sigma.view(-1, 1, 1, 1, 1)

    # Add noise
    noise = torch.randn_like(x_0)
    x_noisy = x_0 + sigma * noise

    # Preconditioned target: network predicts clean x_0 from noisy input
    # c_skip · x_noisy + c_out · F(c_in · x_noisy, log_sigma, cond) → x_0
    # Equivalent to: predict (x_0 - c_skip·x_noisy)/c_out = predict noise rescaled
    # We let model predict the raw denoised output; loss is on the equivalent
    # formulation of EDM Eq. 7: effective target derived from c_in/c_out/c_skip.

    # EDM preconditioning coefficients (Eq. 8-11, simplified for denoising)
    sigma_flat = sigma.view(-1)
    c_in = 1.0 / torch.sqrt(sigma_flat ** 2 + sigma_data ** 2)
    c_out = sigma_flat * sigma_data / torch.sqrt(sigma_flat ** 2 + sigma_data ** 2)
    c_skip = sigma_data ** 2 / (sigma_flat ** 2 + sigma_data ** 2)

    # Precondition input
    x_in = c_in.view(-1, 1, 1, 1, 1) * x_noisy

    # Model sees log(σ) as noise level (continuous float)
    t_input = ln_sigma  # [B] — continuous log-sigma

    # CFG: randomly drop condition
    cond_dropped = cond
    if p_uncond > 0.0 and cond is not None:
        mask = (torch.rand(batch_size, device=device) >= p_uncond).float()
        cond_dropped = cond * mask.view(-1, 1, 1, 1, 1)
    # Model forward (supports unconditional: cond=None)
    if cond_dropped is not None:
        pred = model(x_in, t_input, cond_dropped)
    else:
        pred = model(x_in, t_input)

    # Assemble denoised prediction
    denoised = c_skip.view(-1, 1, 1, 1, 1) * x_noisy + c_out.view(-1, 1, 1, 1, 1) * pred

    # Effective loss weight λ(σ) = (σ² + σ_data²) / (σ·σ_data)² = 1/c_out²
    weight = 1.0 / (c_out ** 2 + 1e-8)
    weight = weight.view(-1, 1, 1, 1, 1)

    loss = (weight * (denoised - x_0) ** 2).mean()
    return loss


# ===========================================================================
# logsnr loss (Kingma 2021) — continuous-time SNR-parameterised diffusion
# ===========================================================================

def logsnr_loss_fn(model, x_0, cond, schedule_cfg: NoiseScheduleConfig, device,
                   p_uncond=0.0):
    """
    Continuous-time diffusion loss parameterised by log-SNR.

    Samples t ~ Uniform(0, 1), maps to γ(t) = γ_max + (γ_min - γ_max) * t,
    then noisifies and denoises.

    Args:
        model:        network, forward(x, t, cond); t = γ (log-SNR float)
        x_0:          clean target [B, C, D, H, W]
        cond:         condition [B, C_cond, D, H, W]
        schedule_cfg: NoiseScheduleConfig with gamma_max, gamma_min
        device:       torch device
    Returns:
        scalar loss
    """
    batch_size = x_0.shape[0]
    gamma_max = schedule_cfg.gamma_max
    gamma_min = schedule_cfg.gamma_min

    # Sample t uniform, map to log-SNR
    t = torch.rand(batch_size, device=device)
    gamma_t = gamma_max + (gamma_min - gamma_max) * t  # [B]
    # Apply SNR shift: positive → higher γ → higher SNR → easier
    if schedule_cfg.snr_shift != 0.0:
        gamma_t = gamma_t + schedule_cfg.snr_shift

    # Derive ᾱ, σ from log-SNR:  SNR = ᾱ / (1-ᾱ) = exp(γ)
    #  → ᾱ = exp(γ) / (1 + exp(γ)) = sigmoid(γ)
    #  → 1-ᾱ = 1 / (1 + exp(γ)) = sigmoid(-γ)
    alpha_bar_t = torch.sigmoid(gamma_t)  # [B]
    sigma_t = torch.sqrt(1.0 - alpha_bar_t)

    # Noisify: x_t = √ᾱ·x_0 + √(1-ᾱ)·ε
    a_sqrt = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1, 1)
    s = sigma_t.view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(x_0)
    x_noisy = a_sqrt * x_0 + s * noise

    # CFG: randomly drop condition
    cond_dropped = cond
    if p_uncond > 0.0 and cond is not None:
        mask = (torch.rand(batch_size, device=device) >= p_uncond).float()
        cond_dropped = cond * mask.view(-1, 1, 1, 1, 1)
    # Model: predict the noise component ε (supports unconditional: cond=None)
    t_input = gamma_t  # [B] — continuous log-SNR as float
    if cond_dropped is not None:
        noise_pred = model(x_noisy, t_input, cond_dropped)
    else:
        noise_pred = model(x_noisy, t_input)

    loss = ((noise_pred - noise) ** 2).mean()
    return loss


# ===========================================================================
# EDM sampling — Heun's 2nd order ODE (Karras et al. 2022, Algorithm 1)
# ===========================================================================

@torch.no_grad()
def sample_edm(model, shape, cond, schedule_cfg: NoiseScheduleConfig, device,
               num_steps=18, S_churn=0.0, S_min=0.0, S_max=float('inf'),
               S_noise=1.0, cfg_weight=1.0):
    """
    EDM Heun 2nd-order ODE sampler with optional Langevin SDE churn.

    Args:
        model:         network forward(x, log_sigma, cond) → F_θ
        shape:         (C, D, H, W) output shape (batch dim auto-added)
        cond:          condition tensor [B, C_cond, D, H, W]
        schedule_cfg:  NoiseScheduleConfig (edm mode)
        device:        torch device
        num_steps:     number of ODE steps (default 18, fewer=faster)
        S_churn:       SDE churn magnitude (0 = pure ODE, >0 adds stochasticity)
        S_min, S_max:  churn clamping
        S_noise:       churn noise scale
    Returns:
        denoised output [B, C, D, H, W]
    """
    if cond is not None:
        batch_size = cond.shape[0]
    else:
        batch_size = 1  # unconditional: single sample, caller controls batching

    sigma_data = schedule_cfg.sigma_data
    sigma_max = schedule_cfg.sigma_max
    sigma_min = schedule_cfg.sigma_min
    rho = schedule_cfg.rho

    # Time step discretisation (EDM Eq. 5)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (sigma_max ** (1 / rho)
               + step_indices / (num_steps - 1)
               * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros([1], device=device)])  # append σ=0

    # Initial noise
    x_cur = torch.randn(batch_size, *shape, device=device) * t_steps[0]

    for i in range(num_steps):
        sigma_cur = t_steps[i]
        sigma_next = t_steps[i + 1]

        # Optional SDE churn (EDM Eq. 21-22)
        gamma = 0.0
        if S_churn > 0 and sigma_cur >= S_min and sigma_cur <= S_max:
            gamma = min(S_churn / num_steps, math.sqrt(2) - 1)
        sigma_hat = sigma_cur + gamma * sigma_cur
        if gamma > 0:
            x_cur = x_cur + math.sqrt(sigma_hat ** 2 - sigma_cur ** 2) * S_noise * torch.randn_like(x_cur)

        # Euler step
        sigma_tensor = torch.full((batch_size,), sigma_hat.item(), device=device)
        log_sigma = torch.log(sigma_tensor)
        # Apply SNR shift so model sees the same noise-level shift as training
        if schedule_cfg.snr_shift != 0.0:
            log_sigma = log_sigma - schedule_cfg.snr_shift

        # EDM preconditioning
        c_in_val = 1.0 / math.sqrt(sigma_hat ** 2 + sigma_data ** 2)
        c_out_val = sigma_hat * sigma_data / math.sqrt(sigma_hat ** 2 + sigma_data ** 2)
        c_skip_val = sigma_data ** 2 / (sigma_hat ** 2 + sigma_data ** 2)

        x_in = c_in_val * x_cur
        if cfg_weight != 1.0 and cond is not None:
            zero_cond = torch.zeros_like(cond)
            pred_uncond = model(x_in, log_sigma, zero_cond)
            pred_cond = model(x_in, log_sigma, cond)
            F_theta = pred_uncond + cfg_weight * (pred_cond - pred_uncond)
        elif cond is not None:
            F_theta = model(x_in, log_sigma, cond)
        else:
            F_theta = model(x_in, log_sigma)
        denoised = c_skip_val * x_cur + c_out_val * F_theta

        # ODE direction: dx/dσ = (x - D_θ(x,σ)) / σ
        d_cur = (x_cur - denoised) / sigma_hat
        x_next_euler = x_cur + (sigma_next - sigma_hat) * d_cur

        # Heun 2nd order correction (skip on last step)
        if i < num_steps - 1:
            sigma_tensor_next = torch.full((batch_size,), sigma_next.item(), device=device)
            log_sigma_next = torch.log(sigma_tensor_next)
            if schedule_cfg.snr_shift != 0.0:
                log_sigma_next = log_sigma_next - schedule_cfg.snr_shift

            c_in_next_val = 1.0 / math.sqrt(sigma_next ** 2 + sigma_data ** 2)
            c_out_next_val = sigma_next * sigma_data / math.sqrt(sigma_next ** 2 + sigma_data ** 2)
            c_skip_next_val = sigma_data ** 2 / (sigma_next ** 2 + sigma_data ** 2)

            x_in_next = c_in_next_val * x_next_euler
            if cfg_weight != 1.0 and cond is not None:
                zero_cond = torch.zeros_like(cond)
                pred_uncond_next = model(x_in_next, log_sigma_next, zero_cond)
                pred_cond_next = model(x_in_next, log_sigma_next, cond)
                F_theta_next = pred_uncond_next + cfg_weight * (pred_cond_next - pred_uncond_next)
            elif cond is not None:
                F_theta_next = model(x_in_next, log_sigma_next, cond)
            else:
                F_theta_next = model(x_in_next, log_sigma_next)
            denoised_next = c_skip_next_val * x_next_euler + c_out_next_val * F_theta_next
            d_next = (x_next_euler - denoised_next) / sigma_next

            # Trapezoidal correction
            x_cur = x_cur + (sigma_next - sigma_hat) * 0.5 * (d_cur + d_next)
        else:
            x_cur = x_next_euler

    return x_cur


# ===========================================================================
# logsnr sampling — ODE in log-SNR space
# ===========================================================================

@torch.no_grad()
def sample_lognsr(model, shape, cond, schedule_cfg: NoiseScheduleConfig, device,
                  num_steps=50, eta=0.0, cfg_weight=1.0):
    """
    ODE sampler in continuous log-SNR space.

    Uniformly discretises γ from γ_min to γ_max into num_steps, then
    integrates the ODE:
      dx/dγ = -(1/2) · exp(-γ) · (x - D_θ(x, γ))

    Args:
        model:        network forward(x, gamma, cond) → ε_pred
        shape:        (C, D, H, W) output shape
        cond:         condition [B, C_cond, D, H, W]
        schedule_cfg: NoiseScheduleConfig (lognsr mode)
        device:       torch device
        num_steps:    number of ODE steps
        eta:          SDE stochasticity (0=deterministic ODE, 1≈SDE)
    Returns:
        denoised output [B, C, D, H, W]
    """
    if cond is not None:
        batch_size = cond.shape[0]
    else:
        batch_size = 1
    gamma_max = schedule_cfg.gamma_max
    gamma_min = schedule_cfg.gamma_min

    # Ascend γ from γ_min (noisy) to γ_max (clean)
    gamma_seq = torch.linspace(gamma_min, gamma_max, num_steps + 1, device=device)
    init_sigma = torch.sqrt(1.0 / (1.0 + torch.exp(gamma_min)))
    x_cur = init_sigma * torch.randn(batch_size, *shape, device=device)

    for i in range(num_steps):
        gamma_s = gamma_seq[i]       # current
        gamma_t = gamma_seq[i + 1]   # next (cleaner, higher SNR)

        gamma_tensor = torch.full((batch_size,), gamma_s, device=device)
        # Apply SNR shift so model sees the same shifted noise level as training
        if schedule_cfg.snr_shift != 0.0:
            gamma_tensor = gamma_tensor + schedule_cfg.snr_shift
        if cfg_weight != 1.0 and cond is not None:
            zero_cond = torch.zeros_like(cond)
            eps_uncond = model(x_cur, gamma_tensor, zero_cond)
            eps_cond = model(x_cur, gamma_tensor, cond)
            noise_pred = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
        elif cond is not None:
            noise_pred = model(x_cur, gamma_tensor, cond)
        else:
            noise_pred = model(x_cur, gamma_tensor)

        # Derive ᾱ from γ: ᾱ = sigmoid(γ), σ = √(1-ᾱ)
        alpha_bar_s = torch.sigmoid(gamma_s)
        sigma_s = torch.sqrt(1.0 - alpha_bar_s)
        alpha_bar_t = torch.sigmoid(gamma_t)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t)

        # DDIM-like ODE step in log-SNR space
        # x̂_0 = (x - σ·ε_θ) / √ᾱ
        x0_pred = (x_cur - sigma_s * noise_pred) / torch.sqrt(alpha_bar_s)

        # Deterministic step toward cleaner state
        x_cur = torch.sqrt(alpha_bar_t) * x0_pred + sigma_t * noise_pred

        # Optional SDE stochasticity
        if eta > 1e-8:
            sigma_t_noise = eta * torch.sqrt(sigma_t ** 2 * (1.0 - alpha_bar_s) / (alpha_bar_t * sigma_s ** 2 + 1e-8))
            noise = sigma_t_noise * torch.randn_like(x_cur)
            x_cur = x_cur + noise

    return x_cur


# ===========================================================================
# argparse helper for noise-schedule arguments
# ===========================================================================

def add_noise_schedule_args(parser):
    """Add EDM / logsnr / legacy schedule arguments to an argparse parser."""
    g = parser.add_argument_group("Noise schedule (β / EDM / logsnr)")
    g.add_argument("--noise_schedule", default="cosine", type=str,
                   choices=["linear", "const", "quad", "jsd", "sigmoid",
                            "cosine", "cosine_reverse", "cosine_anneal",
                            "edm", "lognsr"],
                   help="Noise schedule type")
    # Legacy beta params
    g.add_argument("--beta_start", default=1e-4, type=float,
                   help="Beta start (for legacy schedules)")
    g.add_argument("--beta_end", default=2e-2, type=float,
                   help="Beta end (for legacy schedules)")
    # EDM params
    g.add_argument("--sigma_data", default=0.5, type=float,
                   help="EDM: data pixel std (for preconditioning)")
    g.add_argument("--sigma_max", default=80.0, type=float,
                   help="EDM: max noise std")
    g.add_argument("--sigma_min", default=0.002, type=float,
                   help="EDM: min noise std")
    g.add_argument("--rho", default=7.0, type=float,
                   help="EDM: time-allocation curvature")
    # logsnr params
    g.add_argument("--gamma_max", default=10.0, type=float,
                   help="logsnr: log-SNR at clean state")
    g.add_argument("--gamma_min", default=-10.0, type=float,
                   help="logsnr: log-SNR at pure-noise state")
    # Global SNR shift (applies to edm and logsnr)
    g.add_argument("--snr_shift", default=0.0, type=float,
                   help="SNR shift: positive=easi er (higher SNR), negative=harder (0=off)")


# score based generative model like sampling
def score_sampler(model,shape,x, device,sample_steps=1000,):
    model.eval()
    tK_mins_t0=(1000-1)/100 
    delta_t=torch.tensor([tK_mins_t0/sample_steps],device=device) 
    y_k=torch.normal(0,1,(shape)).to(device)
    with torch.no_grad():
        for t in range(1,sample_steps):
            eps=torch.normal(0,(tK_mins_t0/sample_steps)**0.5,(shape)).to(device)
            t_input=torch.tensor([sample_steps-t],device=device)
            y_k=y_k+eps+delta_t*(0.5*y_k+model(x,y_k,t_input)) # t_input from 999 to 1
    return y_k

# ema
class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(),
                                      model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model,
                     update_fn=lambda e, m: self.decay * e +
                                            (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




# pearson corr
def correlation(X,Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return np.abs(np.corrcoef(X, Y)[0, 1])


def kolmogorov(X,Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return ks_2samp(X, Y)[0]

def wilcox(X,Y):

    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return wilcoxon(X, Y)[0]


def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    
    Source: https://github.com/garydoranjr/rdc
    """
    x = x.reshape((len(x)))
    y = y.reshape((len(y)))
    
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))
