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
def diffusion_loss_fn(model,batch_y, batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps,device):
    batch_size=batch_y.shape[0]
    t=torch.randint(0,n_steps,size=(batch_size,))
    t=t.unsqueeze(-1)
    a=alphas_bar_sqrt[t].to(device).view(-1,1,1,1,1)
    aml=one_minus_alphas_bar_sqrt[t].to(device).view(-1,1,1,1,1)
    e=torch.randn_like(batch_y).to(device)
    y_t=batch_y*a+aml*e
    output=model(y_t,t.squeeze(-1).to(device),batch_x)
    return (e-output).square().mean()

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
                 device, sampling_steps=50, eta=0.0):
    """
    DDIM 加速采样循环 — 在子序列时间步上迭代，大幅减少采样步数。

    参数:
        alphas_bar:  累积 α 乘积 ᾱ_t, shape [n_steps]（非 sqrt 版本）
        sampling_steps: 子序列长度，默认 50 步
        eta:         0.0=确定性, >0 引入随机性
    """
    traj = []
    timesteps = _get_ddim_timesteps(num_steps, sampling_steps)
    x_t = torch.randn(num_samples, input_dim).to(device)
    traj.append(x_t)

    for i, t in enumerate(timesteps):
        t_tensor  = torch.full((num_samples,), t, device=device, dtype=torch.long)
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
                     method='ddpm', sampling_steps=None, eta=0.0):
    """
    扩散模型反向采样入口。

    参数:
        method:         'ddpm' — 原始 DDPM 采样（默认，1000 步）
                        'ddim' — DDIM 加速采样（确定性/随机，可指定步数）
        sampling_steps: DDIM 模式的子序列步数（None 则使用 num_steps）
        eta:            DDIM 随机性系数（0=确定性, 0<η≤1 随机; method='ddpm' 时忽略）
        （其余参数与原 sample_from_diff 完全一致）
    """
    # ======================== DDPM 分支 — 原逻辑一字未改 ========================
    if method == 'ddpm':
        traj = []
        x_t = torch.randn(num_samples, input_dim).to(device)
        traj.append(x_t)
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
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

    # ======================== DDIM 分支 — 新增加速采样 ========================
    elif method == 'ddim':
        steps = sampling_steps if sampling_steps is not None else num_steps
        # DDIM 需要完整 alphas_bar 序列；从 betas 即时计算，无需调用方额外传入
        alphas = 1.0 - betas
        alphas_bar_full = torch.cumprod(alphas, dim=0)
        return _sample_ddim(model, num_samples, input_dim, cond,
                            alphas_bar_full, num_steps, device, steps, eta)

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'ddpm' or 'ddim'.")

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
