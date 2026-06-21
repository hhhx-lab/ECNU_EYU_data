import torch
import torch.nn as nn
import math
from monai.networks.nets import SwinUNETR, AttentionUnet, UNet


# ===========================================================================
# nnU-Net 理念工具
# ===========================================================================

def plan_network_params(input_size=96, gpu_memory_gb=None):
    """
    根据输入尺寸和 GPU 显存自动推网络深度/宽度，不改变 backbone 结构。
    返回 (channels, strides, batch_size_suggestion)。
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            gpu_memory_gb = 8

    num_stages = max(1, int(math.log2(input_size / 4)))

    if gpu_memory_gb >= 32:
        base, cap = 48, 768
    elif gpu_memory_gb >= 16:
        base, cap = 32, 512
    else:
        base, cap = 24, 320

    channels = tuple(min(base * (2 ** i), cap) for i in range(num_stages))
    strides = tuple([2] * (num_stages - 1) + [1])
    batch_size = max(1, int(gpu_memory_gb * 0.5))

    return channels, strides, batch_size



class InitWeights_He:
    """He 初始化，配合 LeakyReLU 使用（nnU-Net 标配）。"""
    def __init__(self, negative_slope=1e-2):
        self.a = negative_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.a, mode='fan_in',
                                    nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.InstanceNorm3d):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.pos_emb = SinusoidalPositionEmbedding(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        emb = self.pos_emb(t)
        return self.net(emb)


# ===========================================================================
# ContinuousNoiseEmbedding — Gaussian Fourier features for EDM / logsnr
# Karras et al. 2022: random Gaussian frequencies → sin/cos → MLP
# Works with float noise levels (log σ for EDM, γ for logsnr).
# ===========================================================================

class ContinuousNoiseEmbedding(nn.Module):
    """Gaussian random Fourier feature embedding for continuous noise levels.

    Unlike SinusoidalPositionEmbedding (designed for discrete integer timesteps),
    this uses learnable Gaussian Fourier features suited for continuous σ or γ values.
    Matches the design from Karras et al. 2022 (EDM).
    """
    def __init__(self, embedding_dim=128, hidden_dim=256, n_frequencies=64):
        super().__init__()
        self.n_frequencies = n_frequencies
        # Random Gaussian frequencies (non-learnable, like EDM)
        self.register_buffer(
            "freqs",
            torch.randn(n_frequencies) * 4.0 + 4.0  # ~N(4, 4²)
        )
        in_features = 2 * n_frequencies + 1  # sin + cos + raw t
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        # t: [B] or [B, 1] float — log(σ) for EDM, γ for logsnr
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() > 1:
            t = t.squeeze(-1)
        t_float = t.float()
        # Fourier features: sin(2π · f · t), cos(2π · f · t)
        arg = 2.0 * math.pi * t_float.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [B, F]
        fourier = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)          # [B, 2F]
        # Concatenate raw t for direct amplitude scaling
        emb_input = torch.cat([fourier, t_float.unsqueeze(-1)], dim=-1)        # [B, 2F+1]
        return self.net(emb_input)


class TimeConditionedSwinUNETR(nn.Module):
    """
    SwinUNETR with time conditioning via extra input channel.
    Concatenates x + cond + time_channel, passes through SwinUNETR.
    Compatible with model.py diffusion_loss_fn: model(y_t, t, cond).
    Does NOT modify SwinUNETR internals.

    noise_embedding_mode:
      - "discrete" (default): SinusoidalPositionEmbedding for integer timesteps (beta schedules)
      - "continuous": Gaussian Fourier features for float noise levels (EDM / logsnr)
    """
    def __init__(self, img_size=(96, 96, 96), in_channels=1, cond_channels=3,
                 out_channels=1, feature_size=48, use_checkpoint=False,
                 time_emb_dim=128, time_hidden_dim=256,
                 noise_embedding_mode="discrete"):
        super().__init__()
        self.noise_embedding_mode = noise_embedding_mode
        if noise_embedding_mode == "continuous":
            self.time_mlp = ContinuousNoiseEmbedding(
                embedding_dim=time_emb_dim, hidden_dim=time_hidden_dim)
        else:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        # SwinUNETR input: x + cond + time_channel
        backbone_in_ch = in_channels + cond_channels + 1
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=backbone_in_ch,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, t, cond=None):
        # x: [B, C, D, H, W] — noisy target image
        # t: [B] — timestep (int for discrete, float for continuous)
        # cond: [B, C_cond, D, H, W] — condition (label)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.squeeze(-1))
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_spatial = t_spatial.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        # Use single time channel (average if needed)
        t_ch = t_spatial[:, :1]
        if cond is not None:
            x_input = torch.cat([x, cond, t_ch], dim=1)
        else:
            x_input = torch.cat([x, t_ch], dim=1)
        return self.backbone(x_input)


class TimeConditionedAttentionUnet(nn.Module):
    def __init__(self, in_channels=1, cond_channels=3, out_channels=1,
                 time_emb_dim=128, time_hidden_dim=256,
                 noise_embedding_mode="discrete"):
        super().__init__()
        self.noise_embedding_mode = noise_embedding_mode
        if noise_embedding_mode == "continuous":
            self.time_mlp = ContinuousNoiseEmbedding(
                embedding_dim=time_emb_dim, hidden_dim=time_hidden_dim)
        else:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        backbone_in_ch = in_channels + cond_channels + 1
        self.backbone = AttentionUnet(
            spatial_dims=3,
            in_channels=backbone_in_ch,
            out_channels=out_channels,
            channels=(48, 96, 192, 384, 768),
            strides=(2, 2, 2, 2, 1),
            kernel_size=3,
            up_kernel_size=3,
            dropout=0.0,
        )

    def forward(self, x, t, cond=None):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.squeeze(-1))
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_spatial = t_spatial.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        t_ch = t_spatial[:, :1]
        if cond is not None:
            x_input = torch.cat([x, cond, t_ch], dim=1)
        else:
            x_input = torch.cat([x, t_ch], dim=1)
        return self.backbone(x_input)


class TimeConditionedUNet(nn.Module):
    def __init__(self, in_channels=1, cond_channels=3, out_channels=1,
                 time_emb_dim=128, time_hidden_dim=256,
                 noise_embedding_mode="discrete"):
        super().__init__()
        self.noise_embedding_mode = noise_embedding_mode
        if noise_embedding_mode == "continuous":
            self.time_mlp = ContinuousNoiseEmbedding(
                embedding_dim=time_emb_dim, hidden_dim=time_hidden_dim)
        else:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        backbone_in_ch = in_channels + cond_channels + 1
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=backbone_in_ch,
            out_channels=out_channels,
            channels=(48, 96, 192, 384, 768),
            strides=(2, 2, 2, 1),
        )

    def forward(self, x, t, cond=None):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.squeeze(-1))
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_spatial = t_spatial.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        t_ch = t_spatial[:, :1]
        if cond is not None:
            x_input = torch.cat([x, cond, t_ch], dim=1)
        else:
            x_input = torch.cat([x, t_ch], dim=1)
        return self.backbone(x_input)


class TimeConditionedUNet_NnU(nn.Module):
    """
    TimeConditionedUNet + nnU-Net 理念（参数自动化 + He 初始化）。
    Backbone 仍是 MONAI UNet，不改变 ResidualUnit 内部结构。
    通过 --generator_type Unet_NnU 选择。
    """
    def __init__(self, in_channels=1, cond_channels=3, out_channels=1,
                 input_size=96, gpu_memory_gb=None,
                 time_emb_dim=128, time_hidden_dim=256,
                 noise_embedding_mode="discrete"):
        super().__init__()
        self.noise_embedding_mode = noise_embedding_mode
        if noise_embedding_mode == "continuous":
            self.time_mlp = ContinuousNoiseEmbedding(
                embedding_dim=time_emb_dim, hidden_dim=time_hidden_dim)
        else:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        backbone_in_ch = in_channels + cond_channels + 1

        if gpu_memory_gb is None:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            else:
                gpu_memory_gb = 8

        channels, strides, _bs = plan_network_params(input_size, gpu_memory_gb)
        print(f"[Unet_NnU] GPU={gpu_memory_gb:.1f}GB | "
              f"channels={channels} strides={strides} | "
              f"suggested_batch={_bs}")

        self.backbone = UNet(
            spatial_dims=3,
            in_channels=backbone_in_ch,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
        )
        self.apply(InitWeights_He(1e-2))

    def forward(self, x, t, cond=None):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.squeeze(-1))
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_spatial = t_spatial.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        t_ch = t_spatial[:, :1]
        if cond is not None:
            x_input = torch.cat([x, cond, t_ch], dim=1)
        else:
            x_input = torch.cat([x, t_ch], dim=1)
        return self.backbone(x_input)


class TimeConditionedPlainConvUNet(nn.Module):
    """
    TimeConditionedUNet + PlainConvUNet backbone + nnU-Net 理念。
    Backbone 替换为 PlainConvUNet（plain conv 堆叠，无残差，ConvTranspose3d 上采样）。
    通过 --generator_type PlainConvUNet 选择。
    """
    def __init__(self, in_channels=1, cond_channels=3, out_channels=1,
                 input_size=96, gpu_memory_gb=None,
                 time_emb_dim=128, time_hidden_dim=256,
                 noise_embedding_mode="discrete"):
        super().__init__()
        self.noise_embedding_mode = noise_embedding_mode
        if noise_embedding_mode == "continuous":
            self.time_mlp = ContinuousNoiseEmbedding(
                embedding_dim=time_emb_dim, hidden_dim=time_hidden_dim)
        else:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        backbone_in_ch = in_channels + cond_channels + 1

        if gpu_memory_gb is None:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            else:
                gpu_memory_gb = 8

        channels, strides_monai, _bs = plan_network_params(input_size, gpu_memory_gb)
        # PlainConvUNet needs an initial [1,1,1] stage before any downsampling
        features_per_stage = [channels[0]] + list(channels)
        strides_pl = [[1, 1, 1]] + [[s, s, s] for s in strides_monai]

        print(f"[PlainConvUNet] GPU={gpu_memory_gb:.1f}GB | "
              f"features={features_per_stage} strides={strides_pl} | "
              f"suggested_batch={_bs}")

        try:
            from dynamic_network_architectures.architectures.unet import PlainConvUNet
        except ImportError:
            raise ImportError(
                "PlainConvUNet requires 'dynamic_network_architectures'. "
                "Install it with: pip install dynamic-network-architectures"
            )

        self.backbone = PlainConvUNet(
            input_channels=backbone_in_ch,
            n_stages=len(features_per_stage),
            features_per_stage=features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * len(features_per_stage),
            strides=strides_pl,
            num_classes=out_channels,
            n_conv_per_stage=2,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            dropout_op=None,
            deep_supervision=False,
        )
        self.apply(InitWeights_He(1e-2))

    def forward(self, x, t, cond=None):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.squeeze(-1))
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_spatial = t_spatial.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        t_ch = t_spatial[:, :1]
        if cond is not None:
            x_input = torch.cat([x, cond, t_ch], dim=1)
        else:
            x_input = torch.cat([x, t_ch], dim=1)
        return self.backbone(x_input)


class LabelDenoiser3D(nn.Module):
    """
    3D denoising network for label diffusion.
    Uses 3D convolutions with time embedding as extra channel.
    Designed for 64x64x64 label volumes.
    """
    def __init__(self, in_channels=3, time_emb_dim=128, time_hidden_dim=256,
                 base_channels=48, noise_embedding_mode="discrete"):
        super().__init__()
        self.noise_embedding_mode = noise_embedding_mode
        if noise_embedding_mode == "continuous":
            self.time_mlp = ContinuousNoiseEmbedding(
                embedding_dim=time_emb_dim, hidden_dim=time_hidden_dim)
        else:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbedding(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        c = base_channels
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels + 1, c, 3, padding=1),
            nn.SiLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(c, c * 2, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(c * 4, c * 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(c * 4, c * 4, 3, padding=1),
            nn.SiLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(c * 4, c * 2, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(c * 4, c, 3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv3d(c * 2, c, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(c, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t.squeeze(-1))
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_spatial = t_spatial.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        x_input = torch.cat([x, t_spatial[:, :1]], dim=1)

        e1 = self.encoder1(x_input)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        b = self.bottleneck(e3)
        d3 = self.decoder3(b)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        out = self.decoder1(d2)
        return out


def get_diffusion_network(args, n_steps=1000):
    """
    Factory function matching the original get_nets() interface.
    - args.in_channels: total input channels from data (scan + label channels)
    - args.out_channels: output channels (scan: 1)
    - Condition channels = in_channels - out_channels (i.e. label channels)
    - args.noise_embedding_mode: "discrete" (default, beta schedules) or "continuous" (EDM / logsnr)
    """
    cond_channels = args.in_channels - args.out_channels  # label channels
    noise_mode = getattr(args, "noise_embedding_mode", "discrete")
    if noise_mode == "continuous":
        print(f"[{args.generator_type}] Using ContinuousNoiseEmbedding (EDM / logsnr mode)")
    if args.generator_type == "SwinUNETR":
        print("Using TimeConditionedSwinUNETR (Diffusion)")
        model = TimeConditionedSwinUNETR(
            img_size=(96, 96, 96),
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            noise_embedding_mode=noise_mode,
        )
    elif args.generator_type == "AttentionUnet":
        print("Using TimeConditionedAttentionUnet (Diffusion)")
        model = TimeConditionedAttentionUnet(
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
            noise_embedding_mode=noise_mode,
        )
    elif args.generator_type == "Unet":
        print("Using TimeConditionedUNet (Diffusion)")
        model = TimeConditionedUNet(
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
            noise_embedding_mode=noise_mode,
        )
    elif args.generator_type == "Unet_NnU":
        print("Using TimeConditionedUNet_NnU (Diffusion + nnU-Net planner)")
        model = TimeConditionedUNet_NnU(
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
            noise_embedding_mode=noise_mode,
        )
    elif args.generator_type == "PlainConvUNet":
        print("Using TimeConditionedPlainConvUNet (Diffusion + PlainConvUNet backbone)")
        model = TimeConditionedPlainConvUNet(
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
            noise_embedding_mode=noise_mode,
        )
    else:
        raise ValueError(f"Unknown generator_type: {args.generator_type}")
    return model
