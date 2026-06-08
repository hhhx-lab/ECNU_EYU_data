import torch
import torch.nn as nn
import math
from monai.networks.nets import SwinUNETR, AttentionUnet, UNet


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


class TimeConditionedSwinUNETR(nn.Module):
    """
    SwinUNETR with time conditioning via extra input channel.
    Concatenates x + cond + time_channel, passes through SwinUNETR.
    Compatible with model.py diffusion_loss_fn: model(y_t, t, cond).
    Does NOT modify SwinUNETR internals.
    """
    def __init__(self, img_size=(96, 96, 96), in_channels=1, cond_channels=3,
                 out_channels=1, feature_size=48, use_checkpoint=False,
                 time_emb_dim=128):
        super().__init__()
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
        # t: [B] — timestep
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
    def __init__(self, in_channels=1, cond_channels=3, out_channels=1, time_emb_dim=128):
        super().__init__()
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
    def __init__(self, in_channels=1, cond_channels=3, out_channels=1, time_emb_dim=128):
        super().__init__()
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


class LabelDenoiser3D(nn.Module):
    """
    3D denoising network for label diffusion.
    Uses 3D convolutions with time embedding as extra channel.
    Designed for 64x64x64 label volumes.
    """
    def __init__(self, in_channels=3, time_emb_dim=128, base_channels=48):
        super().__init__()
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
    """
    cond_channels = args.in_channels - args.out_channels  # label channels
    if args.generator_type == "SwinUNETR":
        print("Using TimeConditionedSwinUNETR (Diffusion)")
        model = TimeConditionedSwinUNETR(
            img_size=(96, 96, 96),
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
        )
    elif args.generator_type == "AttentionUnet":
        print("Using TimeConditionedAttentionUnet (Diffusion)")
        model = TimeConditionedAttentionUnet(
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
        )
    elif args.generator_type == "Unet":
        print("Using TimeConditionedUNet (Diffusion)")
        model = TimeConditionedUNet(
            in_channels=args.out_channels,
            cond_channels=cond_channels,
            out_channels=args.out_channels,
        )
    else:
        raise ValueError(f"Unknown generator_type: {args.generator_type}")
    return model
