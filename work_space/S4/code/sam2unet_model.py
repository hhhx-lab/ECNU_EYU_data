# sam2unet_model.py - SAM2-UNet for 3D Medical Image Segmentation
"""
SAM2-UNet: Adapts SAM2's Hiera-style hierarchical vision transformer
for 3D volumetric medical image segmentation.

Architecture:
  - Encoder: Hierarchical stages with 3D Windowed Multi-Head Self-Attention
    (inspired by SAM2's Hiera backbone)
  - Decoder: UNet-style with skip connections and ConvTranspose upsampling
  - Heads: 4-class main task head plus 1-channel binary RC head that uses
    main-task softmax context
  - Key difference from U-Mamba: replaces Mamba SSM with windowed self-attention

Reference:
  - SAM2: Ravi et al., "SAM 2: Segment Anything in Images and Videos", ICLR 2025
  - SAM2-UNet: Xiong et al., "SAM2-UNet: Segment Anything 2 Makes Strong Encoder
    for Natural and Medical Image Segmentation", 2024
  - Hiera: Ryali et al., "Hiera: A Hierarchical Vision Transformer without
    the Bells-and-Whistles", ICML 2023

Plan 2 interface: returns main_logits and rc_logit for separate losses, while
still offering a legacy 5-channel compatibility tensor on request.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from einops import rearrange


# ============================================================
# Segmentation Target Definition
# ============================================================

# PLAN CHANGE: BraTS 2025 MET uses four foreground regions plus background.
# The model now separates the common main task from the rare RC task:
#   main_head predicts labels 0..3 (BG/NETC/SNFH/ET)
#   rc_head predicts label 4 as a binary RC-vs-non-RC logit.
BRATS_MAIN_CLASS_NAMES = ("background", "NETC", "SNFH", "ET")
BRATS_MAIN_NUM_CLASSES = len(BRATS_MAIN_CLASS_NAMES)
BRATS_MET_CLASS_NAMES = ("background", "NETC", "SNFH", "ET", "RC")
BRATS_MET_NUM_CLASSES = len(BRATS_MET_CLASS_NAMES)
BRATS_RC_LABEL = 4


# ============================================================
# 3D Windowed Multi-Head Self-Attention (Hiera-style)
# ============================================================

class WindowedAttention3D(nn.Module):
    """
    3D windowed multi-head self-attention.
    Partitions the volume into non-overlapping 3D windows and applies
    self-attention within each window. This keeps memory O(window_size^3)
    instead of O(volume_size^3).

    Args:
        dim: number of channels
        num_heads: number of attention heads
        window_size: (wd, wh, ww) window dimensions
        qkv_bias: whether to add bias to QKV projection
        attn_drop: attention dropout rate
        proj_drop: output projection dropout rate
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias (learnable, per-head)
        wd, wh, ww = window_size
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Compute relative position index
        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # (3, wd, wh, ww)
        coords_flat = coords.reshape(3, -1)  # (3, wd*wh*ww)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
        relative_coords[0] += wd - 1
        relative_coords[1] += wh - 1
        relative_coords[2] += ww - 1
        relative_coords[0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[1] *= (2 * ww - 1)
        relative_position_index = relative_coords.sum(0)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

    def _partition_windows(self, x, D, H, W):
        """Partition volume into non-overlapping 3D windows."""
        wd, wh, ww = self.window_size
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        # Pad if needed
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w
        # Reshape into windows
        x = x.view(B, Dp // wd, wd, Hp // wh, wh, Wp // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # (B, nD, nH, nW, wd, wh, ww, C)
        nW = (Dp // wd) * (Hp // wh) * (Wp // ww)
        x = x.view(B * nW, wd * wh * ww, C)  # (B*nW, window_tokens, C)
        return x, Dp, Hp, Wp, nW

    def _merge_windows(self, x, B, Dp, Hp, Wp, D, H, W):
        """Merge windows back into volume."""
        wd, wh, ww = self.window_size
        nD, nH, nW_d = Dp // wd, Hp // wh, Wp // ww
        nW = nD * nH * nW_d

        x = x.view(B, nD, nH, nW_d, wd, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()  # (B, Dp, Hp, Wp, C)
        x = x.view(B, Dp, Hp, Wp, -1)

        # Remove padding
        if Dp > D or Hp > H or Wp > W:
            x = x[:, :D, :H, :W, :].contiguous()

        return x.view(B, D * H * W, -1)

    def forward(self, x, D, H, W):
        """
        Args:
            x: (B, L, C) where L = D*H*W
            D, H, W: spatial dimensions
        Returns:
            (B, L, C)
        """
        B = x.shape[0]

        # Partition into windows
        x_win, Dp, Hp, Wp, nW = self._partition_windows(x, D, H, W)

        # QKV
        qkv = self.qkv(x_win).reshape(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, BnW, heads, tokens, head_dim)
        q, k, v = qkv.unbind(0)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        win_tokens = self.window_size[0] * self.window_size[1] * self.window_size[2]
        rel_bias = self.rel_pos_bias[self.relative_position_index.view(-1)].view(win_tokens, win_tokens, -1)
        rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = attn + rel_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_win = (attn @ v).transpose(1, 2).reshape(-1, win_tokens, self.dim)
        x_win = self.proj(x_win)
        x_win = self.proj_drop(x_win)

        # Merge windows back
        out = self._merge_windows(x_win, B, Dp, Hp, Wp, D, H, W)
        return out


class AttentionLayer3D(nn.Module):
    """
    Transformer block for 3D volumes: LayerNorm -> Windowed Attention -> Residual
    -> LayerNorm -> MLP -> Residual.

    Replaces MambaLayer in UMambaBackbone.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        # Flatten: (B, C, D, H, W) -> (B, D*H*W, C)
        x_flat = rearrange(x, 'b c d h w -> b (d h w) c')

        # Attention block with residual
        x_flat = x_flat + self.attn(self.norm1(x_flat), D, H, W)

        # MLP block with residual
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Reshape back
        out = rearrange(x_flat, 'b (d h w) c -> b c d h w', d=D, h=H, w=W)
        return out


# ============================================================
# UNet Building Blocks (same structure as UMambaBackbone)
# ============================================================

class ConvBlock3D(nn.Module):
    """3D Convolutional block with normalization and activation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()

        if norm == "instance":
            norm_layer = nn.InstanceNorm3d(out_channels, affine=True)
        elif norm == "batch":
            norm_layer = nn.BatchNorm3d(out_channels)
        else:
            norm_layer = nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm_layer,
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
            norm_layer,
            nn.GELU(),
        )

        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class DownBlock(nn.Module):
    """Encoder block: Conv + Attention + Downsample"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv = ConvBlock3D(in_channels, out_channels, dropout=dropout)

        if use_attention:
            self.attn = AttentionLayer3D(
                out_channels,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
            )
        else:
            self.attn = nn.Identity()

        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = self.attn(x)
        skip = x
        down = self.downsample(x)
        return down, skip


class UpBlock(nn.Module):
    """Decoder block: Upsample + Concat + Conv + (optional) Attention"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels + in_channels, out_channels, dropout=dropout)

        if use_attention:
            self.attn = AttentionLayer3D(
                out_channels,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
            )
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attn(x)
        return x


class Bottleneck(nn.Module):
    """Bottleneck with global attention (no windowing at deepest level)."""
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock3D(channels, channels, dropout=dropout)
        self.attn = AttentionLayer3D(
            channels,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attn(x)
        return x


# ============================================================
# SAM2-UNet 3D Model
# ============================================================

class SAM2UNet3D(nn.Module):
    """
    SAM2-UNet for 3D: Hierarchical transformer encoder (Hiera-inspired)
    with UNet decoder for volumetric medical image segmentation.

    The backbone/decoder keeps the previous UMamba-compatible shape, but the
    output contract is two-head by default:
      - main_logits: 4 channels for BG/NETC/SNFH/ET
      - rc_logit: 1 channel for RC vs non-RC

    The key architectural difference from U-Mamba:
      - U-Mamba uses Mamba (SSM) for long-range dependencies
      - SAM2-UNet uses windowed self-attention (transformer)

    Args:
        spatial_size: Input image size (D, H, W)
        in_channels: Number of input channels (4 for t1c, t1n, t2f, t2w)
        out_channels: Final label-map class count. Plan 2 requires 5 labels:
            BG, NETC, SNFH, ET, RC. The actual heads are 4 + 1 channels.
        feature_size: Base feature dimension (default: 48)
        depths: Number of encoder/decoder stages (default: 4)
        num_heads: Number of attention heads per stage (scales with depth)
        window_size: 3D window size for windowed attention
        dropout_rate: Dropout rate
        use_attention: Whether to use attention layers (set False for pure CNN baseline)
        norm_name: Normalization type
        return_dict: If True, forward returns {'main_logits', 'rc_logit'}.
            If False, forward returns legacy concatenated 5-channel logits for
            compatibility only. Two-head losses should use return_dict=True.
    """

    def __init__(
        self,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 4,
        out_channels: int = BRATS_MET_NUM_CLASSES,
        feature_size: int = 48,
        depths: int = 4,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        dropout_rate: float = 0.2,
        use_attention: bool = True,
        norm_name: str = "instance",
        return_dict: bool = True,
    ):
        super().__init__()

        if out_channels != BRATS_MET_NUM_CLASSES:
            raise ValueError(
                "SAM2UNet3D Plan 2 expects final labels 0..4 "
                f"({BRATS_MET_NUM_CLASSES} classes), got out_channels={out_channels}."
            )

        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.out_channels = BRATS_MET_NUM_CLASSES
        self.main_out_channels = BRATS_MAIN_NUM_CLASSES
        self.rc_out_channels = 1
        self.feature_size = feature_size
        self.depths = depths
        self.dropout_rate = dropout_rate
        self.return_dict = return_dict
        # PLAN CHANGE: expose both the final 5-class label order and the 4-class
        # main-task order for training, validation, and config inspection.
        self.class_names = BRATS_MET_CLASS_NAMES
        self.main_class_names = BRATS_MAIN_CLASS_NAMES
        self.rc_label = BRATS_RC_LABEL

        # Feature dimensions at each stage: [48, 96, 192, 384]
        features = [feature_size * (2 ** i) for i in range(depths)]

        # Heads scale with feature dimension (min 2, max 16)
        heads_per_stage = [max(2, min(16, f // 32)) for f in features]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, feature_size, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(feature_size, affine=True),
            nn.GELU(),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(depths):
            in_ch = features[i - 1] if i > 0 else feature_size
            out_ch = features[i]
            # Use attention in deeper layers (smaller spatial → affordable)
            use_attn = use_attention and (i >= depths // 2)
            self.encoders.append(
                DownBlock(
                    in_ch, out_ch,
                    use_attention=use_attn,
                    num_heads=heads_per_stage[i],
                    window_size=window_size,
                    dropout=dropout_rate,
                )
            )

        # Bottleneck (always uses attention)
        self.bottleneck = Bottleneck(
            features[-1],
            num_heads=heads_per_stage[-1],
            window_size=window_size,
            dropout=dropout_rate,
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depths - 1, -1, -1):
            in_ch = features[i]
            out_ch = features[i - 1] if i > 0 else feature_size
            # Decoder uses attention in deeper layers too
            use_attn = use_attention and (i >= depths // 2)
            self.decoders.append(
                UpBlock(
                    in_ch, out_ch,
                    use_attention=use_attn,
                    num_heads=heads_per_stage[i] if i > 0 else heads_per_stage[0],
                    window_size=window_size,
                    dropout=dropout_rate,
                )
            )

        # Shared task feature projection followed by the two Plan 2 heads.
        self.head_features = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(feature_size, affine=True),
            nn.GELU(),
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )
        self.main_head = nn.Conv3d(feature_size, self.main_out_channels, kernel_size=1)
        self.rc_head = nn.Sequential(
            nn.Conv3d(
                feature_size + self.main_out_channels,
                feature_size,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(feature_size, affine=True),
            nn.GELU(),
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv3d(feature_size, self.rc_out_channels, kernel_size=1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, encoder, bottleneck, and decoder."""

        # Stem
        x = self.stem(x)

        # Encoder
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[i])

        return x

    def forward_heads(self, decoder_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply the main and RC heads.

        The RC head receives both decoder features and softmax(main_logits), so
        it can learn RC as a context-dependent binary target.
        """
        task_features = self.head_features(decoder_features)
        main_logits = self.main_head(task_features)
        main_probs = torch.softmax(main_logits.float(), dim=1).to(dtype=task_features.dtype)
        rc_context = torch.cat([task_features, main_probs], dim=1)
        rc_logit = self.rc_head(rc_context)
        return {
            "main_logits": main_logits,
            "rc_logit": rc_logit,
        }

    @staticmethod
    def to_legacy_logits(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate the 4-class main logits and binary RC logit.

        This is a compatibility bridge for old code paths that still expect a
        5-channel tensor. Plan 2 training should compute separate main and RC
        losses from the dictionary returned by forward(..., return_dict=True).
        """
        return torch.cat([outputs["main_logits"], outputs["rc_logit"]], dim=1)

    @staticmethod
    def logits_to_label_map(
        main_logits: torch.Tensor,
        rc_logit: torch.Tensor,
        rc_threshold: float = 0.3,
    ) -> torch.Tensor:
        """
        Convert two-head logits to the final 0..4 prediction map.

        Main prediction is argmax over BG/NETC/SNFH/ET. Voxels whose RC
        probability exceeds rc_threshold are overridden to label 4.
        """
        main_pred = torch.argmax(main_logits, dim=1)
        rc_prob = torch.sigmoid(rc_logit[:, 0])
        final_pred = main_pred.clone()
        final_pred[rc_prob > rc_threshold] = BRATS_RC_LABEL
        return final_pred.long()

    def predict_label_map(self, x: torch.Tensor, rc_threshold: float = 0.3) -> torch.Tensor:
        """Run forward inference and return final labels 0..4."""
        outputs = self.forward(x, return_dict=True)
        return self.logits_to_label_map(
            outputs["main_logits"],
            outputs["rc_logit"],
            rc_threshold=rc_threshold,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, in_channels, D, H, W)
            return_dict: Overrides self.return_dict for this call.

        Returns:
            By default, a dict with:
              main_logits: (B, 4, D, H, W)
              rc_logit: (B, 1, D, H, W)
            If return_dict=False, returns concatenated legacy logits with shape
            (B, 5, D, H, W).
        """
        outputs = self.forward_heads(self.forward_features(x))
        use_return_dict = self.return_dict if return_dict is None else return_dict
        if use_return_dict:
            return outputs
        return self.to_legacy_logits(outputs)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        return {
            'model': 'SAM2UNet3D',
            'spatial_size': self.spatial_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'main_out_channels': self.main_out_channels,
            'rc_out_channels': self.rc_out_channels,
            'feature_size': self.feature_size,
            'depths': self.depths,
            'dropout_rate': self.dropout_rate,
            'class_names': self.class_names,
            'main_class_names': self.main_class_names,
            'rc_label': self.rc_label,
            'return_dict': self.return_dict,
            'num_parameters': self.get_num_parameters(),
        }


def create_sam2unet(
    spatial_size: Tuple[int, int, int] = (96, 96, 96),
    in_channels: int = 4,
    out_channels: int = BRATS_MET_NUM_CLASSES,
    feature_size: int = 48,
    depths: int = 4,
    num_heads: int = 4,
    window_size: Tuple[int, int, int] = (4, 4, 4),
    dropout_rate: float = 0.2,
    use_attention: bool = True,
    **kwargs,
) -> SAM2UNet3D:
    """Factory function to create SAM2-UNet 3D.

    PLAN CHANGE: defaults now match the Plan 2 segmentation setup:
    4 MRI modalities as input, a 4-class main head, and a 1-channel RC head.
    """
    model = SAM2UNet3D(
        spatial_size=spatial_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dropout_rate=dropout_rate,
        use_attention=use_attention,
        **kwargs,
    )
    print(f"Created SAM2-UNet 3D with {model.get_num_parameters():,} parameters")
    print(f"Configuration: {model.get_config()}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_sam2unet(
        spatial_size=(96, 96, 96),
        in_channels=4,
        # PLAN CHANGE: final label map still uses BG, NETC, SNFH, ET, RC.
        out_channels=BRATS_MET_NUM_CLASSES,
        feature_size=48,
        depths=4,
        num_heads=4,
        window_size=(4, 4, 4),
        dropout_rate=0.2,
        use_attention=True,
    ).to(device)

    dummy_input = torch.randn(1, 4, 96, 96, 96).to(device)

    with torch.no_grad():
        output = model(dummy_input)
        legacy_logits = model(dummy_input, return_dict=False)
        prediction = model.predict_label_map(dummy_input, rc_threshold=0.3)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Main logits shape: {output['main_logits'].shape}")
    print(f"RC logit shape:    {output['rc_logit'].shape}")
    print(f"Legacy logits shape: {legacy_logits.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Model successfully created and tested!")
