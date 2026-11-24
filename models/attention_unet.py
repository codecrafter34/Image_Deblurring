"""
Attention-Augmented U-Net for Real-World Image Deblurring.
Integrates Dual (Channel + Spatial) Attention into skip connections and bottleneck,
with global residual learning for stable deblurring convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .attention import DualAttentionBlock, AttentionSkipConnection


class ConvBlock(nn.Module):
    """Two successive 3x3 convolutions with LeakyReLU and optional Dual Attention."""
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention = DualAttentionBlock(out_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return self.attention(out)


class DownSample(nn.Module):
    """Downsamples spatial resolution by factor of 2 via 3x3 strided conv or maxpool."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpSample(nn.Module):
    """Upsamples feature maps by factor of 2 via Bilinear interpolation + 1x1 conv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionAugmentedUNet(nn.Module):
    """
    Attention-Augmented U-Net architecture:
    - 4-level encoder-decoder hierarchy.
    - Squeeze-and-Excitation Channel Attention + CBAM Spatial Attention.
    - Dual attention integrated at:
        1. Bottleneck (high-level semantic contextual reasoning).
        2. Skip connections (filtering encoder features before decoder fusion).
        3. Decoder refinement blocks.
    - Global residual connection: Pred = Input + Delta to preserve color & structural base.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        use_residual: bool = True,
        attention_in_skips: bool = True,
        attention_in_bottleneck: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.attention_in_skips = attention_in_skips

        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256
        c5 = base_channels * 16  # 512 (bottleneck)

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1, use_attention=False)
        self.down1 = DownSample(c1, c2)

        self.enc2 = ConvBlock(c2, c2, use_attention=False)
        self.down2 = DownSample(c2, c3)

        self.enc3 = ConvBlock(c3, c3, use_attention=False)
        self.down3 = DownSample(c3, c4)

        self.enc4 = ConvBlock(c4, c4, use_attention=False)
        self.down4 = DownSample(c4, c5)

        # Bottleneck with dual attention
        self.bottleneck = nn.Sequential(
            ConvBlock(c5, c5, use_attention=False),
            DualAttentionBlock(c5, reduction_ratio=16) if attention_in_bottleneck else nn.Identity(),
            ConvBlock(c5, c5, use_attention=False),
        )

        # Skip Connection Attention Gates
        if attention_in_skips:
            self.skip_gate4 = AttentionSkipConnection(c4, reduction_ratio=16)
            self.skip_gate3 = AttentionSkipConnection(c3, reduction_ratio=16)
            self.skip_gate2 = AttentionSkipConnection(c2, reduction_ratio=16)
            self.skip_gate1 = AttentionSkipConnection(c1, reduction_ratio=16)
        else:
            self.skip_gate4 = None
            self.skip_gate3 = None
            self.skip_gate2 = None
            self.skip_gate1 = None

        # Decoder
        self.up4 = UpSample(c5, c4)
        self.dec4 = ConvBlock(c4 * 2, c4, use_attention=True)

        self.up3 = UpSample(c4, c3)
        self.dec3 = ConvBlock(c3 * 2, c3, use_attention=True)

        self.up2 = UpSample(c3, c2)
        self.dec2 = ConvBlock(c2 * 2, c2, use_attention=True)

        self.up1 = UpSample(c2, c1)
        self.dec1 = ConvBlock(c1 * 2, c1, use_attention=True)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck
        b = self.bottleneck(self.down4(e4))

        # Decoder with attention skips
        d4_up = self.up4(b)
        f4 = self.skip_gate4(e4, d4_up) if self.skip_gate4 is not None else torch.cat([e4, d4_up], dim=1)
        d4 = self.dec4(f4)

        d3_up = self.up3(d4)
        f3 = self.skip_gate3(e3, d3_up) if self.skip_gate3 is not None else torch.cat([e3, d3_up], dim=1)
        d3 = self.dec3(f3)

        d2_up = self.up2(d3)
        f2 = self.skip_gate2(e2, d2_up) if self.skip_gate2 is not None else torch.cat([e2, d2_up], dim=1)
        d2 = self.dec2(f2)

        d1_up = self.up1(d2)
        f1 = self.skip_gate1(e1, d1_up) if self.skip_gate1 is not None else torch.cat([e1, d1_up], dim=1)
        d1 = self.dec1(f1)

        residual = self.out_conv(d1)

        if self.use_residual:
            out = x + residual
        else:
            out = residual

        return torch.clamp(out, 0.0, 1.0)
