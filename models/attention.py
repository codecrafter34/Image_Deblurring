"""
Attention Modules for Image Deblurring:
1. Channel Attention (Squeeze-and-Excitation with GAP, reduction ratio r=16, ReLU, Sigmoid).
2. Spatial Attention (Channel-wise AvgPool + MaxPool, 7x7 Conv, Sigmoid).
3. DualAttentionBlock (Sequential Channel + Spatial Attention with residual connection).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation):
    - Global Average Pooling (GAP) squeezes spatial dimensions (H, W) -> 1x1.
    - Two fully-connected (1x1 conv) layers with reduction ratio r=16.
    - ReLU activation in bottleneck, Sigmoid activation for channel gating.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, min_channels: int = 4):
        super().__init__()
        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction_ratio, min_channels)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        channel_weights = self.fc(self.gap(x))  # (B, C, 1, 1)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM-style):
    - Channel-wise Average Pooling & Max Pooling concatenated into 2 channels.
    - 7x7 Convolution to capture inter-spatial relationships.
    - Sigmoid activation to generate 2D spatial attention mask.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)        # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)       # (B, 1, H, W)
        cat_out = torch.cat([avg_out, max_out], dim=1)       # (B, 2, H, W)
        spatial_mask = self.sigmoid(self.conv(cat_out))     # (B, 1, H, W)
        return x * spatial_mask


class DualAttentionBlock(nn.Module):
    """
    Dual Attention Block combining Channel Attention and Spatial Attention sequentially.
    Includes an optional residual shortcut for smooth gradient backpropagation.
    """
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_residual: bool = True,
    ):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction_ratio=reduction_ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ca(x)
        out = self.sa(out)
        if self.use_residual:
            return x + out
        return out


class AttentionSkipConnection(nn.Module):
    """
    Skip connection gate that applies dual attention to encoder feature maps
    before concatenating with decoder upsampled features.
    Suppresses background clutter and highlights blurry edge/texture regions.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.attention = DualAttentionBlock(channels, reduction_ratio=reduction_ratio)

    def forward(self, enc_feat: torch.Tensor, dec_feat: torch.Tensor) -> torch.Tensor:
        attended_enc = self.attention(enc_feat)
        return torch.cat([attended_enc, dec_feat], dim=1)
