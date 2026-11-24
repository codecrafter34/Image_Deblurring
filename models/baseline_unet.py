"""
Baseline Vanilla U-Net for Image Deblurring (for ablation and comparative analysis).
Identical structure to AttentionAugmentedUNet but with attention modules removed.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownSample(nn.Module):
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


class BaselineUNet(nn.Module):
    """
    Standard Baseline U-Net without channel or spatial attention.
    Provides direct ablation baseline for measuring gains from attention augmentation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.down1 = DownSample(c1, c2)

        self.enc2 = ConvBlock(c2, c2)
        self.down2 = DownSample(c2, c3)

        self.enc3 = ConvBlock(c3, c3)
        self.down3 = DownSample(c3, c4)

        self.enc4 = ConvBlock(c4, c4)
        self.down4 = DownSample(c4, c5)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(c5, c5),
            ConvBlock(c5, c5),
        )

        # Decoder
        self.up4 = UpSample(c5, c4)
        self.dec4 = ConvBlock(c4 * 2, c4)

        self.up3 = UpSample(c4, c3)
        self.dec3 = ConvBlock(c3 * 2, c3)

        self.up2 = UpSample(c3, c2)
        self.dec2 = ConvBlock(c2 * 2, c2)

        self.up1 = UpSample(c2, c1)
        self.dec1 = ConvBlock(c1 * 2, c1)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(torch.cat([e4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([e3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))

        residual = self.out_conv(d1)
        out = x + residual if self.use_residual else residual
        return torch.clamp(out, 0.0, 1.0)
