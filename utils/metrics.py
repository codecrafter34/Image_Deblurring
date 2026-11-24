"""
Evaluation metrics for image deblurring:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index with 11x11 Gaussian window)
- LPIPS (Learned Perceptual Image Patch Similarity with official lpips library or native fallback)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    Args:
        pred: (B, C, H, W) or (C, H, W) in range [0, data_range]
        target: same shape as pred
        data_range: maximum dynamic range (default 1.0)
    """
    diff = (pred - target).pow(2)
    mse = diff.mean().item()
    if mse < eps:
        return 100.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> float:
    """
    Calculates Structural Similarity Index (SSIM) matching Wang et al. (2004).
    Uses an 11x11 Gaussian window with sigma=1.5.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma).repeat(channels, 1, 1, 1).to(pred.device)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


class LPIPSMetric(nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric evaluator.
    If the official `lpips` library is available, uses `lpips.LPIPS(net='vgg')`.
    Otherwise, provides a pure PyTorch VGG-based normalized perceptual distance fallback.
    """
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_official = False

        try:
            import lpips
            self.model = lpips.LPIPS(net="vgg").to(self.device)
            self.model.eval()
            self.use_official = True
        except Exception:
            # Fallback using torchvision VGG-19
            try:
                from losses.perceptual import VGG19FeatureExtractor
            except ImportError:
                from ..losses.perceptual import VGG19FeatureExtractor
            self.model = VGG19FeatureExtractor().to(self.device)
            self.model.eval()
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            )

    @torch.no_grad()
    def calculate(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Input: tensors in [0, 1] range of shape (B, 3, H, W).
        """
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if target.ndim == 3:
            target = target.unsqueeze(0)

        pred = pred.to(self.device)
        target = target.to(self.device)

        if self.use_official:
            # lpips expects inputs in [-1, 1]
            pred_lpips = pred * 2.0 - 1.0
            target_lpips = target * 2.0 - 1.0
            dist = self.model(pred_lpips, target_lpips)
            return dist.mean().item()
        else:
            p_norm = (pred - self.mean) / self.std
            t_norm = (target - self.mean) / self.std
            p_feats = self.model(p_norm)
            t_feats = self.model(t_norm)

            dist = 0.0
            for pf, tf in zip(p_feats, t_feats):
                # Cosine / normalized L2 distance across channels
                pf_norm = pf / (torch.norm(pf, dim=1, keepdim=True) + 1e-10)
                tf_norm = tf / (torch.norm(tf, dim=1, keepdim=True) + 1e-10)
                dist += (pf_norm - tf_norm).pow(2).mean().item()
            return dist / len(p_feats)
