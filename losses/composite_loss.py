"""
Composite Loss for Real-World Image Deblurring.
Combines:
  1. Charbonnier Spatial Loss (lambda_char = 1.0, eps = 1e-3)
  2. Perceptual VGG-19 Loss   (lambda_perc = 0.1)
  3. 2D Frequency FFT Loss    (lambda_fft  = 0.05)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .charbonnier import CharbonnierLoss
from .perceptual import VGGPerceptualLoss
from .fft_loss import FFTLoss


class CompositeDeblurLoss(nn.Module):
    """
    Composite Multi-Domain Deblurring Loss:
        L_total = lambda_char * L_char + lambda_perc * L_perc + lambda_fft * L_fft
    """
    def __init__(
        self,
        lambda_char: float = 1.0,
        lambda_perc: float = 0.1,
        lambda_fft: float = 0.05,
        charbonnier_eps: float = 1e-3,
        use_vgg: bool = True,
    ):
        super().__init__()
        self.lambda_char = lambda_char
        self.lambda_perc = lambda_perc
        self.lambda_fft = lambda_fft
        self.use_vgg = use_vgg

        self.char_loss = CharbonnierLoss(eps=charbonnier_eps)
        self.fft_loss = FFTLoss()

        if use_vgg and lambda_perc > 0.0:
            try:
                self.perc_loss = VGGPerceptualLoss()
            except Exception as e:
                print(f"[Warning] Failed to initialize pre-trained VGG perceptual loss ({e}). Falling back without VGG.")
                self.perc_loss = None
        else:
            self.perc_loss = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Spatial Charbonnier Loss
        l_char = self.char_loss(pred, target)
        total_loss = self.lambda_char * l_char

        # 2. Frequency Domain 2D FFT Loss
        l_fft = self.fft_loss(pred, target)
        total_loss = total_loss + self.lambda_fft * l_fft

        # 3. Perceptual VGG Loss
        l_perc = torch.tensor(0.0, device=pred.device)
        if self.perc_loss is not None and self.lambda_perc > 0.0:
            l_perc = self.perc_loss(pred, target)
            total_loss = total_loss + self.lambda_perc * l_perc

        loss_dict = {
            "total_loss": total_loss.item(),
            "char_loss": l_char.item(),
            "fft_loss": l_fft.item(),
            "perc_loss": l_perc.item(),
        }

        return total_loss, loss_dict
