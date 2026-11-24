"""
2D Frequency-Domain FFT L1 Loss for Image Deblurring.
Directly penalizes discrepancies in spatial frequency distribution (blur suppression
and recovery of high-frequency textures and edge details).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTLoss(nn.Module):
    """
    Computes 2D Fast Fourier Transform (rfft2) on prediction and target tensors,
    and calculates the L1 distance across real and imaginary components.
    
    Formula:
        F_pred = rfft2(pred)
        F_target = rfft2(target)
        L_fft = L1(Re(F_pred), Re(F_target)) + L1(Im(F_pred), Im(F_target))
    """
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        fft_pred = torch.fft.rfft2(pred, norm="backward")
        fft_target = torch.fft.rfft2(target, norm="backward")

        real_loss = F.l1_loss(fft_pred.real, fft_target.real)
        imag_loss = F.l1_loss(fft_pred.imag, fft_target.imag)

        return self.loss_weight * (real_loss + imag_loss)
