"""
Charbonnier Loss (differentiable smooth L1 loss approximation).
Formula: L(x, y) = sqrt((x - y)^2 + eps^2)
Standard for image restoration / deblurring tasks.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss with configurable epsilon (default eps=1e-3).
    Ensures continuous first and second derivatives and robustness to outliers.
    """
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.eps_sq = eps ** 2
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
