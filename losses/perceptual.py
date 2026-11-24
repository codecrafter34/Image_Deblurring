"""
Perceptual VGG Loss using frozen pre-trained VGG-19 feature representations.
Computes multi-layer feature reconstruction error between deblurred and ground truth images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import socket
from typing import List, Optional


class VGG19FeatureExtractor(nn.Module):
    """
    Extracts intermediate activation maps from a frozen VGG-19 network.
    Selected layers:
        relu1_2 (index 3)
        relu2_2 (index 8)
        relu3_4 (index 17)
        relu4_4 (index 26)
        relu5_4 (index 35)
    """
    def __init__(self, layer_indices: Optional[List[int]] = None, pretrained: bool = True):
        super().__init__()
        if layer_indices is None:
            # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
            self.layer_indices = [3, 8, 17, 26, 35]
        else:
            self.layer_indices = layer_indices

        try:
            socket.setdefaulttimeout(3)
            from torchvision.models import VGG19_Weights
            weights = VGG19_Weights.DEFAULT if pretrained else None
            vgg = models.vgg19(weights=weights).features
        except Exception as e:
            try:
                vgg = models.vgg19(pretrained=pretrained).features
            except Exception:
                print(f"[Notice] Could not download VGG-19 pre-trained weights ({e}). Initializing feature extractor in offline mode.")
                try:
                    vgg = models.vgg19(weights=None).features
                except Exception:
                    vgg = models.vgg19(pretrained=False).features

        max_idx = max(self.layer_indices) + 1
        self.slice = nn.Sequential(*[vgg[i] for i in range(max_idx)])

        # Freeze all parameters
        for param in self.slice.parameters():
            param.requires_grad = False
        self.slice.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for idx, layer in enumerate(self.slice):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
        return features


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss based on VGG-19 feature activations.
    Automatically normalizes [0, 1] RGB tensors with ImageNet mean and std.
    """
    def __init__(self, layer_indices: Optional[List[int]] = None, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.vgg = VGG19FeatureExtractor(layer_indices=layer_indices)

        # ImageNet normalization parameters
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 3, H, W) normalized to [0, 1]
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)

        pred_feats = self.vgg(pred_norm)
        with torch.no_grad():
            target_feats = self.vgg(target_norm)

        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf)

        return self.loss_weight * loss
