"""
Synchronized Data Augmentations for Real-World Image Deblurring.
Includes:
- Synchronized Random 256x256 Cropping
- Synchronized Random Horizontal & Vertical Flips (p=0.5)
- CutBlur Augmentation (Choi et al., CVPR 2020)
"""

import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SynchronizedAugmentation:
    """
    Applies synchronized geometric and data augmentations to paired (blur, sharp) images.
    Input images are expected to be torch.Tensor with shape (C, H, W) normalized to [0, 1].
    """
    def __init__(
        self,
        patch_size: int = 256,
        flip_prob: float = 0.5,
        rotation: bool = True,
        cutblur_prob: float = 0.5,
        cutblur_ratio_range: Tuple[float, float] = (0.2, 0.8),
    ):
        self.patch_size = patch_size
        self.flip_prob = flip_prob
        self.rotation = rotation
        self.cutblur_prob = cutblur_prob
        self.cutblur_ratio_range = cutblur_ratio_range

    def random_crop(
        self, blur: torch.Tensor, sharp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts identical spatial crops of size (patch_size, patch_size) from both images.
        If image dimensions are smaller than patch_size, symmetric reflection padding is applied.
        """
        _, h, w = blur.shape

        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)

        if pad_h > 0 or pad_w > 0:
            blur = F.pad(blur, (0, pad_w, 0, pad_h), mode="reflect")
            sharp = F.pad(sharp, (0, pad_w, 0, pad_h), mode="reflect")
            _, h, w = blur.shape

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)

        blur_crop = blur[:, top : top + self.patch_size, left : left + self.patch_size]
        sharp_crop = sharp[:, top : top + self.patch_size, left : left + self.patch_size]

        return blur_crop, sharp_crop

    def random_flip_and_rotate(
        self, blur: torch.Tensor, sharp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies identical horizontal flip, vertical flip, and 90-degree rotations.
        """
        # Horizontal flip
        if random.random() < self.flip_prob:
            blur = torch.flip(blur, dims=[2])
            sharp = torch.flip(sharp, dims=[2])

        # Vertical flip
        if random.random() < self.flip_prob:
            blur = torch.flip(blur, dims=[1])
            sharp = torch.flip(sharp, dims=[1])

        # Random 90 degree rotation
        if self.rotation and random.random() < self.flip_prob:
            k = random.choice([1, 2, 3])
            blur = torch.rot90(blur, k=k, dims=[1, 2])
            sharp = torch.rot90(sharp, k=k, dims=[1, 2])

        return blur, sharp

    def apply_cutblur(
        self, blur: torch.Tensor, sharp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CutBlur (Choi et al., CVPR 2020):
        Pastes a random rectangular sub-patch from the ground-truth sharp image
        into the blurry input image.
        The target ground truth remains unchanged (sharp).
        This forces the network to learn both identity mapping on sharp areas
        and deblurring on blurred areas without hallucinating artifacts.
        """
        if random.random() >= self.cutblur_prob:
            return blur, sharp

        _, h, w = blur.shape
        low, high = self.cutblur_ratio_range

        # Random aspect ratio and scale for the CutBlur box
        box_w = int(w * random.uniform(low, high))
        box_h = int(h * random.uniform(low, high))

        # Clamp sizes
        box_w = max(8, min(box_w, w - 1))
        box_h = max(8, min(box_h, h - 1))

        x1 = random.randint(0, w - box_w)
        y1 = random.randint(0, h - box_h)
        x2 = x1 + box_w
        y2 = y1 + box_h

        # Clone blur to prevent in-place mutation of cache
        blur_aug = blur.clone()
        blur_aug[:, y1:y2, x1:x2] = sharp[:, y1:y2, x1:x2]

        return blur_aug, sharp

    def __call__(
        self, blur: torch.Tensor, sharp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full pipeline: Random Crop -> Random Flips -> CutBlur
        """
        blur, sharp = self.random_crop(blur, sharp)
        blur, sharp = self.random_flip_and_rotate(blur, sharp)
        blur, sharp = self.apply_cutblur(blur, sharp)
        return blur.contiguous(), sharp.contiguous()


class TestTransform:
    """
    Test/Validation transform that performs optional center-crop or ensures
    dimensions are multiples of 16 / 32 for U-Net downsampling paths.
    """
    def __init__(self, crop_size: Optional[int] = None, pad_multiple: int = 16):
        self.crop_size = crop_size
        self.pad_multiple = pad_multiple

    def __call__(
        self, blur: torch.Tensor, sharp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.crop_size is not None:
            _, h, w = blur.shape
            top = max(0, (h - self.crop_size) // 2)
            left = max(0, (w - self.crop_size) // 2)
            blur = blur[:, top : top + self.crop_size, left : left + self.crop_size]
            sharp = sharp[:, top : top + self.crop_size, left : left + self.crop_size]
        elif self.pad_multiple > 1:
            _, h, w = blur.shape
            mod_h = h % self.pad_multiple
            mod_w = w % self.pad_multiple
            pad_h = (self.pad_multiple - mod_h) if mod_h != 0 else 0
            pad_w = (self.pad_multiple - mod_w) if mod_w != 0 else 0
            if pad_h > 0 or pad_w > 0:
                blur = F.pad(blur, (0, pad_w, 0, pad_h), mode="reflect")
                sharp = F.pad(sharp, (0, pad_w, 0, pad_h), mode="reflect")

        return blur.contiguous(), sharp.contiguous()
