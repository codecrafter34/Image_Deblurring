"""
Comprehensive Unit Tests for the Deblurring Pipeline.
Tests:
- Synchronized Augmentations (Crop, Flips, CutBlur)
- Dataset & DataLoader
- Attention Modules (Channel, Spatial, DualAttention)
- AttentionAugmentedUNet, BaselineUNet, and SwinDeblur
- Composite Loss (Charbonnier, Perceptual, FFT) & Backpropagation
- Metrics (PSNR, SSIM, LPIPS) and Latency Profiling
"""

import unittest
import torch
import torch.nn as nn
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.augmentations import SynchronizedAugmentation, TestTransform
from data.dataset import SyntheticPairedDataset, create_dataloaders
from models.attention import ChannelAttention, SpatialAttention, DualAttentionBlock, AttentionSkipConnection
from models.attention_unet import AttentionAugmentedUNet
from models.baseline_unet import BaselineUNet
from models.swin_transformer import SwinDeblur
from losses.charbonnier import CharbonnierLoss
from losses.fft_loss import FFTLoss
from losses.composite_loss import CompositeDeblurLoss
from utils.metrics import calculate_psnr, calculate_ssim, LPIPSMetric
from utils.benchmark import count_parameters, benchmark_latency


class TestAugmentationsAndData(unittest.TestCase):
    def test_synchronized_crop_and_flips(self):
        aug = SynchronizedAugmentation(patch_size=64, flip_prob=1.0, cutblur_prob=0.0)
        img1 = torch.rand(3, 128, 128)
        img2 = img1.clone()  # Identical images to verify synchronization

        c1, c2 = aug(img1, img2)
        self.assertEqual(c1.shape, (3, 64, 64))
        self.assertEqual(c2.shape, (3, 64, 64))
        # Since cutblur is 0 and transforms are synchronized, c1 must equal c2
        self.assertTrue(torch.allclose(c1, c2, atol=1e-6))

    def test_cutblur_augmentation(self):
        aug = SynchronizedAugmentation(patch_size=64, flip_prob=0.0, rotation=False, cutblur_prob=1.0)
        blur = torch.zeros(3, 64, 64)
        sharp = torch.ones(3, 64, 64)

        b_aug, s_aug = aug(blur, sharp)
        self.assertEqual(b_aug.shape, (3, 64, 64))
        # CutBlur must have pasted a region of ones into the zeros
        self.assertGreater(b_aug.sum().item(), 0.0)
        # Sharp must remain untouched
        self.assertTrue(torch.allclose(s_aug, sharp))

    def test_synthetic_dataloader(self):
        train_loader, val_loader = create_dataloaders(
            batch_size=4,
            patch_size=64,
            num_workers=0,
            use_synthetic_fallback=True,
            synthetic_samples=8,
        )
        self.assertIsNotNone(train_loader)
        batch = next(iter(train_loader))
        blurry, sharp, names = batch
        self.assertEqual(blurry.shape, (4, 3, 64, 64))
        self.assertEqual(sharp.shape, (4, 3, 64, 64))
        self.assertEqual(len(names), 4)


class TestModelsAndAttention(unittest.TestCase):
    def test_channel_attention(self):
        ca = ChannelAttention(in_channels=32, reduction_ratio=16)
        x = torch.randn(2, 32, 32, 32)
        out = ca(x)
        self.assertEqual(out.shape, x.shape)

    def test_spatial_attention(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 32, 32, 32)
        out = sa(x)
        self.assertEqual(out.shape, x.shape)

    def test_dual_attention_block(self):
        dab = DualAttentionBlock(channels=32, reduction_ratio=16, kernel_size=7)
        x = torch.randn(2, 32, 32, 32)
        out = dab(x)
        self.assertEqual(out.shape, x.shape)

    def test_attention_augmented_unet(self):
        model = AttentionAugmentedUNet(in_channels=3, out_channels=3, base_channels=16)
        x = torch.rand(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 3, 64, 64))
        # Clamped output in [0, 1]
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_baseline_unet(self):
        model = BaselineUNet(in_channels=3, out_channels=3, base_channels=16)
        x = torch.rand(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 3, 64, 64))

    def test_swin_deblur(self):
        model = SwinDeblur(in_channels=3, out_channels=3, embed_dim=32, window_size=8)
        x = torch.rand(1, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))


class TestLossesAndBackprop(unittest.TestCase):
    def test_charbonnier_loss(self):
        loss_fn = CharbonnierLoss(eps=1e-3)
        p = torch.ones(2, 3, 32, 32)
        t = torch.zeros(2, 3, 32, 32)
        val = loss_fn(p, t)
        self.assertGreater(val.item(), 0.0)

    def test_fft_loss(self):
        loss_fn = FFTLoss()
        p = torch.randn(2, 3, 32, 32)
        t = torch.randn(2, 3, 32, 32)
        val = loss_fn(p, t)
        self.assertGreater(val.item(), 0.0)

    def test_composite_loss_and_backward(self):
        model = AttentionAugmentedUNet(in_channels=3, out_channels=3, base_channels=8)
        criterion = CompositeDeblurLoss(
            lambda_char=1.0,
            lambda_perc=0.1,
            lambda_fft=0.05,
            use_vgg=False,  # Test without VGG weights download in sandbox
        )
        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        target = torch.rand(2, 3, 32, 32)

        pred = model(x)
        total_loss, loss_dict = criterion(pred, target)

        self.assertIn("char_loss", loss_dict)
        self.assertIn("fft_loss", loss_dict)
        self.assertGreater(total_loss.item(), 0.0)

        # Verify backpropagation
        total_loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")


class TestMetricsAndBenchmark(unittest.TestCase):
    def test_psnr_and_ssim(self):
        x = torch.rand(1, 3, 64, 64)
        # Identical images
        psnr_id = calculate_psnr(x, x)
        ssim_id = calculate_ssim(x, x)
        self.assertGreater(psnr_id, 80.0)
        self.assertAlmostEqual(ssim_id, 1.0, places=3)

        # Different images
        y = torch.rand(1, 3, 64, 64)
        psnr_diff = calculate_psnr(x, y)
        ssim_diff = calculate_ssim(x, y)
        self.assertLess(psnr_diff, psnr_id)
        self.assertLess(ssim_diff, ssim_id)

    def test_benchmark_functions(self):
        model = AttentionAugmentedUNet(base_channels=8)
        params = count_parameters(model)
        self.assertGreater(params["total_params"], 0)
        self.assertGreater(params["total_size_mb"], 0.0)

        lat = benchmark_latency(model, input_size=(1, 3, 32, 32), warmup_runs=2, test_runs=5)
        self.assertIn("mean_latency_ms", lat)
        self.assertIn("fps", lat)


if __name__ == "__main__":
    unittest.main()
