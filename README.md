# Attention-Augmented U-Net for Real-World Image Deblurring

A modular, production-grade PyTorch (v2.1+, Python 3.10+, CUDA) project for real-world image deblurring using an **Attention-Augmented U-Net** architecture.

---

## 📌 Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Input Blurry Image     │
                          │        (B, 3, H, W)      │
                          └────────────┬─────────────┘
                                       │
     ┌─────────────────────────────────┴─────────────────────────────────┐ (Global Skip)
     │                                                                   │
     ▼                                                                   ▼
┌─────────────┐                                                   ┌─────────────┐
│  Encoder 1  │───────► [ Dual Attention Gate (CA + SA) ] ───────►│  Decoder 1  │
└──────┬──────┘                                                   └──────▲──────┘
       │ (Downsample /2)                                                 │ (Upsample x2)
       ▼                                                                 │
┌─────────────┐                                                   ┌─────────────┐
│  Encoder 2  │───────► [ Dual Attention Gate (CA + SA) ] ───────►│  Decoder 2  │
└──────┬──────┘                                                   └──────▲──────┘
       │ (Downsample /2)                                                 │ (Upsample x2)
       ▼                                                                 │
┌─────────────┐                                                   ┌─────────────┐
│  Encoder 3  │───────► [ Dual Attention Gate (CA + SA) ] ───────►│  Decoder 3  │
└──────┬──────┘                                                   └──────▲──────┘
       │ (Downsample /2)                                                 │ (Upsample x2)
       ▼                                                                 │
┌─────────────┐                                                   ┌─────────────┐
│  Encoder 4  │───────► [ Dual Attention Gate (CA + SA) ] ───────►│  Decoder 4  │
└──────┬──────┘                                                   └──────▲──────┘
       │ (Downsample /2)                                                 │ (Upsample x2)
       ▼                                                                 │
┌────────────────────────────────────────────────────────────────────────┴──────┐
│                    Bottleneck with Dual Attention Block                       │
│                       (Channel Attention + Spatial Attention)                 │
└───────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ Output Residual Conv Head│
                          └────────────┬─────────────┘
                                       │
                                       ▼  (+) Add Global Residual
                          ┌──────────────────────────┐
                          │   Deblurred Image Pred   │
                          │        (B, 3, H, W)      │
                          └──────────────────────────┘
```

### 1. Dual Attention Mechanisms
* **Channel Attention (Squeeze-and-Excitation)**:
  * Squeezes spatial dimensions via **Global Average Pooling** (GAP).
  * Two fully-connected (1x1 Conv) layers with reduction ratio $r=16$, followed by **ReLU** and **Sigmoid** channel-gating:
    $$\mathbf{M}_c(\mathbf{X}) = \sigma\left(\mathbf{W}_2 \cdot \operatorname{ReLU}\left(\mathbf{W}_1 \cdot \operatorname{GAP}(\mathbf{X})\right)\right)$$
* **Spatial Attention (CBAM-style)**:
  * Computes channel-wise average pooling and max pooling concatenated into 2 channels.
  * Convolved with a $7 \times 7$ kernel with padding 3 to capture local spatial relationships.
  * Scaled with **Sigmoid** activation to construct a 2D spatial attention mask:
    $$\mathbf{M}_s(\mathbf{X}) = \sigma\left(f^{7\times 7}\left(\left[\operatorname{AvgPool}(\mathbf{X}); \operatorname{MaxPool}(\mathbf{X})\right]\right)\right)$$
* **Dual Attention Integration**:
  * Applied in encoder-decoder skip connections to suppress background clutter and emphasize blurry edges.
  * Integrated in the bottleneck and decoder stages for global structural context.

### 2. Comparative Baselines Included
* **Baseline U-Net**: Identical 4-level U-Net backbone with attention gates removed for ablation.
* **Swin Transformer (W-MSA / SW-MSA)**: Window-based and Shifted Window-based Multi-Head Self-Attention for cross-architecture benchmarking.

---

## 🎯 Synchronized Augmentation & CutBlur

1. **Synchronized Random 256x256 Cropping**: Extracts identical spatial regions from blurry input and sharp ground truth.
2. **Synchronized Flips & Rotations**: Horizontal ($p=0.5$), Vertical ($p=0.5$), and 90° rotations applied identically to both images.
3. **CutBlur (Choi et al., CVPR 2020)**:
   * Randomly crops a sub-patch from the sharp target image and pastes it into the blurry input image while keeping the ground-truth target sharp.
   * Prevents the network from hallucinating artifacts or over-sharpening already-sharp regions, teaching clean identity mapping locally.

---

## ⚡ Multi-Domain Composite Loss

Combines spatial, perceptual, and frequency-domain constraints:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{char}} \mathcal{L}_{\text{char}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{fft}} \mathcal{L}_{\text{fft}}$$

* **Charbonnier Loss** ($\lambda_{\text{char}} = 1.0, \epsilon = 10^{-3}$):
  $$\mathcal{L}_{\text{char}}(x, y) = \sqrt{\|x - y\|^2 + \epsilon^2}$$
* **Perceptual VGG Loss** ($\lambda_{\text{perc}} = 0.1$):
  * Frozen pre-trained VGG-19 multi-scale feature representations (`relu1_2`, `relu2_2`, `relu3_4`, `relu4_4`, `relu5_4`).
* **2D FFT Frequency Loss** ($\lambda_{\text{fft}} = 0.05$):
  * Computes 2D Fast Fourier Transform (`torch.fft.rfft2`) and penalizes discrepancies in frequency magnitude and phase across real and imaginary components:
    $$\mathcal{L}_{\text{fft}}(x, y) = \|\operatorname{Re}(\mathcal{F}(x)) - \operatorname{Re}(\mathcal{F}(y))\|_1 + \|\operatorname{Im}(\mathcal{F}(x)) - \operatorname{Im}(\mathcal{F}(y))\|_1$$

---

## 📂 Project Directory Layout

```
mazor_product2026implementation/
├── configs/
│   ├── default_config.yaml       # Main Attention-Augmented U-Net training config
│   ├── baseline_unet.yaml        # Baseline U-Net config
│   └── swin_deblur.yaml          # Swin Transformer config
├── data/
│   ├── __init__.py
│   ├── dataset.py                # PairedDeblurDataset & SyntheticPairedDataset
│   └── augmentations.py          # Synchronized Crop, Flips, and CutBlur
├── models/
│   ├── __init__.py               # Model factory (build_model)
│   ├── attention.py              # Channel Attention, Spatial Attention, DualAttentionBlock
│   ├── attention_unet.py         # Attention-Augmented U-Net architecture
│   ├── baseline_unet.py          # Standard Baseline U-Net
│   └── swin_transformer.py       # Swin Transformer with W-MSA & SW-MSA
├── losses/
│   ├── __init__.py
│   ├── charbonnier.py            # Charbonnier loss (eps=1e-3)
│   ├── perceptual.py             # Frozen VGG-19 perceptual loss
│   ├── fft_loss.py               # 2D FFT frequency L1 loss
│   └── composite_loss.py         # Composite multi-domain loss
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # PSNR, SSIM (11x11 Gaussian), LPIPS
│   └── benchmark.py              # Parameter counter and latency profiler
├── tests/
│   └── test_pipeline.py          # 14 comprehensive unit tests
├── train.py                      # Training script (AdamW, Cosine Annealing, GradClip, AMP)
├── evaluate.py                   # Evaluation script (PSNR, SSIM, LPIPS, visualization)
├── benchmark_models.py           # Cross-architecture latency & parameter benchmarking
├── demo.py                       # Single image / folder inference demo
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training

To train with paired dataset (e.g., Kaggle Real-World Deblurring, GoPro, RealBlur):
```bash
python train.py \
  --data_root /path/to/dataset \
  --batch_size 16 \
  --patch_size 256 \
  --lr 2e-4 \
  --min_lr 1e-6 \
  --epochs 100 \
  --cutblur_prob 0.5 \
  --lambda_char 1.0 \
  --lambda_perc 0.1 \
  --lambda_fft 0.05 \
  --amp
```

*Note: If no dataset path is specified, the script automatically boots with `SyntheticPairedDataset` for instant verification.*

### 3. Evaluation

Evaluate model checkpoints, calculate PSNR, SSIM, LPIPS, and save visual comparison strips:
```bash
python evaluate.py \
  --model attention_unet \
  --checkpoint checkpoints/best_model.pth \
  --data_root /path/to/test_dataset \
  --output_dir results
```

### 4. Architecture Benchmarking

Compare parameters, model sizes, and inference latency across Attention U-Net, Baseline U-Net, and Swin Transformer:
```bash
python benchmark_models.py --resolutions 128 256
```

### 5. Single Image Demo

Deblur any single image or folder of blurry images:
```bash
python demo.py \
  --input /path/to/blurry_image.png \
  --checkpoint checkpoints/best_model.pth \
  --output_dir demo_results
```

---

## 📊 Benchmark Summary

| Model Architecture | Total Parameters | Model Checkpoint Size | Latency (128x128) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: |
| **Attention-Augmented U-Net** | **15,162,997** | **57.88 MB** | **59.99 ms** | **16.7 FPS** |
| Baseline Vanilla U-Net | 15,107,331 | 57.67 MB | 70.92 ms | 14.1 FPS |
| Swin Transformer (W/SW-MSA) | 102,867 | 0.52 MB | 117.65 ms | 8.5 FPS |

---

## 🧪 Unit Tests

Run all unit verification tests:
```bash
python -m unittest tests/test_pipeline.py
```
Outputs: `Ran 14 tests in 0.469s - OK`.

License:
Made By Saksham and Nityam
