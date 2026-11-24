"""
Training Pipeline for Attention-Augmented U-Net Image Deblurring.
Supports:
- Paired real-world or synthetic image dataset
- AdamW optimizer (lr=2e-4, weight_decay=1e-4)
- Cosine Annealing Learning Rate Scheduler (down to 1e-6)
- Gradient Clipping (max_norm = 1.0)
- Automatic Mixed Precision (AMP)
- Composite Loss: Charbonnier (1.0) + VGG Perceptual (0.1) + 2D FFT (0.05)
- CutBlur and Synchronized Data Augmentation
- Periodic validation (PSNR, SSIM, LPIPS) & Checkpoint saving (latest, best)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.utils as vutils

from data import create_dataloaders
from models import build_model
from losses import CompositeDeblurLoss
from utils import calculate_psnr, calculate_ssim, LPIPSMetric, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention-Augmented U-Net for Image Deblurring")
    # Model architecture
    parser.add_argument("--model", type=str, default="attention_unet", choices=["attention_unet", "baseline_unet", "swin_deblur"], help="Model architecture")
    parser.add_argument("--base_channels", type=int, default=32, help="Base channel width for U-Net")

    # Dataset & Augmentation
    parser.add_argument("--data_root", type=str, default=None, help="Root path of paired dataset (e.g., Kaggle Real-World Deblurring)")
    parser.add_argument("--train_blur_dir", type=str, default=None, help="Directory containing train blurry images")
    parser.add_argument("--train_sharp_dir", type=str, default=None, help="Directory containing train sharp ground-truth images")
    parser.add_argument("--val_blur_dir", type=str, default=None, help="Directory containing validation blurry images")
    parser.add_argument("--val_sharp_dir", type=str, default=None, help="Directory containing validation sharp images")
    parser.add_argument("--patch_size", type=int, default=256, help="Training patch crop resolution")
    parser.add_argument("--cutblur_prob", type=float, default=0.5, help="CutBlur augmentation probability")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    # Optimization Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per step (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for Cosine Annealing")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping maximum L2 norm")

    # Loss Weights
    parser.add_argument("--lambda_char", type=float, default=1.0, help="Weight for Charbonnier loss")
    parser.add_argument("--lambda_perc", type=float, default=0.1, help="Weight for VGG perceptual loss")
    parser.add_argument("--lambda_fft", type=float, default=0.05, help="Weight for 2D FFT frequency loss")
    parser.add_argument("--char_eps", type=float, default=1e-3, help="Epsilon for Charbonnier loss")
    parser.add_argument("--no_vgg", action="store_true", default=False, help="Disable VGG perceptual loss (useful for fast offline training)")

    # Checkpoints & Logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--samples_dir", type=str, default="samples", help="Directory to save visual validation samples")
    parser.add_argument("--val_interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--save_interval", type=int, default=5, help="Save periodic checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target computing device")

    return parser.parse_args()


def validate(model, val_loader, lpips_metric, device, epoch, samples_dir):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    first_batch_saved = False

    with torch.no_grad():
        for i, (blurry, sharp, filenames) in enumerate(val_loader):
            blurry = blurry.to(device)
            sharp = sharp.to(device)

            pred = model(blurry)

            for b in range(pred.shape[0]):
                p = pred[b:b+1]
                s = sharp[b:b+1]
                total_psnr += calculate_psnr(p, s)
                total_ssim += calculate_ssim(p, s)
                if lpips_metric is not None:
                    total_lpips += lpips_metric.calculate(p, s)
                count += 1

            # Save qualitative visual comparison strip for the first batch
            if not first_batch_saved and samples_dir is not None:
                n_vis = min(4, pred.shape[0])
                vis_strip = torch.cat([blurry[:n_vis], pred[:n_vis], sharp[:n_vis]], dim=0)
                os.makedirs(samples_dir, exist_ok=True)
                sample_path = os.path.join(samples_dir, f"epoch_{epoch:03d}_val_samples.png")
                vutils.save_image(vis_strip, sample_path, nrow=n_vis, normalize=False)
                first_batch_saved = True

    avg_psnr = total_psnr / max(1, count)
    avg_ssim = total_ssim / max(1, count)
    avg_lpips = total_lpips / max(1, count) if lpips_metric is not None else 0.0

    return avg_psnr, avg_ssim, avg_lpips


def train():
    args = parse_args()
    device = torch.device(args.device)
    print("=" * 70)
    print(f"Starting Training: Model={args.model.upper()} | Device={device} | AMP={args.amp and device.type == 'cuda'}")
    print(f"Batch Size={args.batch_size} | Patch Size={args.patch_size}x{args.patch_size} | Epochs={args.epochs}")
    print(f"Optimizer=AdamW(lr={args.lr}, wd={args.weight_decay}) | GradClip={args.max_grad_norm}")
    print(f"Loss: {args.lambda_char}*Char + {args.lambda_perc}*VGG + {args.lambda_fft}*FFT")
    print("=" * 70)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # 1. Dataloaders
    train_loader, val_loader = create_dataloaders(
        root_dir=args.data_root,
        train_blur_dir=args.train_blur_dir,
        train_sharp_dir=args.train_sharp_dir,
        val_blur_dir=args.val_blur_dir,
        val_sharp_dir=args.val_sharp_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        cutblur_prob=args.cutblur_prob,
        use_synthetic_fallback=True,
    )
    print(f"Dataset loaded: Train batches={len(train_loader)}" + (f", Val batches={len(val_loader)}" if val_loader else ""))

    # 2. Build Model
    model = build_model(args.model, base_channels=args.base_channels).to(device)
    params_info = count_parameters(model)
    print(f"Model parameters: {params_info['trainable_params']:,} trainable ({params_info['total_size_mb']:.2f} MB)")

    # 3. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    # 4. Composite Loss Function
    criterion = CompositeDeblurLoss(
        lambda_char=args.lambda_char,
        lambda_perc=args.lambda_perc if not args.no_vgg else 0.0,
        lambda_fft=args.lambda_fft,
        charbonnier_eps=args.char_eps,
        use_vgg=(args.lambda_perc > 0.0 and not args.no_vgg),
    ).to(device)

    # 5. Metrics & AMP Scaler
    lpips_evaluator = LPIPSMetric(device=device) if not args.no_vgg else None
    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_psnr = -1.0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_psnr = checkpoint.get("best_psnr", -1.0)
            print(f"Resumed from epoch {checkpoint['epoch']} with best PSNR: {best_psnr:.2f} dB")
        else:
            print(f"[Warning] Checkpoint '{args.resume}' not found. Starting from scratch.")

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        running_char = 0.0
        running_fft = 0.0
        running_perc = 0.0

        for step, (blurry, sharp, _) in enumerate(train_loader):
            blurry = blurry.to(device)
            sharp = sharp.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                deblurred = model(blurry)
                loss, loss_dict = criterion(deblurred, sharp)

            scaler.scale(loss).backward()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss_dict["total_loss"]
            running_char += loss_dict["char_loss"]
            running_fft += loss_dict["fft_loss"]
            running_perc += loss_dict["perc_loss"]

        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        num_steps = len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Total: {running_loss / num_steps:.4f} | "
            f"Char: {running_char / num_steps:.4f} | "
            f"FFT: {running_fft / num_steps:.4f} | "
            f"VGG: {running_perc / num_steps:.4f}"
        )

        # Validation Step
        if val_loader is not None and (epoch % args.val_interval == 0 or epoch == args.epochs):
            val_psnr, val_ssim, val_lpips = validate(
                model=model,
                val_loader=val_loader,
                lpips_metric=lpips_evaluator,
                device=device,
                epoch=epoch,
                samples_dir=args.samples_dir,
            )
            print(f"  --> [Validation] PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}")

            # Save best checkpoint
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "val_ssim": val_ssim,
                    "val_lpips": val_lpips,
                    "config": vars(args),
                }, best_path)
                print(f"  [+] Saved new best model checkpoint to {best_path} (PSNR: {best_psnr:.2f} dB)")

        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "latest_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "config": vars(args),
        }, latest_path)

        if epoch % args.save_interval == 0:
            periodic_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, periodic_path)

    print("=" * 70)
    print(f"Training Complete! Best Validation PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints saved in '{args.checkpoint_dir}'.")


if __name__ == "__main__":
    train()
