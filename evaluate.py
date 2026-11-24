"""
Evaluation and Testing Script for Image Deblurring Models.
Computes PSNR, SSIM, and LPIPS metrics on evaluation datasets,
generates visual deblurring comparison figures, and benchmarks inference latency.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.utils as vutils

from data import PairedDeblurDataset, SyntheticPairedDataset, TestTransform
from models import build_model
from utils import calculate_psnr, calculate_ssim, LPIPSMetric, count_parameters, benchmark_latency


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Image Deblurring Models")
    parser.add_argument("--model", type=str, default="attention_unet", choices=["attention_unet", "baseline_unet", "swin_deblur"], help="Model type")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data_root", type=str, default=None, help="Root path of evaluation dataset")
    parser.add_argument("--blur_dir", type=str, default=None, help="Directory containing test blurry images")
    parser.add_argument("--sharp_dir", type=str, default=None, help="Directory containing test sharp ground-truth images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save deblurred images and comparison figures")
    parser.add_argument("--crop_size", type=int, default=None, help="Optional crop size (default: None, evaluate full resolution)")
    parser.add_argument("--save_images", action="store_true", default=True, help="Save deblurred image results")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Benchmark parameter count and inference latency")
    parser.add_argument("--no_vgg", action="store_true", default=False, help="Disable VGG perceptual metric (fast offline mode)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for evaluation")
    return parser.parse_args()


@torch.no_grad()
def evaluate():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Evaluation Pipeline: Model={args.model.upper()} | Device={device}")
    print("=" * 70)

    # 1. Build Model
    model = build_model(args.model).to(device)

    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("[Notice] Running with initialized weights (no checkpoint loaded).")

    model.eval()

    # 2. Benchmark Model Complexity and Latency
    if args.benchmark:
        params_info = count_parameters(model)
        latency_info = benchmark_latency(model, input_size=(1, 3, 256, 256), device=device)
        print("\n--- Model Benchmark ---")
        print(f"Total Parameters     : {params_info['total_params']:,}")
        print(f"Trainable Parameters : {params_info['trainable_params']:,}")
        print(f"Model Memory Footprint: {params_info['total_size_mb']:.2f} MB")
        print(f"Mean Inference Latency: {latency_info['mean_latency_ms']:.2f} ms ({latency_info['fps']} FPS at 256x256)")
        print("-----------------------\n")

    # 3. Load Evaluation Dataset
    transform = TestTransform(crop_size=args.crop_size, pad_multiple=16)
    dataset = None
    try:
        if args.blur_dir and args.sharp_dir:
            dataset = PairedDeblurDataset(blur_dir=args.blur_dir, sharp_dir=args.sharp_dir, transform=transform)
        elif args.data_root:
            dataset = PairedDeblurDataset(root_dir=args.data_root, split="val", transform=transform)
    except Exception as e:
        print(f"[Notice] Could not load dataset from path ({e}). Using SyntheticPairedDataset fallback.")

    if dataset is None:
        dataset = SyntheticPairedDataset(num_samples=16, image_size=256, transform=transform)

    print(f"Evaluating {len(dataset)} paired images...")

    lpips_evaluator = LPIPSMetric(device=device) if not args.no_vgg else None

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    for idx in range(len(dataset)):
        blurry, sharp, filename = dataset[idx]
        blurry = blurry.unsqueeze(0).to(device)
        sharp = sharp.unsqueeze(0).to(device)

        deblurred = model(blurry)

        psnr_val = calculate_psnr(deblurred, sharp)
        ssim_val = calculate_ssim(deblurred, sharp)
        lpips_val = lpips_evaluator.calculate(deblurred, sharp) if lpips_evaluator is not None else 0.0

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        lpips_scores.append(lpips_val)

        if args.save_images and idx < 10:
            # Save side-by-side [Blurry | Deblurred | Ground Truth]
            vis_strip = torch.cat([blurry, deblurred, sharp], dim=0)
            stem = Path(filename).stem
            save_path = os.path.join(args.output_dir, f"{stem}_comparison.png")
            vutils.save_image(vis_strip, save_path, nrow=3, normalize=False)

    mean_psnr = sum(psnr_scores) / len(psnr_scores)
    mean_ssim = sum(ssim_scores) / len(ssim_scores)
    mean_lpips = sum(lpips_scores) / len(lpips_scores)

    print("=" * 70)
    print("EVALUATION RESULTS:")
    print(f"  * PSNR : {mean_psnr:.2f} dB")
    print(f"  * SSIM : {mean_ssim:.4f}")
    print(f"  * LPIPS: {mean_lpips:.4f}")
    print(f"Visual comparison images saved in: '{args.output_dir}'")
    print("=" * 70)

    return {"psnr": mean_psnr, "ssim": mean_ssim, "lpips": mean_lpips}


if __name__ == "__main__":
    evaluate()
