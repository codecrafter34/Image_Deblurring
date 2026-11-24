"""
Comparative Benchmark Script for Image Deblurring Architectures:
- AttentionAugmentedUNet
- BaselineUNet
- SwinDeblur (W-MSA / SW-MSA)
Measures parameter counts, model sizes, and inference latency/FPS across different resolutions.
"""

import argparse
import torch
from models import build_model
from utils import count_parameters, benchmark_latency


def benchmark_all_models():
    parser = argparse.ArgumentParser(description="Benchmark Deblurring Architectures")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 512], help="Spatial resolutions to test")
    parser.add_argument("--runs", type=int, default=30, help="Number of benchmark repetitions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(f"BENCHMARKING DEBLURRING ARCHITECTURES ON DEVICE: {device}")
    print("=" * 80)

    models_to_test = [
        ("Attention-Augmented U-Net", "attention_unet", {}),
        ("Baseline U-Net", "baseline_unet", {}),
        ("Swin Transformer (W/SW-MSA)", "swin_deblur", {"embed_dim": 48}),
    ]

    for model_title, model_key, kwargs in models_to_test:
        print(f"\n--- Model: {model_title} ---")
        model = build_model(model_key, **kwargs).to(device)
        model.eval()

        params = count_parameters(model)
        print(f"  * Total Parameters     : {params['total_params']:,}")
        print(f"  * Trainable Parameters : {params['trainable_params']:,}")
        print(f"  * Model Checkpoint Size: {params['total_size_mb']:.2f} MB")

        for res in args.resolutions:
            try:
                lat = benchmark_latency(model, input_size=(1, 3, res, res), device=device, test_runs=args.runs)
                print(f"  * Resolution {res}x{res}: {lat['mean_latency_ms']} ms/image (Throughput: {lat['fps']} FPS)")
            except Exception as e:
                print(f"  * Resolution {res}x{res}: Benchmark error ({e})")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_all_models()
