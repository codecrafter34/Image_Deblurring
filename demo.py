"""
Demo Script: Deblur single image or image directory using trained Attention-Augmented U-Net.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Image Deblurring Demo")
    parser.add_argument("--input", type=str, required=True, help="Path to input blurry image or folder of images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="attention_unet", help="Model architecture")
    parser.add_argument("--output_dir", type=str, default="demo_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def run_demo():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args.model).to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("[Notice] Using initialized model weights.")
    model.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = [p for p in input_path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    else:
        raise ValueError(f"Input path '{args.input}' does not exist.")

    print(f"Processing {len(image_files)} image(s)...")

    for img_path in image_files:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = TF.to_tensor(img).unsqueeze(0).to(device)

        # Pad to multiple of 16
        _, _, h, w = tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            padded_tensor = tensor

        output = model(padded_tensor)

        # Crop back to original dimensions
        output = output[:, :, :h, :w]

        out_filename = os.path.join(args.output_dir, f"deblurred_{img_path.name}")
        vutils.save_image(output, out_filename, normalize=False)

        # Also save side-by-side comparison
        vis_comparison = torch.cat([tensor, output], dim=0)
        cmp_filename = os.path.join(args.output_dir, f"comparison_{img_path.name}")
        vutils.save_image(vis_comparison, cmp_filename, nrow=2, normalize=False)
        print(f"Saved: {out_filename}")

    print("Demo complete!")


if __name__ == "__main__":
    run_demo()
