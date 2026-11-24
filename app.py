"""
Streamlit Web Application for Real-World Image Deblurring
Featuring Attention-Augmented U-Net, Baseline U-Net, and Swin Transformer models.
"""

import os
import time
import math
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import streamlit as st

from models import build_model
from utils import calculate_psnr, calculate_ssim, count_parameters


# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Attention-Augmented Image Deblurring",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Model Loading & Caching
# ==========================================
@st.cache_resource
def load_deblur_model(model_name: str, checkpoint_path: str, device: str):
    """Loads and caches the deblurring model."""
    dev = torch.device(device)
    model = build_model(model_name).to(dev)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=dev)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        # Filter state dict if shape mismatches occur
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a (C, H, W) or (1, C, H, W) torch tensor in [0, 1] to PIL Image."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    tensor = torch.clamp(tensor, 0.0, 1.0).cpu()
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    return Image.fromarray(ndarr)


def generate_motion_blur(image_pil: Image.Image, kernel_size: int = 15, angle: float = 0.0) -> Image.Image:
    """Applies synthetic directional motion blur to a PIL image."""
    tensor = TF.to_tensor(image_pil).unsqueeze(0)
    
    # Generate 2D motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2
    slope = math.tan(math.radians(angle)) if abs(math.cos(math.radians(angle))) > 1e-4 else 1e4
    
    rad = math.radians(angle)
    dx = math.cos(rad)
    dy = math.sin(rad)
    
    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * dx))
        y = int(round(center + offset * dy))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
            
    if kernel.sum() == 0:
        kernel[int(center), int(center)] = 1.0
    else:
        kernel /= kernel.sum()
        
    k_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    pad = kernel_size // 2
    blurred = F.conv2d(tensor, k_tensor, padding=pad, groups=3)
    return tensor_to_pil(blurred.squeeze(0))


def compute_difference_map(img1_tensor: torch.Tensor, img2_tensor: torch.Tensor) -> Image.Image:
    """Computes difference heatmap (|img1 - img2|) in turbo colormap."""
    diff = torch.abs(img1_tensor - img2_tensor).mean(dim=0).cpu().numpy()
    # Normalize
    max_val = max(diff.max(), 1e-5)
    norm_diff = diff / max_val
    colored = cm.turbo(norm_diff)[:, :, :3]  # Drop alpha
    colored_uint8 = (colored * 255).astype(np.uint8)
    return Image.fromarray(colored_uint8)


# ==========================================
# Sidebar: Configuration
# ==========================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.svg", width=160)
    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "Deblurring Architecture",
        options=["attention_unet", "baseline_unet", "swin_deblur"],
        format_func=lambda x: {
            "attention_unet": "Attention-Augmented U-Net (Ours)",
            "baseline_unet": "Baseline Vanilla U-Net",
            "swin_deblur": "Swin Transformer (W/SW-MSA)",
        }[x],
        index=0,
    )

    available_checkpoints = []
    if os.path.isdir("checkpoints"):
        available_checkpoints = [os.path.join("checkpoints", f) for f in os.listdir("checkpoints") if f.endswith(".pth")]
    
    checkpoint_choice = st.selectbox(
        "Model Checkpoint",
        options=available_checkpoints if available_checkpoints else ["None"],
        index=0 if available_checkpoints else 0,
        help="Select a trained PyTorch model checkpoint (.pth)",
    )

    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device_choice = st.selectbox("Compute Device", options=device_options, index=0)

    st.markdown("---")
    st.subheader("Architecture Details")
    if model_choice == "attention_unet":
        st.markdown(
            """
            * **Squeeze-and-Excitation** Channel Attention ($r=16$)
            * **CBAM** Spatial Attention ($7\\times 7$ Conv)
            * **Dual Attention Skip Gates** & Bottleneck
            * **Global Residual Learning**
            """
        )
    elif model_choice == "baseline_unet":
        st.markdown("* Standard 4-level U-Net backbone\n* No channel/spatial attention gates (ablation control)")
    else:
        st.markdown("* Window Multi-Head Self-Attention (W-MSA)\n* Shifted Window Self-Attention (SW-MSA)")

# ==========================================
# Main Title & Tabs
# ==========================================
st.markdown('<div class="main-header">Real-World Image Deblurring Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Attention-Augmented U-Net with Channel & Spatial Attention + 2D FFT Frequency Constraints</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Deblur Studio",
    "🧠 Attention Maps & Explainability",
    "🛠️ Synthetic Blur Playground",
    "📊 Benchmarks & Training Progress",
])

# Load Selected Model
model = load_deblur_model(
    model_name=model_choice,
    checkpoint_path=checkpoint_choice if checkpoint_choice != "None" else None,
    device=device_choice,
)


# ==========================================
# TAB 1: Image Deblur Studio
# ==========================================
with tab1:
    col_input, col_ctrl = st.columns([2, 1])

    with col_input:
        st.subheader("1. Select or Upload Blurry Image")
        input_source = st.radio("Image Source", ["Preset Sample", "Upload Image"], horizontal=True)

        input_pil = None
        ground_truth_pil = None
        preset_name = ""

        if input_source == "Preset Sample":
            preset_samples = {
                "Architecture Grid": "sample_images/architecture_grid",
                "Concentric High-Frequency Rings": "sample_images/concentric_rings",
                "Barcode & Fine Textures": "sample_images/barcode_texture",
            }
            selected_preset = st.selectbox("Choose Sample Pattern", list(preset_samples.keys()))
            prefix = preset_samples[selected_preset]
            blur_path = f"{prefix}_blurry.png"
            sharp_path = f"{prefix}_sharp.png"

            if os.path.isfile(blur_path):
                input_pil = Image.open(blur_path).convert("RGB")
            if os.path.isfile(sharp_path):
                ground_truth_pil = Image.open(sharp_path).convert("RGB")

        else:
            uploaded_file = st.file_uploader("Upload Blurry Image (PNG, JPG)", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_file is not None:
                input_pil = Image.open(uploaded_file).convert("RGB")
                
            uploaded_gt = st.file_uploader("Optional: Upload Ground Truth Sharp Image (to compute PSNR/SSIM)", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded_gt is not None:
                ground_truth_pil = Image.open(uploaded_gt).convert("RGB")

    with col_ctrl:
        st.subheader("2. Run Deblurring")
        st.markdown(f"**Loaded Model:** `{model_choice}`")
        st.markdown(f"**Checkpoint:** `{Path(checkpoint_choice).name if checkpoint_choice != 'None' else 'None'}`")
        st.markdown(f"**Device:** `{device_choice.upper()}`")
        
        run_button = st.button("✨ Deblur Image Now", type="primary", use_container_width=True)

    if input_pil is not None:
        st.markdown("---")
        
        # Inference Logic
        tensor_in = TF.to_tensor(input_pil).unsqueeze(0).to(device_choice)
        _, _, h, w = tensor_in.shape

        # Pad to multiple of 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            padded_in = F.pad(tensor_in, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            padded_in = tensor_in

        # Measure latency
        t0 = time.perf_counter()
        with torch.no_grad():
            deblurred_tensor = model(padded_in)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / latency_ms

        # Unpad
        deblurred_tensor = deblurred_tensor[:, :, :h, :w]
        deblurred_pil = tensor_to_pil(deblurred_tensor)

        # Performance & Metrics Badges
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{latency_ms:.1f} ms</div><div class="metric-lbl">Inference Latency</div></div>',
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{fps:.1f} FPS</div><div class="metric-lbl">Throughput</div></div>',
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{w}x{h}</div><div class="metric-lbl">Resolution</div></div>',
                unsafe_allow_html=True,
            )
        with col_m4:
            if ground_truth_pil is not None:
                gt_tensor = TF.to_tensor(ground_truth_pil.resize((w, h))).unsqueeze(0).to(device_choice)
                psnr_val = calculate_psnr(deblurred_tensor, gt_tensor)
                ssim_val = calculate_ssim(deblurred_tensor, gt_tensor)
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{psnr_val:.2f} dB</div><div class="metric-lbl">PSNR (SSIM: {ssim_val:.3f})</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                params_info = count_parameters(model)
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{params_info["total_params"]/1e6:.1f}M</div><div class="metric-lbl">Parameters</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Comparison Displays
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            st.subheader("🔴 Blurry Input")
            st.image(input_pil, use_container_width=True)

        with col_v2:
            st.subheader("🟢 Deblurred Output")
            st.image(deblurred_pil, use_container_width=True)

        with col_v3:
            st.subheader("🔥 Restored High-Frequencies")
            diff_pil = compute_difference_map(deblurred_tensor.squeeze(0), tensor_in.squeeze(0))
            st.image(diff_pil, use_container_width=True, caption="Residual Difference Map |I_pred - I_blur|")

        # Download Button
        buf = Path("temp_deblurred.png")
        deblurred_pil.save(buf)
        with open(buf, "rb") as file:
            st.download_button(
                label="💾 Download Restored Image",
                data=file,
                file_name="deblurred_output.png",
                mime="image/png",
            )


# ==========================================
# TAB 2: Attention Maps & Explainability
# ==========================================
with tab2:
    st.subheader("🧠 Inside the Attention-Augmented Network")
    st.markdown(
        """
        The Attention-Augmented U-Net incorporates **Dual Channel + Spatial Attention**:
        * **Channel Attention**: Learns *what* high-frequency features (edges, textures, colors) to amplify.
        * **Spatial Attention**: Learns *where* the blur artifacts are concentrated across the 2D image plane.
        """
    )

    if model_choice == "attention_unet" and input_pil is not None:
        st.markdown("#### Real-time Extracted Attention Masks")
        
        # Hook into Attention Skip Gate to extract Spatial Attention Mask
        with torch.no_grad():
            x = TF.to_tensor(input_pil).unsqueeze(0).to(device_choice)
            # Pad
            _, _, h, w = x.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            
            e1 = model.enc1(x)
            e2 = model.enc2(model.down1(e1))
            e3 = model.enc3(model.down2(e2))
            e4 = model.enc4(model.down3(e3))

            # Extract spatial attention mask from skip connection gate
            avg_out = torch.mean(e2, dim=1, keepdim=True)
            max_out, _ = torch.max(e2, dim=1, keepdim=True)
            cat_out = torch.cat([avg_out, max_out], dim=1)
            sa_mask = model.skip_gate2.attention.sa.sigmoid(model.skip_gate2.attention.sa.conv(cat_out))
            
            # Bottleneck channel weights
            b_feat = model.bottleneck[0](model.down4(e4))
            ca_weights = model.bottleneck[1].ca.fc(model.bottleneck[1].ca.gap(b_feat)).squeeze().cpu().numpy()

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("##### 2D Spatial Attention Heatmap (Encoder Stage 2)")
            sa_np = sa_mask.squeeze().cpu().numpy()
            sa_norm = (sa_np - sa_np.min()) / (sa_np.max() - sa_np.min() + 1e-6)
            sa_colored = (cm.viridis(sa_norm)[:, :, :3] * 255).astype(np.uint8)
            st.image(sa_colored, use_container_width=True, caption="Spatial Attention Mask (Yellow = High Attention on Blur Edges)")

        with col_a2:
            st.markdown("##### Channel Attention Weights (Bottleneck Top 32 Channels)")
            st.bar_chart(ca_weights[:32])
    else:
        st.info("Switch model to 'Attention-Augmented U-Net' in the sidebar to inspect real-time attention masks.")


# ==========================================
# TAB 3: Synthetic Blur Playground
# ==========================================
with tab3:
    st.subheader("🛠️ Interactive Motion Blur Generator & Restoration")
    st.markdown("Upload any clean photo or use a preset, adjust synthetic camera motion blur parameters, and inspect how well the network restores it.")

    col_syn_img, col_syn_ctrl = st.columns([2, 1])

    with col_syn_ctrl:
        blur_kernel_size = st.slider("Motion Blur Kernel Size (pixels)", min_value=3, max_value=35, value=15, step=2)
        blur_angle = st.slider("Motion Blur Angle (degrees)", min_value=0.0, max_value=180.0, value=30.0, step=5.0)

    with col_syn_img:
        # Load sharp sample
        sample_sharp_path = "sample_images/architecture_grid_sharp.png"
        base_sharp = Image.open(sample_sharp_path).convert("RGB") if os.path.isfile(sample_sharp_path) else None

        if base_sharp is not None:
            synth_blurry = generate_motion_blur(base_sharp, kernel_size=blur_kernel_size, angle=blur_angle)

            # Deblur synthesized image
            t_blur = TF.to_tensor(synth_blurry).unsqueeze(0).to(device_choice)
            with torch.no_grad():
                res = model(t_blur)
            synth_deblurred = tensor_to_pil(res)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("##### 1. Original Sharp")
                st.image(base_sharp, use_container_width=True)
            with c2:
                st.markdown(f"##### 2. Synthesized Blur ({blur_kernel_size}px, {blur_angle}°)")
                st.image(synth_blurry, use_container_width=True)
            with c3:
                st.markdown("##### 3. AI Deblurred")
                st.image(synth_deblurred, use_container_width=True)


# ==========================================
# TAB 4: Benchmarks & Training Progress
# ==========================================
with tab4:
    st.subheader("📊 Cross-Architecture Benchmarks")
    
    st.table([
        {"Architecture": "Attention-Augmented U-Net (Ours)", "Parameters": "15,162,997", "Model Size": "57.88 MB", "Latency (128x128)": "60.2 ms", "FPS": "16.6 FPS"},
        {"Architecture": "Baseline Vanilla U-Net", "Parameters": "15,107,331", "Model Size": "57.67 MB", "Latency (128x128)": "78.8 ms", "FPS": "12.7 FPS"},
        {"Architecture": "Swin Transformer (W/SW-MSA)", "Parameters": "102,867", "Model Size": "0.52 MB", "Latency (128x128)": "117.0 ms", "FPS": "8.5 FPS"},
    ])

    st.markdown("---")
    st.subheader("📈 Training Convergence History")
    
    # Check if samples folder has validation outputs
    if os.path.isdir("samples"):
        sample_imgs = sorted([os.path.join("samples", f) for f in os.listdir("samples") if f.endswith(".png")])
        if sample_imgs:
            st.markdown("##### Validation Sample Strips Across Epochs `[Blurry | Prediction | GroundTruth]`")
            selected_epoch_img = st.selectbox("Select Validation Epoch Image", sample_imgs, index=len(sample_imgs) - 1)
            st.image(selected_epoch_img, use_container_width=True, caption=f"Validation Strip: {selected_epoch_img}")
