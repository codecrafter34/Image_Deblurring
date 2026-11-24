from .attention import ChannelAttention, SpatialAttention, DualAttentionBlock, AttentionSkipConnection
from .attention_unet import AttentionAugmentedUNet
from .baseline_unet import BaselineUNet
from .swin_transformer import SwinDeblur, SwinTransformerBlock

__all__ = [
    "ChannelAttention",
    "SpatialAttention",
    "DualAttentionBlock",
    "AttentionSkipConnection",
    "AttentionAugmentedUNet",
    "BaselineUNet",
    "SwinDeblur",
    "SwinTransformerBlock",
    "build_model",
]


def build_model(model_name: str = "attention_unet", **kwargs):
    model_name = model_name.lower()
    if model_name in ["attention_unet", "attn_unet", "attention_augmented_unet"]:
        return AttentionAugmentedUNet(**kwargs)
    elif model_name in ["baseline_unet", "unet"]:
        return BaselineUNet(**kwargs)
    elif model_name in ["swin", "swin_transformer", "swin_deblur"]:
        return SwinDeblur(**kwargs)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from ['attention_unet', 'baseline_unet', 'swin_deblur'].")
