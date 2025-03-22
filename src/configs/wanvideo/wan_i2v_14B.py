# src/configs/models/wanvideo/wan_i2v_14B.py
import torch
from dataclasses import dataclass
from typing import Tuple

from .shared_config import WanSharedConfig, wan_shared_cfg

@dataclass
class WanI2V14BConfig(WanSharedConfig):
    """Configuration for WanVideo I2V 14B model."""

    # Model name
    name: str = "Wan I2V 14B"

    # t5 parameters
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"

    # clip parameters
    clip_model: str = "clip_xlm_roberta_vit_h_14"
    clip_dtype: torch.dtype = torch.float16
    clip_checkpoint: str = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    clip_tokenizer: str = "xlm-roberta-large"

    # vae parameters
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    vae_stride: Tuple[int, int, int] = (4, 8, 8)

    # transformer parameters
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    dim: int = 5120
    ffn_dim: int = 13824
    freq_dim: int = 256
    num_heads: int = 40
    num_layers: int = 40
    window_size: Tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6

# Create config instance
i2v_14B = WanI2V14BConfig()