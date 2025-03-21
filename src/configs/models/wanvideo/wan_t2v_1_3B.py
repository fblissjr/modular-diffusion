# src/configs/models/wanvideo/wan_t2v_1_3B.py
import torch
from dataclasses import dataclass
from typing import Tuple

from .shared_config import WanSharedConfig, wan_shared_cfg

@dataclass
class WanT2V1_3BConfig(WanSharedConfig):
    """Configuration for WanVideo T2V 1.3B model."""

    # Model name
    name: str = "Wan T2V 1.3B"

    # t5 parameters
    t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth"
    t5_tokenizer: str = "google/umt5-xxl"

    # vae parameters
    vae_checkpoint: str = "Wan2.1_VAE.pth"
    vae_stride: Tuple[int, int, int] = (4, 8, 8)

    # transformer parameters
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    dim: int = 1536
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    num_layers: int = 30
    window_size: Tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6

# Create config instance
t2v_1_3B = WanT2V1_3BConfig()