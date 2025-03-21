# src/configs/models/wanvideo/shared_config.py
import torch
from dataclasses import dataclass, field

@dataclass
class WanSharedConfig:
    """
    Shared configuration parameters for WanVideo models.

    This replaces the EasyDict-based config from the original repo
    with a native Python dataclass implementation.
    """

    # t5 parameters
    t5_model: str = "umt5_xxl"
    t5_dtype: torch.dtype = torch.bfloat16
    text_len: int = 512

    # transformer parameters
    param_dtype: torch.dtype = torch.bfloat16

    # inference parameters
    num_train_timesteps: int = 1000
    sample_fps: int = 16
    sample_neg_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# Create default shared config instance
wan_shared_cfg = WanSharedConfig()