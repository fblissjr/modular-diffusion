# src/configs/models/wanvideo/__init__.py
import os
import copy
from pathlib import Path

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import configs
from .shared_config import wan_shared_cfg
from .wan_i2v_14B import i2v_14B
from .wan_t2v_1_3B import t2v_1_3B
from .wan_t2v_14B import t2v_14B

# Create t2i_14B config (same as t2v_14B)
t2i_14B = copy.deepcopy(t2v_14B)
t2i_14B.name = "Wan T2I 14B"

# Config registry
WAN_CONFIGS = {
    "t2v-14B": t2v_14B,
    "t2v-1.3B": t2v_1_3B,
    "i2v-14B": i2v_14B,
    "t2i-14B": t2i_14B,
}

# Video resolution configs
SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "1024*1024": (1024, 1024),
}

# Supported sizes for each model
SUPPORTED_SIZES = {
    't2v-14B': ('720*1280', '1280*720', '480*832', '832*480'),
    't2v-1.3B': ('480*832', '832*480'),
    'i2v-14B': ('720*1280', '1280*720', '480*832', '832*480'),
    't2i-14B': tuple(SIZE_CONFIGS.keys()),
}

def get_wan_config(model_path, model_type=None):
    """
    Get the appropriate WanVideo model config based on path and type.

    Args:
        model_path: Path to model directory
        model_type: Model type (t2v, i2v, t2i) or None to detect

    Returns:
        Model configuration
    """
    path_str = str(model_path).lower()
    
    # Determine model type if not provided
    if model_type is None:
        if "t2v" in path_str or "text-to-video" in path_str:
            model_type = "t2v"
        elif "i2v" in path_str or "image-to-video" in path_str:
            model_type = "i2v"
        elif "t2i" in path_str or "text-to-image" in path_str:
            model_type = "t2i"
        else:
            # Default to t2v if we can't determine
            model_type = "t2v"
    
    # Determine model size
    if "14b" in path_str or "14B" in path_str:
        model_size = "14B"
    else:
        model_size = "1.3B"
    
    # Get config key
    config_key = f"{model_type}-{model_size}"
    
    # Return the config if it exists, otherwise use the default
    return WAN_CONFIGS.get(config_key, t2v_1_3B)


def register_wanvideo_configs():
    """Register WanVideo config getters with the factory."""
    from ...factory import ConfigFactory

    ConfigFactory.register_model_type("wanvideo", get_wan_config)