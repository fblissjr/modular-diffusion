# src/configs/__init__.py
from src.configs.factory import ConfigFactory
from .models.wanvideo import register_wanvideo_configs


def get_model_config(model_path, model_type=None):
    """
    Get the appropriate model config based on path and type.

    This is the main entry point for obtaining model configurations.
    """
    return ConfigFactory.get_config(model_path, model_type)


# Import and register model config handlers
try:
    register_wanvideo_configs()
except ImportError:
    pass
