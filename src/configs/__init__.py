# src/configs/__init__.py
from .factory import ConfigFactory

def get_model_config(model_path, model_type=None):
    """
    Get the appropriate model config based on path and type.

    This is the main entry point for obtaining model configurations.
    """
    return ConfigFactory.get_config(model_path, model_type)

# Import and register model config handlers
try:
    # Only import the register function to avoid circular imports
    from .models.wanvideo import register_wanvideo_configs
    register_wanvideo_configs()
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Failed to register configs: {e}")