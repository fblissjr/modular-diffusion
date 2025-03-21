# src/configs/factory.py
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigFactory:
    """
    Factory for model configurations.
    
    This class provides a centralized way to retrieve configurations
    for different model types, similar to how LLM inference engines
    use model-specific configs.
    """
    
    # Config registry for different model types
    _registry = {}
    
    @classmethod
    def register_model_type(cls, model_type, config_getter):
        """Register a model type with its config getter function."""
        cls._registry[model_type] = config_getter
    
    @classmethod
    def get_config(cls, model_path, model_type=None):
        """Get the appropriate model config based on path and type."""
        model_path = Path(model_path)
        
        # Try to determine model type if not provided
        if model_type is None:
            model_type = cls._infer_model_type(model_path)
        
        # Use registered handler for this model type
        if model_type in cls._registry:
            return cls._registry[model_type](model_path, model_type)
        
        logger.warning(f"Unknown model type: {model_type}, using generic config")
        return {}
    
    @classmethod
    def _infer_model_type(cls, model_path):
        """Infer model type from model path or config files."""
        path_str = str(model_path).lower()
        
        # Determine from path
        if "wan" in path_str:
            return "wanvideo"
        
        # Check for config files
        config_path = model_path / "config.json"
        if config_path.exists():
            import json
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                if "model_type" in config:
                    return config["model_type"]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        return "generic"