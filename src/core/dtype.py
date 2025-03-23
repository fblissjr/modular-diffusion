# src/core/dtype.py
import torch
import torch.nn as nn
from typing import Dict, Set, Optional, List, Union, Any
from contextlib import contextmanager
import logging
import functools

logger = logging.getLogger(__name__)

class DtypeManager:
    """
    Centralized dtype management across components.
    
    This handles consistent type conversion and maintains policy on
    which operations need higher precision, similar to how LLMs handle
    mixed precision training and inference.
    """
    
    def __init__(
        self,
        compute_dtype: torch.dtype,
        param_dtype: Optional[torch.dtype] = None,
        keep_modules_fp32: Optional[Set[str]] = None,
    ):
        """
        Initialize dtype manager.
        
        Args:
            compute_dtype: Default dtype for computation
            param_dtype: Default dtype for parameters (defaults to compute_dtype)
            keep_modules_fp32: Module types to keep in FP32 for stability
        """
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype or compute_dtype
        
        # Default modules to keep in FP32 for numerical stability
        # Similar to LLM mixed precision approach where certain layers
        # like normalization need higher precision
        self.fp32_modules = keep_modules_fp32 or {
            "norm", "ln", "layernorm", "rmsnorm", "embedding", "bias"
        }
        
        logger.info(
            f"Initialized DtypeManager: "
            f"compute_dtype={compute_dtype}, "
            f"param_dtype={self.param_dtype}"
        )
    
    def needs_fp32(self, name: str) -> bool:
        """Check if a module/parameter needs FP32 for stability."""
        return any(keyword in name.lower() for keyword in self.fp32_modules)
    
    def get_param_dtype(self, name: str) -> torch.dtype:
        """Get appropriate dtype for a parameter based on name."""
        return torch.float32 if self.needs_fp32(name) else self.param_dtype
    
    def apply_to_model(self, model: nn.Module) -> nn.Module:
        """
        Apply dtype policy to model parameters.
        
        Args:
            model: PyTorch model to convert
            
        Returns:
            Model with converted dtypes
        """
        converted = 0
        kept_fp32 = 0
        
        for name, param in model.named_parameters():
            target_dtype = self.get_param_dtype(name)
            
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                if target_dtype == torch.float32:
                    kept_fp32 += 1
                else:
                    converted += 1
        
        logger.info(
            f"Applied dtype policy: "
            f"converted {converted} parameters to {self.param_dtype}, "
            f"kept {kept_fp32} parameters in FP32"
        )
        
        return model
    
    def register_hooks(self, model: nn.Module) -> nn.Module:
        """
        Register hooks for runtime dtype management.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model with dtype hooks
        """
        hooks_added = 0
        
        for name, module in model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue
                
            # Define hooks
            def make_pre_hook(module_name):
                def hook(module, input):
                    if not input or not isinstance(input[0], torch.Tensor):
                        return input
                    
                    target_dtype = torch.float32 if self.needs_fp32(module_name) else self.compute_dtype
                    return tuple(x.to(target_dtype) if isinstance(x, torch.Tensor) else x for x in input)
                return hook
            
            def make_post_hook(module_name):
                def hook(module, input, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    
                    return output.to(self.compute_dtype)
                return hook
            
            # Register hooks with module name for context
            module.register_forward_pre_hook(make_pre_hook(name))
            module.register_forward_hook(make_post_hook(name))
            hooks_added += 1
        
        logger.info(f"Registered dtype hooks for {hooks_added} modules")
        return model
    
    @contextmanager
    def autocast(self):
        """
        Context manager for mixed precision computation.
        
        Similar to how LLMs use autocast for mixed precision inference.
        """
        if self.compute_dtype != torch.float32 and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                yield
        else:
            yield

    @classmethod
    def no_autocast(cls, func):
        """
        Class method decorator to disable autocast for precision-critical functions.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=False):
                return func(*args, **kwargs)
        return wrapper