# src/utils/dtype.py
"""
Dtype management for consistent precision handling across model components.

This module provides centralized control over data types throughout the 
diffusion pipeline, ensuring consistent precision while allowing selective 
use of higher precision for numerical stability when needed.

Similar to how LLM inference engines handle mixed precision, this allows
maintaining performance while preserving stability in critical operations.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Set, Optional, Union, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DtypeManager:
    """
    Centralized manager for dtype consistency across model components.
    
    This is conceptually similar to how LLM engines handle KV cache precision,
    where certain operations need higher precision while others can use lower
    precision for performance.
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
        """Apply dtype policy to model parameters."""
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
        """Register hooks for runtime dtype management."""
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
        
        logger.info(f"Registered dtype hooks to {hooks_added} modules")
        return model
    
    @contextmanager
    def autocast(self):
        """Context manager for mixed precision computation."""
        if self.compute_dtype != torch.float32 and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                yield
        else:
            yield

    def convert_tensor_dtype(tensor, target_dtype, force_contiguous=True):
        """
        Convert tensor to target dtype with proper handling.
        
        This handles edge cases like complex dtypes, quantized types,
        and ensures memory layout is optimized.
        
        Args:
            tensor: Input tensor
            target_dtype: Target data type  
            force_contiguous: Whether to force contiguous memory layout
            
        Returns:
            Converted tensor
        """
        # Skip if already correct dtype
        if tensor.dtype == target_dtype:
            return tensor.contiguous() if force_contiguous else tensor
        
        # Special handling for complex types
        if tensor.is_complex() and not target_dtype.is_complex:
            # Handle complex to real conversion
            tensor = tensor.real
        
        # Convert to target dtype
        result = tensor.to(target_dtype)
        
        # Ensure contiguous memory layout for optimal performance
        if force_contiguous and not result.is_contiguous():
            result = result.contiguous()
        
        return result