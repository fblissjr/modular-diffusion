# In a new file: modular_diffusion/utils/fp8_utils.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def fp8_linear_forward(module, original_dtype, input):
    """
    Custom forward pass for linear layers using FP8 precision.
    
    This function runs matrix multiplication in FP8 precision for memory efficiency,
    similar to GPTQ quantization used in LLMs.
    
    Args:
        module: Linear module to run in FP8
        original_dtype: Original data type for input/output
        input: Input tensor
        
    Returns:
        Output tensor in original dtype
    """
    weight_dtype = module.weight.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            # For sequence inputs (B, L, D)
            target_dtype = torch.float8_e5m2 if weight_dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
            inn = input.reshape(-1, input.shape[2]).to(target_dtype)
            w = module.weight.t()

            scale = torch.ones((1), device=input.device, dtype=torch.float32)
            bias = module.bias.to(original_dtype) if module.bias is not None else None

            # Use scaled matrix multiplication for FP8
            if bias is not None:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, bias=bias, scale_a=scale, scale_b=scale)
            else:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, scale_a=scale, scale_b=scale)

            if isinstance(o, tuple):
                o = o[0]

            return o.reshape((-1, input.shape[1], module.weight.shape[0]))
        else:
            # For non-sequence inputs, fall back to original
            return module.original_forward(input.to(original_dtype))
    else:
        # For non-FP8 weights, use original forward
        return module.original_forward(input)

def convert_fp8_linear(model, original_dtype, params_to_keep={}):
    """
    Convert linear layers in a model to use FP8 precision.
    
    This patches the forward methods of linear layers to use torch._scaled_mm
    with FP8 precision, similar to techniques used in LLM inference optimization.
    
    Args:
        model: Model to convert
        original_dtype: Original data type to convert back to after matrix multiplication
        params_to_keep: Parameters to keep in original precision
        
    Returns:
        None (modifies model in-place)
    """
    logger.info("Converting linear layers to FP8")
    
    # Enable FP8 matmul
    setattr(model, "fp8_matmul_enabled", True)
   
    # Patch each linear layer
    for name, module in model.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(module, nn.Linear):
                # Store original forward for fallback
                original_forward = module.forward
                setattr(module, "original_forward", original_forward)
                
                # Create new forward function using FP8
                setattr(module, "forward", 
                        lambda input, m=module: fp8_linear_forward(m, original_dtype, input))
                
                logger.debug(f"Converted {name} to FP8")