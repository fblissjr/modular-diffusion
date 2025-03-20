# utils/memory.py
"""
Memory management utilities for working with large models.

This module provides tools for tracking and optimizing memory usage
during inference, helping to run larger models on limited hardware.
"""

from contextlib import contextmanager
import gc
import torch
import structlog
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Union, Callable

logger = logging.getLogger(__name__)

class MemoryConfig:
    """
    Configuration for memory optimizations.

    This handles various techniques to reduce GPU memory usage:
    - Data type selection (FP32, FP16, BF16)
    - Quantization (FP8, INT8)
    - Torch compilation for performance
    - Block swapping and other memory efficiency techniques

    Similar to LLM quantization, these techniques help run larger models
    on limited hardware with controlled quality trade-offs.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,  # Default dtype for computation
        quantize_method: str = "none",  # Quantization method to use
        use_torch_compile: bool = False,  # Whether to use torch.compile
        compile_mode: str = "default",  # Mode for torch.compile
        keep_norm_fp32: bool = True,  # Keep norm layers in fp32
        offload_to_cpu: bool = False,  # Offload unused components to CPU
        block_swap_count: int = 0,  # Number of blocks to swap
        vae_tiling: bool = True,  # Use tiling for VAE operations
        vae_offload: bool = True,  # Offload VAE to CPU when not in use
        text_encoder_offload: bool = True,  # Offload text encoder when not in use
    ):
        """
        Initialize memory configuration.

        Args:
            dtype: Default data type for model parameters and computation
            quantize_method: Quantization method to apply ("none", "fp8_e4m3fn", "int8_dynamic")
            use_torch_compile: Whether to apply torch.compile to the model
            compile_mode: Mode for torch.compile ("default", "reduce-overhead", "max-autotune")
            keep_norm_fp32: Whether to keep normalization layers in FP32
            offload_to_cpu: Whether to offload unused components to CPU
            block_swap_count: Number of transformer blocks to swap to CPU
            vae_tiling: Whether to use tiling for VAE operations
            vae_offload: Whether to offload VAE to CPU when not in use
            text_encoder_offload: Whether to offload text encoder when not in use
        """
        self.dtype = dtype
        self.quantize_method = quantize_method
        self.use_torch_compile = use_torch_compile
        self.compile_mode = compile_mode
        self.keep_norm_fp32 = keep_norm_fp32
        self.offload_to_cpu = offload_to_cpu
        self.block_swap_count = block_swap_count
        self.vae_tiling = vae_tiling
        self.vae_offload = vae_offload
        self.text_encoder_offload = text_encoder_offload

        # Log configuration
        logger.info(
            "Memory configuration initialized",
            dtype=str(dtype),
            quantize=quantize_method,
            compile=use_torch_compile,
            block_swap=block_swap_count,
        )

    def apply_to_model(self, model):
        """
        Apply memory optimizations to a model.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        # Start with data type conversion
        self._convert_dtype(model)

        # Apply quantization if requested
        if self.quantize_method != "none":
            model = self._apply_quantization(model)

        # Apply torch.compile if requested
        if self.use_torch_compile:
            model = self._apply_torch_compile(model)

        # Configure block swapping
        if self.block_swap_count > 0 and hasattr(model, "configure_block_swap"):
            model.configure_block_swap(self.block_swap_count)

        return model

    def _convert_dtype(self, model):
        """
        Convert model to specified dtype, with special handling for certain layers.

        Args:
            model: PyTorch model to convert

        Returns:
            Model with converted data types
        """
        # Parameters to keep in higher precision for stability
        norm_keywords = {"norm", "ln", "layernorm", "rmsnorm", "bias"}

        for name, param in model.named_parameters():
            # Keep normalization layers in fp32 if specified
            if self.keep_norm_fp32 and any(k in name.lower() for k in norm_keywords):
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(self.dtype)

        return model

    def _apply_quantization(self, model):
        """
        Apply quantization to model.

        Args:
            model: PyTorch model to quantize

        Returns:
            Quantized model
        """
        if self.quantize_method == "fp8_e4m3fn":
            # Check if we can use torch.scaled_mm
            if hasattr(torch, "_scaled_mm"):
                logger.info("Applying FP8_E4M3FN quantization using torch._scaled_mm")
                try:
                    from modular_diffusion.utils.fp8_utils import convert_fp8_linear

                    # Linear layers to keep in higher precision
                    params_to_keep = {"norm", "layer_norm", "rmsnorm", "head", "bias"}
                    convert_fp8_linear(model, self.dtype, params_to_keep=params_to_keep)
                    logger.info("FP8 quantization applied successfully")
                except Exception as e:
                    logger.warning(f"FP8 quantization failed: {e}")
            else:
                logger.warning("FP8 quantization not available (requires PyTorch 2.1+)")

        elif self.quantize_method == "int8_dynamic":
            # Dynamic quantization to INT8
            try:
                import torch.quantization as quant

                # Define quantizable operations
                quantized_model = quant.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )

                logger.info("INT8 dynamic quantization applied successfully")
                return quantized_model
            except Exception as e:
                logger.warning(f"INT8 quantization failed: {e}")

        # Default: return model unchanged
        return model

    def _apply_torch_compile(self, model):
        """
        Apply torch.compile to model.

        Args:
            model: PyTorch model to compile

        Returns:
            Compiled model
        """
        try:
            # Only available in PyTorch 2.0+
            if hasattr(torch, "compile"):
                logger.info(f"Applying torch.compile with mode {self.compile_mode}")

                # For large models, it's often better to compile only the transformer blocks
                if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
                    logger.info("Compiling transformer blocks individually")
                    for i, block in enumerate(model.blocks):
                        model.blocks[i] = torch.compile(
                            block,
                            mode=self.compile_mode,
                            fullgraph=False,  # Safer option for complex models
                        )
                else:
                    # Compile the whole model
                    model = torch.compile(
                        model,
                        mode=self.compile_mode,
                        fullgraph=False,  # Safer option for complex models
                    )

                logger.info("torch.compile applied successfully")
            else:
                logger.warning("torch.compile not available, skipping compilation")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

        return model


class MemoryTracker:
    """
    Track and log memory usage during model inference.
    
    This class provides methods to monitor VRAM usage during
    different stages of the inference process, helping to
    identify and optimize bottlenecks.
    """
    
    def __init__(self):
        """Initialize the memory tracker."""
        self.logger = logger.bind(component="MemoryTracker")
        self.peak_usage = {}
        self.snapshots = []

    @contextmanager
    def track_usage(self, label: str = "Operation"):
        """
        Context manager to track memory usage during an operation.

        Args:
            label: Description of the operation being tracked

        Yields:
            None
        """
        # Record before state
        start_stats = self.log_memory_usage(f"Before {label}")

        try:
            # Yield control to the context block
            yield
        finally:
            # Record after state
            end_stats = self.log_memory_usage(f"After {label}")

            # Calculate memory difference
            if torch.cuda.is_available():
                for device_name in end_stats:
                    if device_name in start_stats:
                        diff_allocated = (
                            end_stats[device_name]["allocated_gb"]
                            - start_stats[device_name]["allocated_gb"]
                        )
                        self.logger.info(
                            f"{label} memory impact",
                            device=device_name,
                            diff_gb=f"{diff_allocated:.3f}",
                        )

    def log_memory_usage(self, label: str = "Current"):
        """
        Log current memory usage statistics.
        
        Args:
            label: Descriptive label for this memory measurement
        """
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available, skipping memory tracking")
            return
        
        result = {}
        
        for device_idx in range(torch.cuda.device_count()):
            # Get memory statistics
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)    # GB
            max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024**3)
            
            # Update peak usage
            device_name = f"cuda:{device_idx}"
            self.peak_usage[device_name] = max(
                self.peak_usage.get(device_name, 0), 
                allocated
            )
            
            # Log the statistics
            self.logger.info(
                f"{label} memory usage", 
                device=device_name, 
                allocated_gb=f"{allocated:.3f}",
                reserved_gb=f"{reserved:.3f}",
                max_allocated_gb=f"{max_allocated:.3f}",
                peak_usage_gb=f"{self.peak_usage[device_name]:.3f}"
            )
            
            # Store in result
            result[device_name] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "peak_usage_gb": self.peak_usage[device_name]
            }
        
        # Store snapshot
        self.snapshots.append({
            "label": label,
            "stats": result
        })
        
        return result
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.peak_usage = {}
            self.logger.debug("Reset peak memory statistics")
    
    def clear_cache(self):
        """
        Clear CUDA cache and run garbage collection.
        
        This can help reclaim memory between operations.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("Cleared CUDA cache and ran garbage collection")
    
    def get_memory_summary(self):
        """
        Get a summary of memory usage over time.
        
        Returns:
            Dictionary with memory usage statistics
        """
        return {
            "peak_usage": self.peak_usage,
            "snapshots": self.snapshots
        }

class DeviceManager:
    """
    Manage device operations for memory efficiency.
    
    This class provides methods for moving models between devices
    to optimize memory usage, such as offloading to CPU when not in use.
    """
    
    def __init__(self, main_device: str = "cuda", offload_device: str = "cpu"):
        """
        Initialize the device manager.
        
        Args:
            main_device: Primary computation device
            offload_device: Device to offload models to when not in use
        """
        self.logger = logger.bind(component="DeviceManager")
        self.main_device = torch.device(main_device)
        self.offload_device = torch.device(offload_device)
        self.memory_tracker = MemoryTracker()
    
    def run_with_device_swap(
        self,
        module: torch.nn.Module,
        run_fn: Callable,
        *args,
        **kwargs
    ):
        """
        Move module to main device, run function, then return to offload device.
        
        This pattern allows running a module temporarily on the GPU for
        inference, then moving it back to CPU to free up memory.
        
        Args:
            module: PyTorch module to swap
            run_fn: Function to run with module on main device
            args, kwargs: Arguments to run_fn
        
        Returns:
            Result of run_fn
        """
        # Record initial device
        initial_device = next(module.parameters()).device
        
        # Only swap if needed
        needs_swap = initial_device != self.main_device
        
        if needs_swap:
            self.logger.debug(f"Moving module to {self.main_device}")
            self.memory_tracker.log_memory_usage("Before module to main device")
            module.to(self.main_device)
            self.memory_tracker.log_memory_usage("After module to main device")
        
        try:
            # Run the function
            result = run_fn(module, *args, **kwargs)
            
            # Move back to original device if needed
            if needs_swap:
                self.logger.debug(f"Moving module back to {self.offload_device}")
                module.to(self.offload_device)
                self.memory_tracker.clear_cache()
                self.memory_tracker.log_memory_usage("After module to offload device")
            
            return result
            
        except Exception as e:
            # Ensure module is moved back on error
            if needs_swap:
                module.to(initial_device)
                self.memory_tracker.clear_cache()
            raise e
    
    def optimize_module_memory(
        self,
        module: torch.nn.Module,
        offload_pattern: List[str] = None
    ):
        """
        Optimize memory usage of a module by selectively offloading parts.
        
        This method selectively moves parts of a large model to the
        offload device, keeping only the necessary components on the
        main device.
        
        Args:
            module: PyTorch module to optimize
            offload_pattern: List of parameter name patterns to offload
            
        Returns:
            Optimized module
        """
        if offload_pattern is None:
            return module
        
        # Group parameters by whether they should be offloaded
        offload_params = []
        keep_params = []
        
        for name, param in module.named_parameters():
            if any(pattern in name for pattern in offload_pattern):
                offload_params.append((name, param))
            else:
                keep_params.append((name, param))
        
        # Log what we're doing
        self.logger.info(
            "Optimizing module memory",
            offload_count=len(offload_params),
            keep_count=len(keep_params)
        )
        
        # Move parameters to appropriate devices
        for name, param in offload_params:
            param.data = param.data.to(self.offload_device)
        
        for name, param in keep_params:
            param.data = param.data.to(self.main_device)
        
        return module