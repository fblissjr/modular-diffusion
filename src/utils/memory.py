# utils/memory.py
"""
Memory management utilities for working with large models.

This module provides tools for tracking and optimizing memory usage
during inference, helping to run larger models on limited hardware.
"""

import gc
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Union, Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# utils/memory.py
import gc
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Union, Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryTracker:
    """
    Track and log memory usage during model inference.

    Similar to profiling tools used in LLM inference optimization,
    this helps identify and optimize memory bottlenecks.
    """

    def __init__(self, verbose: bool = False):
        """Initialize the memory tracker."""
        self.verbose = verbose
        self.peak_usage = {}
        self.snapshots = []

    @contextmanager
    def track_usage(self, label: str = "Operation"):
        """
        Context manager to track memory usage during an operation.

        Args:
            label: Description of the operation being tracked
        """
        # Record before state
        if not torch.cuda.is_available():
            yield
            return

        start_stats = self.log_memory_usage(f"Before {label}")

        try:
            # Yield control to the context block
            yield
        finally:
            # Record after state
            end_stats = self.log_memory_usage(f"After {label}")

            # Calculate memory difference
            for device_name in end_stats:
                if device_name in start_stats:
                    diff_allocated = (
                        end_stats[device_name]["allocated_gb"]
                        - start_stats[device_name]["allocated_gb"]
                    )
                    if (
                        self.verbose or abs(diff_allocated) > 0.1
                    ):  # Only log significant changes
                        logger.info(
                            f"{label} memory impact: {diff_allocated:.3f} GB on {device_name}"
                        )

    def log_memory_usage(self, label: str = "Current") -> Dict:
        """
        Log current memory usage statistics.

        Args:
            label: Descriptive label for this memory measurement

        Returns:
            Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {}

        result = {}

        for device_idx in range(torch.cuda.device_count()):
            # Get memory statistics
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
            max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024**3)

            # Update peak usage
            device_name = f"cuda:{device_idx}"
            self.peak_usage[device_name] = max(
                self.peak_usage.get(device_name, 0), allocated
            )

            # Log the statistics if verbose
            if self.verbose:
                logger.info(
                    f"{label} memory usage - {device_name}: "
                    f"allocated={allocated:.3f} GB, "
                    f"reserved={reserved:.3f} GB, "
                    f"peak={self.peak_usage[device_name]:.3f} GB"
                )

            # Store in result
            result[device_name] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
                "peak_usage_gb": self.peak_usage[device_name],
            }

        # Store snapshot
        self.snapshots.append({"label": label, "stats": result})

        return result

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.peak_usage = {}
            logger.debug("Reset peak memory statistics")

    def clear_cache(self):
        """
        Clear CUDA cache and run garbage collection.

        Similar to techniques used in LLM servers to free memory
        between batches or operations.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                logger.debug("Cleared CUDA cache and ran garbage collection")


class MemoryManager:
    """
    Centralized memory optimization manager.

    This handles various memory optimization techniques:
    - Block swapping (moving transformer blocks between devices)
    - Component offloading
    - Data type conversion
    - Attention optimizations
    - Tiled processing

    This separates memory concerns from the core logic, similar to
    how LLM engines have separate memory managers.
    """

    def __init__(
        self,
        config,
        main_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize memory manager.

        Args:
            config: Memory configuration
            main_device: Primary computation device
            offload_device: Device for offloading components
        """
        self.config = config
        self.main_device = main_device
        self.offload_device = offload_device
        self.tracker = MemoryTracker(verbose=False)

        logger.info(
            f"Memory manager initialized - "
            f"main_device={main_device}, "
            f"offload_device={offload_device}"
        )

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply memory optimizations to a model.

        This includes:
        - Data type conversion
        - Quantization
        - Block swapping configuration
        - torch.compile (if enabled)

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model
        """
        # Start with data type conversion
        model = self._convert_dtype(model)

        # Apply quantization if requested
        if self.config.quantize_method != "none":
            model = self._apply_quantization(model)

        # Configure block swapping if enabled
        if hasattr(model, "configure_block_swap") and self.config.block_swap_count > 0:
            model.configure_block_swap(self.config.block_swap_count, non_blocking=True)

        # Apply torch.compile if enabled
        if self.config.use_torch_compile and hasattr(torch, "compile"):
            model = self._apply_torch_compile(model)

        return model

    def run_with_vae_offload(self, vae: nn.Module, func: Callable, *args, **kwargs):
        """
        Run function with VAE temporarily on main device.

        This moves the VAE to the main device, runs the function,
        then returns the VAE to the offload device to save memory.

        Args:
            vae: VAE module
            func: Function to run with VAE on main device
            *args, **kwargs: Arguments to function

        Returns:
            Function result
        """
        # Get initial device
        initial_device = next(vae.parameters()).device

        # Move to main device if needed
        if initial_device != self.main_device:
            with self.tracker.track_usage("Moving VAE to main device"):
                vae.to(self.main_device)

        try:
            # Run function with VAE on main device
            return func(*args, **kwargs)
        finally:
            # Move back to original device if needed
            if initial_device != self.main_device:
                with self.tracker.track_usage("Moving VAE to offload device"):
                    vae.to(initial_device)
                    self.tracker.clear_cache()

    def run_with_diffusion_offload(
        self, diffusion_model: nn.Module, func: Callable, *args, **kwargs
    ):
        """
        Run function with diffusion model only when needed.

        This is particularly useful when using block swapping,
        as it ensures blocks are properly offloaded after use.

        Args:
            diffusion_model: Diffusion model
            func: Function to run with model
            *args, **kwargs: Arguments to function

        Returns:
            Function result
        """
        # Run the function
        result = func(diffusion_model, *args, **kwargs)

        # Ensure blocks are offloaded if using block swapping
        if (
            hasattr(diffusion_model, "blocks_to_swap")
            and diffusion_model.blocks_to_swap > 0
        ):
            self.tracker.clear_cache()

        return result

    def decode_with_tiling(
        self,
        vae: nn.Module,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latents with memory-efficient tiling.

        This breaks large videos into tiles for processing,
        similar to how large images are processed in patches
        in vision transformers.

        Args:
            vae: VAE module
            latents: Latent tensor to decode

        Returns:
            Decoded video tensor
        """
        # Run VAE with optimal settings
        return self.run_with_vae_offload(
            vae,
            lambda: (
                vae.tiled_decode(
                    latents,
                    self.main_device,
                    self.config.vae_tile_size,
                    self.config.vae_tile_stride,
                )
                if self.config.vae_tiling
                else vae.decode(latents.to(self.main_device))
            ),
        )

    def encode_with_tiling(
        self,
        vae: nn.Module,
        videos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode videos with memory-efficient tiling.

        Args:
            vae: VAE module
            videos: Video tensor to encode

        Returns:
            Latent tensor
        """
        # Run VAE with optimal settings
        return self.run_with_vae_offload(
            vae,
            lambda: (
                vae.tiled_encode(
                    videos,
                    self.main_device,
                    self.config.vae_tile_size,
                    self.config.vae_tile_stride,
                )
                if self.config.vae_tiling
                else vae.encode(videos.to(self.main_device))
            ),
        )

    def _convert_dtype(self, model: nn.Module) -> nn.Module:
        """
        Convert model parameters to specified data type.

        Similar to LLM quantization, this keeps certain layers
        (like normalization) in higher precision for stability.

        Args:
            model: Model to convert

        Returns:
            Model with converted data types
        """
        # Identify which parameters to keep in higher precision
        norm_keywords = {"norm", "ln", "layer", "embedding", "bias"}

        # Convert parameters
        with self.tracker.track_usage("Converting model dtype"):
            for name, param in model.named_parameters():
                # Keep normalization layers in FP32 if specified
                if self.config.keep_norm_fp32 and any(
                    k in name.lower() for k in norm_keywords
                ):
                    param.data = param.data.to(torch.float32)
                else:
                    param.data = param.data.to(self.config.dtype)

        return model

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """
        Apply quantization to model.

        Args:
            model: Model to quantize

        Returns:
            Quantized model
        """
        method = self.config.quantize_method

        if method == "fp8_e4m3fn":
            # Check if PyTorch supports FP8
            if hasattr(torch, "_scaled_mm"):
                logger.info("Applying FP8_E4M3FN quantization")
                try:
                    # Import utilities for FP8 conversion
                    from utils.fp8_utils import convert_fp8_linear

                    # Apply conversion
                    params_to_keep = {"norm", "layer_norm", "embedding", "head", "bias"}
                    model = convert_fp8_linear(model, self.config.dtype, params_to_keep)
                    logger.info("FP8 quantization applied successfully")
                except Exception as e:
                    logger.warning(f"FP8 quantization failed: {e}")
                    logger.info("Continuing with unquantized model")
            else:
                logger.warning(
                    "FP8 quantization requested but not supported by PyTorch"
                )
                logger.info("Requires PyTorch 2.1+ with scaled_mm support")

        elif method == "int8_dynamic":
            try:
                import torch.quantization as quant

                logger.info("Applying INT8 dynamic quantization")

                # Define quantizable operations
                quantized_model = quant.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )

                logger.info("INT8 dynamic quantization applied successfully")
                return quantized_model
            except Exception as e:
                logger.warning(f"INT8 quantization failed: {e}")
                logger.info("Continuing with unquantized model")

        return model

    def _apply_torch_compile(self, model: nn.Module) -> nn.Module:
        """
        Apply torch.compile to model for performance.

        Similar to how LLM frameworks use TensorRT or ONNX compilation
        for inference speedups without code changes.

        Args:
            model: Model to compile

        Returns:
            Compiled model
        """
        try:
            if hasattr(torch, "compile"):
                logger.info(
                    f"Applying torch.compile with mode {self.config.compile_mode}"
                )

                # Only compile transformer blocks to avoid issues with complex models
                if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
                    logger.info("Compiling transformer blocks individually")
                    for i, block in enumerate(model.blocks):
                        model.blocks[i] = torch.compile(
                            block,
                            mode=self.config.compile_mode,
                            fullgraph=False,  # Safer option for complex models
                        )
                else:
                    # Try to compile the whole model
                    model = torch.compile(
                        model,
                        mode=self.config.compile_mode,
                        fullgraph=False,
                    )

                logger.info("torch.compile applied successfully")
            else:
                logger.warning("torch.compile requested but not available")
                logger.info("Requires PyTorch 2.0+")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            logger.info("Continuing with uncompiled model")

        return model


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