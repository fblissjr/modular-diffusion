"""
Memory management utilities for working with large models.

This module provides tools for tracking and optimizing memory usage
during inference, helping to run larger models on limited hardware.
"""

import gc
import torch
import structlog
from typing import Dict, List, Optional, Union, Callable

logger = structlog.get_logger()

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