# src/core/component.py
from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict, Any, Union

class Component(ABC):
    """
    Base interface for all pipeline components.
    
    This provides consistent handling of device, dtype, and configuration
    across all components in the system, similar to how LLM engines
    have standardized component interfaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize component from configuration.
        
        Args:
            config: Component configuration dictionary
        """
        self.config = config
        self._device = self._resolve_device()
        self._dtype = self._resolve_dtype()
        self.name = config.get("name", self.__class__.__name__)
        
    def _resolve_device(self) -> torch.device:
        """Resolve component device from config."""
        device_str = self.config.get("device", "cuda")
        return torch.device(device_str)
    
    def _resolve_dtype(self) -> torch.dtype:
        """Resolve component dtype from config."""
        dtype_str = self.config.get("dtype", "fp32")
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    @property
    def device(self) -> torch.device:
        """Get component's primary computation device."""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get component's primary computation dtype."""
        return self._dtype
    
    @abstractmethod
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> "Component":
        """
        Move component to specified device and dtype.
        
        Args:
            device: Target device or None to keep current
            dtype: Target dtype or None to keep current
            
        Returns:
            Self for chaining
        """
        if device is not None:
            self._device = device
        if dtype is not None:
            self._dtype = dtype
        return self