# src/core/component.py
from abc import ABC, abstractmethod
import torch  # Still needed for torch.device type hints
from typing import Optional, Dict, Any, Union

class Component(ABC):
    """
    Base interface for all pipeline components.
    
    This provides consistent handling of device, dtype, and configuration
    across all components in the system, while remaining engine-agnostic.
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
        d=self.config.get("device")
        if d is None:d="cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(d)

    
    def _resolve_dtype(self) -> torch.dtype:
        """Resolve component dtype from config."""
        dt=self.config.get("dtype")
        if isinstance(dt,torch.dtype):return dt
        m={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16,"f32":torch.float32,"f16":torch.float16}
        if dt in m:return m[dt]
        if dt=="fp8" and hasattr(torch,"float8_e4m3fn"):return torch.float8_e4m3fn
        return torch.float32

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