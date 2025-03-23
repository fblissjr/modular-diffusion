# src/core/torch_component.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from .component import Component

class TorchComponent(Component, nn.Module):
    """
    PyTorch-specific implementation of Component.
    
    This class combines our Component interface with PyTorch's nn.Module,
    allowing components to leverage PyTorch's module functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PyTorch component.
        
        Args:
            config: Component configuration
        """
        Component.__init__(self, config)
        nn.Module.__init__(self)
        
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> "TorchComponent":
        """
        Move component to specified device and dtype.
        
        Args:
            device: Target device or None to keep current
            dtype: Target dtype or None to keep current
            
        Returns:
            Self for chaining
        """
        # Call parent method to update internal device/dtype
        Component.to(self, device, dtype)
        
        # Use PyTorch's to() method for parameters
        if device is not None or dtype is not None:
            nn.Module.to(self, device=device, dtype=dtype)
            
        return self