# src/models/diffusion/base.py
from abc import ABC,abstractmethod
from typing import List,Dict,Any,Union,Optional,Tuple
import torch,torch.nn as nn
from src.core.component import Component

class DiffusionModel(Component,nn.Module,ABC):
    """
    Base interface for diffusion models.
    
    This provides a common interface for different diffusion models,
    """
    
    
    def __init__(self,config:Dict[str,Any]):
        Component.__init__(self,config)
        nn.Module.__init__(self)
        self.in_channels=self.out_channels=None  # Set by implementation
    
    def to(self,device=None,dtype=None)->"DiffusionModel":
        Component.to(self,device,dtype)
        if device or dtype:nn.Module.to(self,device=device,dtype=dtype)
        return self
    
    @abstractmethod
    def forward(self,latents:List[torch.Tensor],timestep:torch.Tensor,
               text_embeds:List[torch.Tensor],**kwargs)->Tuple[List[torch.Tensor],Optional[Dict]]:
        """
        Forward pass through diffusion model.
        
        Args:
            latents: List of latent tensors
            timestep: Current timestep
            text_embeds: Text embeddings for conditioning
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (predicted noise, optional auxiliary outputs)
        """
        pass