# src/models/diffusion/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import torch
from src.core.component import Component

class DiffusionModel(Component, ABC):
    """
    Base interface for diffusion models.
    
    This provides a common interface for different diffusion models,
    similar to how LLM decoders have standard interfaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize diffusion model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.in_channels = None  # Set by implementation
        self.out_channels = None  # Set by implementation
    
    @abstractmethod
    def forward(self, 
               latents: List[torch.Tensor], 
               timestep: torch.Tensor, 
               text_embeds: List[torch.Tensor], 
               **kwargs) -> Tuple[List[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through diffusion model.
        
        Args:
            latents: List of latent tensors
            timestep: Current timestep
            text_embeds: Text embeddings for conditioning
            **kwargs: Additional arguments
                - seq_len: Sequence length
                - is_uncond: Whether this is unconditional
                - current_step_percentage: Current step progress
            
        Returns:
            Tuple of (predicted noise, optional auxiliary outputs)
        """
        pass