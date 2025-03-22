# src/models/vae/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional, Tuple
import torch
from src.core.component import Component

class VAE(Component, ABC):
    """
    Base interface for VAEs.
    
    This provides a common interface for VAE encoding/decoding,
    similar to how LLM tokenizers have encode/decode methods.
    """
    
    @abstractmethod
    def encode(self, videos: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Encode videos to latent space.
        
        Args:
            videos: List of video tensors [C, T, H, W]
            
        Returns:
            List of latent tensors
        """
        pass
    
    @abstractmethod
    def decode(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Decode latent representations to pixel space.
        
        Args:
            latents: List of latent tensors
            
        Returns:
            List of decoded video tensors
        """
        pass
    
    def tiled_decode(self, latents: torch.Tensor, 
                   tile_size: Tuple[int, int] = (256, 256),
                   tile_stride: Tuple[int, int] = (128, 128)) -> torch.Tensor:
        """
        Decode latents with tiling for memory efficiency.
        
        Args:
            latents: Latent tensor to decode
            tile_size: Size of tiles (height, width)
            tile_stride: Stride between tiles
            
        Returns:
            Decoded video
        """
        # Default implementation falls back to regular decode
        # Subclasses should override with efficient implementation
        if isinstance(latents, torch.Tensor):
            latents = [latents]
        return self.decode(latents)[0]