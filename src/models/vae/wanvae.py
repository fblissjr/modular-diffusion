# src/models/vae/wanvae.py
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path

from src.core.component import Component
from src.models.vae.base import VAE
from src.core.registry import register_component
from src.models.wan.vae import WanVAE as OriginalWanVAE

logger = logging.getLogger(__name__)

@register_component("WanVAEAdapter", VAE)
class WanVAEAdapter(VAE):
    """
    Adapter for WanVideo VAE.
    
    This adapts the original WanVAE implementation to our modular
    interface, allowing it to be used consistently within our pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WanVAE adapter.
        
        Args:
            config: VAE configuration
        """
        super().__init__(config)
        
        # Get configuration parameters
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for WanVAE")
            
        # VAE parameters
        z_dim = config.get("z_dim", 16)
        
        # Create original VAE with our parameters
        self.vae = OriginalWanVAE(
            z_dim=z_dim,
            vae_pth=model_path,
            dtype=self.dtype,
            device=self.device
        )
        
        logger.info(f"Initialized WanVAE with z_dim={z_dim}, checkpoint={model_path}")
        
    def encode(self, videos: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        encode videos to latent tensors with proper cache management.
        """
        # ensure videos are on the correct device/dtype
        videos = [v.to(self.device, self.dtype) for v in videos]
        
        # clear cache before encoding
        if hasattr(self.vae, 'clear_cache'):
            self.vae.clear_cache()
        
        # use original vae encode
        latents = self.vae.encode(videos)
        
        return latents
        
    def decode(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        decode latent tensors to videos with proper cache management.
        """
        # ensure latents are on correct device/dtype
        latents = [z.to(self.device, self.dtype) for z in latents]
        
        # clear cache before decoding
        if hasattr(self.vae, 'clear_cache'):
            self.vae.clear_cache()
        
        # use original vae decode
        videos = self.vae.decode(latents)
        
        return videos
        
        
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
        # This is where we could implement improved tiling for memory efficiency
        # Example implementation for future enhancement:
        
        # Prepare output tensor
        if isinstance(latents, torch.Tensor):
            latents = [latents]
            
        # For now, just use regular decode
        # In future, implement chunked decoding with overlapping tiles
        return self.decode(latents)[0]
        
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> "WanVAEAdapter":
        """
        Move VAE to specified device and dtype.
        
        Args:
            device: Target device
            dtype: Target dtype
            
        Returns:
            Self for chaining
        """
        super().to(device, dtype)
        
        if device is not None or dtype is not None:
            # Update original VAE
            if device is not None:
                self.vae.device = device
                self.vae.model = self.vae.model.to(device)
                self.vae.mean = self.vae.mean.to(device)
                self.vae.std = self.vae.std.to(device)
                
            if dtype is not None:
                self.vae.dtype = dtype
                self.vae.mean = self.vae.mean.to(dtype)
                self.vae.std = self.vae.std.to(dtype)
                # Don't change model dtype as it requires careful handling
                
            # Rebuild scale
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
            
        return self
        
    def requires_grad_(self, requires_grad: bool = True) -> "WanVAEAdapter":
        """
        Set requires_grad flag for parameters.
        
        Args:
            requires_grad: Whether parameters require gradients
            
        Returns:
            Self for chaining
        """
        if hasattr(self, "vae") and hasattr(self.vae, "model"):
            self.vae.model.requires_grad_(requires_grad)
        return self