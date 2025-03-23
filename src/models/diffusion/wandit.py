# src/models/diffusion/wandit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import safetensors.torch
from dataclasses import dataclass

from src.core.component import Component
from src.models.diffusion.base import DiffusionModel
from src.core.registry import register_component

logger = logging.getLogger(__name__)

@dataclass
class WanDiTConfig:
    """Configuration for WanDiT model components."""
    dim: int
    num_heads: int
    hidden_dim: int
    patch_size: Tuple[int, int, int]
    window_size: Tuple[int, int]
    num_layers: int
    qk_norm: bool
    cross_attn_norm: bool
    in_channels: int = 16  # Default latent channels
    out_channels: int = 16  # Default output channels
    freq_dim: int = 256    # Default frequency dimensions

class RMSNorm(nn.Module):
    """RMSNorm layer from T5."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to input."""
        # Calculate RMS along last dimension
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale
        # Apply normalization and scaling
        return x * norm * self.weight

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embeddings."""
    
    def __init__(self, dim: int, max_time: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_time = max_time
        half_dim = dim // 2
        
        # Generate log-spaced frequencies
        frequencies = torch.exp(
            torch.arange(0, half_dim).float() * (-math.log(max_time) / (half_dim - 1))
        )
        
        self.register_buffer("frequencies", frequencies)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep t."""
        # Ensure t is a 1D tensor
        t = t.view(-1)
        
        # Calculate embeddings
        freqs = t.unsqueeze(1) * self.frequencies.unsqueeze(0)
        embeddings = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)
        
        # Handle odd dimensions
        if self.dim % 2:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=1)
            
        return embeddings

class CrossAttention(nn.Module):
    """Cross-attention mechanism for text conditioning."""
    
    def __init__(self, query_dim: int, context_dim: int, dim_head: int, heads: int, 
                dropout: float = 0.0, use_qk_norm: bool = False):
        super().__init__()
        inner_dim = dim_head * heads
        
        # Projections
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        # Optional QK normalization
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(dim_head)
            self.k_norm = RMSNorm(dim_head)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention."""
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape to multihead
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        
        # Apply QK normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project back
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return self.to_out(out)

class WanViTBlock(nn.Module):
    """
    Transformer block for WanDiT.
    
    This combines standard transformer operations with specialized
    attention mechanisms for video generation.
    """
    
    def __init__(self, dim: int, num_heads: int, hidden_dim: int,
                 qk_norm: bool = False, cross_attn_dim: Optional[int] = None,
                 cross_attn_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        # Self-attention and feedforward dimension
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Layer norms
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim) if cross_attn_dim else None
        
        # Self-attention
        self.attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            use_qk_norm=qk_norm
        )
        
        # Cross-attention if text conditioning provided
        if cross_attn_dim:
            self.cross_attn = CrossAttention(
                query_dim=dim,
                context_dim=cross_attn_dim,
                dim_head=dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                use_qk_norm=cross_attn_norm
            )
        else:
            self.cross_attn = None
        
        # Feed-forward network
        self.ff = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [B, L, D]
            context: Optional conditioning context
            
        Returns:
            Transformed tensor
        """
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x))
        
        # Cross-attention if context provided
        if self.cross_attn and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
            
        # Feed-forward
        x = x + self.ff(x)
        
        return x

@register_component("WanDiT", DiffusionModel)
class WanDiT(DiffusionModel):
    """
    WanDiT Diffusion Transformer model.
    
    This implements the Diffusion Transformer architecture for
    WanVideo's text-to-video generation, similar to how LLM
    decoders generate text with conditioning information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WanDiT model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Get model path and load config
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for WanDiT")
            
        # Load model config
        model_config = config.get("model_config")
        if not model_config:
            from src.configs import get_model_config
            model_config = get_model_config(model_path, "wanvideo")
            
        # Set up parameters from config
        self.dim = model_config.dim
        self.num_heads = model_config.num_heads
        self.hidden_dim = model_config.ffn_dim
        self.patch_size = model_config.patch_size
        self.window_size = model_config.window_size
        self.num_layers = model_config.num_layers
        self.qk_norm = model_config.qk_norm
        self.cross_attn_norm = model_config.cross_attn_norm
        
        # Set in/out channels
        self.in_channels = config.get("in_channels", 16)
        self.out_channels = config.get("out_channels", 16)
        
        # Frequency dimensions for time embedding
        self.freq_dim = model_config.freq_dim
        
        # Create model components
        self._build_model()
        
        # Load weights
        self._load_weights(model_path)
        
        # Set to eval mode
        self.eval()
        
        logger.info(f"Initialized WanDiT with dim={self.dim}, num_layers={self.num_layers}")
        
    def _build_model(self):
        """Build model components."""
        # Embeddings and projections
        self.time_embedding = SinusoidalEmbedding(self.freq_dim)
        self.time_proj = nn.Linear(self.freq_dim, self.dim)
        
        # Patcher (for embedding input patches)
        self.patcher = nn.Conv3d(
            self.in_channels, 
            self.dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WanViTBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                qk_norm=self.qk_norm,
                cross_attn_dim=1536,  # T5-XXL dimension
                cross_attn_norm=self.cross_attn_norm
            )
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.unpatcher = nn.Conv3d(
            self.dim, 
            self.out_channels,
            kernel_size=1
        )
        
    def _load_weights(self, model_path: Union[str, Path]):
        """
        Load model weights.
        
        Args:
            model_path: Path to model weights
        """
        model_path = Path(model_path)
        
        # Find safetensors file
        if model_path.is_dir():
            # Look for safetensors files
            safetensor_files = list(model_path.glob("*.safetensors"))
            if not safetensor_files:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
                
            # Prefer dtype-specific files in this order: bf16, fp16, fp32
            # This matches our typical use case where we want to load the bf16 model by default
            for dtype_suffix in ["bf16", "fp16", "fp32"]:
                for file in safetensor_files:
                    if dtype_suffix in file.name:
                        model_path = file
                        break
                if isinstance(model_path, Path) and not model_path.is_dir():
                    break
                    
            # If we didn't find a specific file, use the first one
            if model_path.is_dir():
                model_path = safetensor_files[0]
        
        # Load weights
        logger.info(f"Loading model weights from {model_path}")
        state_dict = safetensors.torch.load_file(model_path)
        
        # Try to load state dict
        try:
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            
            if missing:
                logger.warning(f"Missing keys: {len(missing)} keys")
                
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)} keys")
                
        except Exception as e:
            logger.error(f"Failed to load state dict: {e}")
            raise
    
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
                - is_uncond: Whether this is unconditional generation
                - current_step_percentage: Current step progress
            
        Returns:
            Tuple of (predicted noise, optional auxiliary outputs)
        """
        # Process inputs
        results = []
        
        for i, (latent, text_embed) in enumerate(zip(latents, text_embeds)):
            # Get single timestep value
            t = timestep.to(latent.device)
            if len(t.shape) == 0:
                t = t.view(1)
                
            # Embed time
            time_embed = self.time_embedding(t)
            time_embed = self.time_proj(time_embed).unsqueeze(1)  # [B, 1, D]
            
            # Get shape info
            batch_size, channels, frames, height, width = latent.shape
            
            # Apply patching
            x = self.patcher(latent)  # [B, D, F', H', W']
            
            # Reshape to sequence
            patch_frames = x.shape[2]
            patch_height = x.shape[3]
            patch_width = x.shape[4]
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, F', H', W', D]
            x = x.view(batch_size, -1, self.dim)  # [B, F'*H'*W', D]
            
            # Process through transformer blocks
            for block in self.blocks:
                # Add time embedding
                block_input = x + time_embed
                
                # Process through block
                x = block(block_input, text_embed)
            
            # Reshape back to 3D
            x = x.view(batch_size, patch_frames, patch_height, patch_width, self.dim)
            x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, D, F', H', W']
            
            # Un-patch
            output = self.unpatcher(x)
            
            # Store result
            results.append(output)
            
        return results, None
        
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> "WanDiT":
        """
        Move model to specified device and dtype.
        
        Args:
            device: Target device
            dtype: Target dtype
            
        Returns:
            Self for chaining
        """
        super().to(device, dtype)
        
        # Move model
        if device is not None or dtype is not None:
            # Use parent method
            nn.Module.to(self, device=device, dtype=dtype)
            
        return self