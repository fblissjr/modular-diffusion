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

@register_component("WanDiT", DiffusionModel)
class WanDiT(DiffusionModel):
    """
    WanDiT Diffusion Transformer model.
    
    This implements the Diffusion Transformer architecture for
    WanVideo's text-to-video generation.
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
        """Build model components matching the checkpoint structure."""
        # Following the exact structure from wan/modules/model.py
        
        # Patch embedding
        self.patch_embedding = nn.Conv3d(
            self.in_channels, 
            self.dim,
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, self.dim * 6)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                dim=self.dim,
                ffn_dim=self.hidden_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                qk_norm=self.qk_norm,
                cross_attn_norm=self.cross_attn_norm
            )
            for _ in range(self.num_layers)
        ])
        
        # Output head
        self.head = Head(self.dim, self.out_channels, self.patch_size)
        
        # Register buffer for position encoding
        d = self.dim // self.num_heads
        self.register_buffer(
            "freqs",
            torch.cat([
                self._rope_params(1024, d - 4 * (d // 6)),
                self._rope_params(1024, 2 * (d // 6)),
                self._rope_params(1024, 2 * (d // 6))
            ], dim=1)
        )
    
    def _rope_params(self, max_seq_len, dim, theta=10000):
        """Generate rotary position embedding parameters."""
        assert dim % 2 == 0
        freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
        )
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def _load_weights(self, model_path: Union[str, Path]):
        """Load model weights with exact key matching."""
        model_path = Path(model_path)
        
        # Find safetensors file
        if model_path.is_dir():
            # Look for safetensors files
            safetensor_files = list(model_path.glob("*.safetensors"))
            if not safetensor_files:
                raise FileNotFoundError(f"No safetensors files found in {model_path}")
                
            # Prefer dtype-specific files in this order: bf16, fp16, fp32
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
        try:
            state_dict = safetensors.torch.load_file(model_path)
        except Exception as e:
            logger.error(f"Error loading safetensors file: {e}")
            return
        
        # Show sample keys
        sample_keys = list(state_dict.keys())[:5]
        logger.info(f"Sample keys from checkpoint: {sample_keys}")
        
        # Map checkpoint keys to our model structure exactly
        mapped_dict = {}
        our_state_dict = self.state_dict()
        
        for ckpt_key, tensor in state_dict.items():
            if ckpt_key.startswith('model.'):
                # Remove model. prefix
                our_key = ckpt_key[6:]
                
                # Check if key exists in our model
                if our_key in our_state_dict:
                    mapped_dict[our_key] = tensor
                    continue
                    
                # Try to map to our structure
                if ckpt_key.startswith('model.diffusion_model.'):
                    # Remove diffusion_model. prefix
                    clean_key = ckpt_key[len('model.diffusion_model.'):]
                    
                    # Try direct mapping
                    if clean_key in our_state_dict:
                        mapped_dict[clean_key] = tensor
        
        # Check if we have all keys needed
        missing_keys = set(our_state_dict.keys()) - set(mapped_dict.keys())
        
        # Try to load state dict
        try:
            missing, unexpected = self.load_state_dict(missing_keys, strict=False)
            
            if missing:
                logger.warning(f"Missing keys: {len(missing)} keys")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)} keys")
                
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            logger.warning("Continuing with uninitialized weights - model may not work correctly")
    
    def forward(self, 
               latents: List[torch.Tensor], 
               timestep: torch.Tensor, 
               text_embeds: List[torch.Tensor], 
               **kwargs) -> Tuple[List[torch.Tensor], Optional[Dict]]:
        """
        Forward pass through diffusion model based on wan/modules/model.py.
        
        Args:
            latents: List of latent tensors
            timestep: Current timestep
            text_embeds: Text embeddings for conditioning
            **kwargs: Additional arguments
                - seq_len: Sequence length
                - is_uncond: Whether this is unconditional generation
        """
        # Process inputs
        results = []
        seq_len = kwargs.get("seq_len", None)
        
        for i, (latent, text_embed) in enumerate(zip(latents, text_embeds)):
            # Ensure device consistency
            t = timestep.to(latent.device)
            if len(t.shape) == 0:
                t = t.view(1)
                
            # Ensure freqs is on the right device
            if self.freqs.device != latent.device:
                self.freqs = self.freqs.to(latent.device)
            
            # Embed time
            time_embed = self.time_embedding(self._sinusoidal_embedding(self.freq_dim, t))
            time_proj = self.time_projection(time_embed)
            
            # Apply patches
            x = self.patch_embedding(latent)
            
            # Calculate grid sizes for position encoding
            grid_sizes = torch.tensor([[x.shape[2], x.shape[3], x.shape[4]]], device=x.device)
            
            # Reshape to sequence
            batch_size = x.shape[0]
            x = x.flatten(2).transpose(1, 2)
            
            # Process through transformer blocks with position encoding
            for block in self.blocks:
                x = block(
                    x,
                    time_proj,
                    torch.tensor([seq_len], device=x.device),
                    grid_sizes,
                    self.freqs,
                    text_embed,
                    None  # context_lens
                )
            
            # Apply head to get output
            output = self.head(x, time_embed)
            
            # Add to results
            results.append(output)
            
        return results, None
    
    def _sinusoidal_embedding(self, dim, position):
        """Generate sinusoidal embedding for position."""
        # Match the implementation in wan/modules/model.py
        half = dim // 2
        pos = position.float()
        
        # Calculate sinusoid embedding
        emb = torch.log(torch.tensor(10000.0)) / half
        emb = torch.exp(torch.arange(half, device=pos.device) * -emb)
        emb = pos.view(-1, 1) * emb.view(1, -1)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            
        return emb
    
    def unpatchify(self, x, grid_sizes):
        """
        Reshape from sequence to video.
        
        Args:
            x: Output tensor with shape [B, L, C]
            grid_sizes: Tensor with video dimensions [B, 3]
            
        Returns:
            Unpatchified tensor with shape [B, C, F, H, W]
        """
        c = self.out_channels
        
        # Unpatchify for each item in batch
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            f, h, w = v
            # Reshape to 5D tensor: [F, H, W, *patch_size, C]
            u = u[:f*h*w].view(f, h, w, *self.patch_size, c)
            # Permute dimensions to [C, F, H*patch_h, W*patch_w]
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, f, h*self.patch_size[1], w*self.patch_size[2])
            out.append(u)
        
        return out
    
class WanAttentionBlock(nn.Module):
    """
    Attention block for WanDiT based on wan/modules/model.py.
    """
    
    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # Layer norms
        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)
        self.norm3 = RMSNorm(dim, eps)
        
        # Attention layers
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps
        )
        
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=cross_attn_norm,
            eps=eps
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            RMSNorm(dim, eps),
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        
        # Modulation parameters
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        # Split modulation tensor
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        
        # Self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs
        )
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = x + y * e[2]
        
        # Cross-attention
        y = self.cross_attn(self.norm2(x), context, context_lens)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = x + y
        
        # Feed-forward
        y = self.ffn(self.norm3(x).float() * (1 + e[4]) + e[3])
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
        
        return x


class SelfAttention(nn.Module):
    """Self-attention implementation from wan/modules/model.py."""
    
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Projection layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        # Normalization
        if qk_norm:
            self.norm_q = RMSNorm(dim, eps)
            self.norm_k = RMSNorm(dim, eps)
    
    def forward(self, x, seq_lens, grid_sizes, freqs):
        # Project to q, k, v
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Apply normalization
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # Apply rotary position embedding
        q = q.view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        k = k.view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        
        # Apply RoPE
        q = self._rope_apply(q, grid_sizes, freqs)
        k = self._rope_apply(k, grid_sizes, freqs)
        
        # Compute attention
        x = self._flash_attention(q, k, v, seq_lens, self.window_size)
        
        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        
        return x
    
    def _rope_apply(self, x, grid_sizes, freqs):
        # Implement rotary position embedding from wan/modules/model.py
        b, s, n, d = x.shape
        x_complex = torch.view_as_complex(x.reshape(b, s, n, d//2, 2))
        
        # Split frequencies
        freqs_split = freqs.split([d//2 - 2*(d//6), d//6, d//6], dim=1)
        
        # Apply to each position
        out_list = []
        for i, (g, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = g * h * w
            
            # Spatial-temporal positions
            pos_f = torch.arange(g, device=x.device).view(g, 1, 1).expand(g, h, w)
            pos_h = torch.arange(h, device=x.device).view(1, h, 1).expand(g, h, w)
            pos_w = torch.arange(w, device=x.device).view(1, 1, w).expand(g, h, w)
            
            # Flatten positions
            pos_f = pos_f.reshape(-1)
            pos_h = pos_h.reshape(-1)
            pos_w = pos_w.reshape(-1)
            
            # Apply frequencies
            freqs_f = freqs_split[0][:g].view(g, 1, -1).expand(g, h*w, -1)
            freqs_h = freqs_split[1][:h].view(1, h, 1, -1).expand(g, h, w, -1).reshape(g*h*w, -1)
            freqs_w = freqs_split[2][:w].view(1, 1, w, -1).expand(g, h, w, -1).reshape(g*h*w, -1)
            
            # Combine frequencies
            combined_freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=1)
            
            # Apply rotation
            x_i = x_complex[i, :seq_len] * combined_freqs
            x_i = torch.view_as_real(x_i).flatten(2)
            
            # Add to output
            out_list.append(x_i)
            
        return torch.stack(out_list)
    
    def _flash_attention(self, q, k, v, seq_lens, window_size):
        # Implement flash attention with local window support
        from torch.nn.functional import scaled_dot_product_attention
        
        # Reshape for multi-head attention
        b, s, n, d = q.shape
        q = q.permute(0, 2, 1, 3)  # [B, N, S, D]
        k = k.permute(0, 2, 1, 3)  # [B, N, S, D]
        v = v.permute(0, 2, 1, 3)  # [B, N, S, D]
        
        # Create mask for local window attention if needed
        if window_size[0] > 0 or window_size[1] > 0:
            mask = self._create_window_mask(s, window_size, q.device)
        else:
            mask = None
        
        # Apply attention
        output = scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        # Reshape output
        output = output.permute(0, 2, 1, 3)  # [B, S, N, D]
        
        return output
    
    def _create_window_mask(self, seq_len, window_size, device):
        # Create mask for local window attention
        mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
        
        # Set local window to 0 (allows attention)
        for i in range(seq_len):
            left = max(0, i - window_size[0])
            right = min(seq_len, i + window_size[1] + 1)
            mask[i, left:right] = 0
            
        return mask


class CrossAttention(nn.Module):
    """Cross-attention implementation from wan/modules/model.py."""
    
    def __init__(self, dim, num_heads, qk_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Projection layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        # Normalization
        if qk_norm:
            self.norm_q = RMSNorm(dim, eps)
            self.norm_k = RMSNorm(dim, eps)
    
    def forward(self, x, context, context_lens=None):
        # Get shapes
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project to q, k, v
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        
        # Apply normalization
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention
        from torch.nn.functional import scaled_dot_product_attention
        output = scaled_dot_product_attention(q, k, v)
        
        # Reshape output
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        output = self.o(output)
        
        return output


class Head(nn.Module):
    """
    Output head based on wan/modules/model.py.
    """
    
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        
        # Layers
        out_channels = patch_size[0] * patch_size[1] * patch_size[2] * out_dim
        self.norm = RMSNorm(dim, eps)
        self.head = nn.Linear(dim, out_channels)
        
        # Modulation parameters
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
    
    def forward(self, x, e):
        # Apply modulation
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        
        return x


class RMSNorm(nn.Module):
    """RMS normalization from wan/modules/model.py."""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Apply RMS normalization
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale
        return x * norm * self.weight

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        
        # Generate log-spaced frequencies
        frequencies = torch.exp(
            torch.arange(0, half_dim).float() * (-math.log(10000) / (half_dim - 1))
        )
        
        self.register_buffer("frequencies", frequencies)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep t."""
        # Ensure t is a 1D tensor
        t = t.view(-1, 1)
        
        # Make sure frequencies is on the same device as t
        frequencies = self.frequencies.to(t.device)
        
        # Calculate embeddings
        freqs = t * frequencies.unsqueeze(0)
        embeddings = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)
        
        # Handle odd dimensions
        if self.dim % 2:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=1)
            
        return embeddings


class RMSNorm(nn.Module):
    """RMSNorm layer."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm."""
        # Calculate RMS along last dimension
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    """
    Transformer block matching the checkpoint structure.
    """
    
    def __init__(self, dim: int, num_heads: int, hidden_dim: int,
                 qk_norm: bool = False, cross_attn_norm: bool = False):
        super().__init__()
        # Layer norms for attention blocks
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        
        # Self-attention - note that context dimension is same as dim
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm
        )
        
        # Cross-attention - critical change: context dimension is same as dim
        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=dim,  # This should match the checkpoint: 1536 not 4096
            num_heads=num_heads,
            qk_norm=cross_attn_norm
        )
        
        # Feed-forward network
        self.ff = FeedForward(
            dim=dim,
            hidden_dim=hidden_dim
        )
    
    def forward(self, x, context=None):
        # Self-attention
        x = x + self.attn(self.norm1(x))
        
        # Cross-attention
        x = x + self.cross_attn(self.norm2(x), context)
        
        # Feed-forward
        x = x + self.ff(self.norm3(x))
        
        return x


class SelfAttention(nn.Module):
    """Self-attention layer."""
    
    def __init__(self, dim, num_heads, qk_norm=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        
        # Projection layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        # QK norm if enabled - IMPORTANT: use dim not head_dim for weight sizes!
        if qk_norm:
            self.norm_q = RMSNorm(dim)  # Use dim instead of head_dim
            self.norm_k = RMSNorm(dim)  # Use dim instead of head_dim
        
        # Initialize weights
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project to q, k, v
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Apply QK norm if enabled - apply before reshaping
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape to original format
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        out = self.o(out)
        
        return out


class CrossAttention(nn.Module):
    """Cross-attention layer."""
    
    def __init__(self, dim, context_dim, num_heads, qk_norm=False):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        
        # Projection layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(context_dim, dim)
        self.v = nn.Linear(context_dim, dim)
        self.o = nn.Linear(dim, dim)
        
        # QK norm if enabled - IMPORTANT: use dim not head_dim for weight sizes!
        if qk_norm:
            self.norm_q = RMSNorm(dim)  # Use dim instead of head_dim
            self.norm_k = RMSNorm(dim)  # Use dim instead of head_dim
        
        # Initialize weights
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, context):
        batch_size, seq_len = x.shape[0], x.shape[1]
        context_len = context.shape[1]
        
        # Project to q, k, v
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        
        # Apply QK norm if enabled - apply before reshaping
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape to original format
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        out = self.o(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x