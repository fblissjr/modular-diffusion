"""
WanVideo Diffusion Transformer Model

This module implements the core DiT (Diffusion Transformer) model
used in WanVideo. The DiT model is a transformer-based architecture
that performs the denoising process in diffusion models.

Key components:
- WanDiT: Main diffusion transformer model
- WanAttentionBlock: Transformer block with self and cross attention
- WanSelfAttention: Self-attention mechanism with RoPE
- WanCrossAttention: Cross-attention for text conditioning

The implementation includes memory optimizations like block swapping
to run large models on consumer hardware.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog
from typing import List, Dict, Optional, Tuple, Union, Any
from einops import rearrange, repeat

logger = structlog.get_logger()

class WanRMSNorm(nn.Module):
    """
    Root Mean Square Normalization used in WanVideo.
    
    RMSNorm is similar to LayerNorm but normalizes using RMS
    rather than mean and variance, which can improve training.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            dim: Hidden dimension size
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [B, L, C]
            
        Returns:
            Normalized tensor
        """
        # Convert to float32 for better numerical stability
        x_float = x.float()
        # Calculate RMS along the last dimension
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale with learned weight
        normalized = (x_float / rms) * self.weight
        # Convert back to original dtype
        return normalized.type_as(x)

class WanLayerNorm(nn.LayerNorm):
    """
    Layer Normalization with improved dtype handling.
    
    This variant ensures proper handling of mixed precision.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """
        Initialize layer norm.
        
        Args:
            dim: Hidden dimension size
            eps: Small constant for numerical stability
            elementwise_affine: Whether to include learnable parameters
        """
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape [B, L, C]
            
        Returns:
            Normalized tensor
        """
        # Always normalize in float32 for stability
        normalized = super().forward(x.float())
        # Convert back to original dtype
        return normalized.type_as(x)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input for RoPE.
    
    Args:
        x: Input tensor
    
    Returns:
        Tensor with half the features rotated
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Rotary Position Embedding (RoPE) encodes positions as rotations
    in the complex plane, which helps transformers handle sequences
    effectively.
    
    Args:
        q: Query tensor of shape [B, H, L, D]
        k: Key tensor of shape [B, H, L, D]
        freqs: Frequency tensor for rotary embeddings
    
    Returns:
        Tuple of query and key tensors with position information
    """
    # Extract dimensions
    batch_size, n_heads, seq_len, d = q.shape
    # Apply RoPE to query
    q_embed = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    # Apply RoPE to key 
    k_embed = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    
    return q_embed, k_embed

def get_rope_freqs(max_seq_len: int, dim: int, base: int = 10000, device=None) -> torch.Tensor:
    """
    Generate frequency tensor for rotary position embeddings.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension of the embeddings (divided by 2 for RoPE)
        base: Base for the frequency calculation
        device: Device to place tensors on
    
    Returns:
        Frequency tensor for rotary embeddings
    """
    # RoPE uses half the dimensions for rotation
    half_dim = dim // 2
    # Create frequency bands
    freqs = torch.arange(half_dim, device=device) / half_dim
    freqs = 1.0 / (base ** freqs)
    
    # Create position indices
    t = torch.arange(max_seq_len, device=device)
    # Outer product to get frequencies for each position
    freqs = torch.outer(t, freqs)
    # Convert to complex format for later calculations
    freqs = torch.cat([freqs, freqs], dim=-1)
    
    return freqs

class WanSelfAttention(nn.Module):
    """
    Self-attention mechanism with RoPE used in WanVideo.
    
    This implementation includes:
    - Rotary position embeddings
    - Optional QK normalization for stability
    - Support for different attention implementations
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        attention_mode: str = "sdpa",
    ):
        """
        Initialize self-attention.
        
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            head_dim: Dimension per head (if None, calculated from dim/num_heads)
            window_size: Size of local attention window (-1 for global)
            qk_norm: Whether to apply normalization to Q and K
            eps: Small constant for numerical stability
            attention_mode: Type of attention implementation
        """
        super().__init__()
        
        # Calculate head dimension if not provided
        head_dim = head_dim or dim // num_heads
        inner_dim = num_heads * head_dim
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode
        
        # Projection matrices
        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        self.o = nn.Linear(inner_dim, dim)
        
        # Optional QK normalization
        self.norm_q = WanRMSNorm(inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(inner_dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        seq_lens: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply self-attention with RoPE.
        
        Args:
            x: Input tensor of shape [B, L, C]
            seq_lens: Length of each sequence in batch
            freqs: Pre-computed frequencies for RoPE
        
        Returns:
            Attended tensor
        """
        batch_size, seq_length, _ = x.shape
        
        # Project to query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings if provided
        if freqs is not None:
            # Reshape for applying RoPE
            q = q.transpose(1, 2)  # [B, H, L, D]
            k = k.transpose(1, 2)  # [B, H, L, D]
            
            # Apply RoPE
            q, k = apply_rotary_pos_emb(q, k, freqs)
        else:
            # Generate freqs if not provided
            max_len = seq_length
            rope_freqs = get_rope_freqs(
                max_len, self.head_dim, device=x.device
            )
            
            # Reshape for applying RoPE
            q = q.transpose(1, 2)  # [B, H, L, D]
            k = k.transpose(1, 2)  # [B, H, L, D]
            
            # Apply RoPE
            q, k = apply_rotary_pos_emb(q, k, rope_freqs)
        
        # Apply attention
        if self.attention_mode == "sdpa":
            # PyTorch's scaled dot product attention
            v = v.transpose(1, 2)  # [B, H, L, D]
            
            # Create attention mask for varying sequence lengths
            attn_mask = None
            if seq_lens is not None:
                attn_mask = torch.zeros(
                    batch_size, seq_length, seq_length, 
                    device=x.device, dtype=torch.bool
                )
                for i, length in enumerate(seq_lens):
                    attn_mask[i, :length, :length] = True
                # Broadcast across heads
                attn_mask = attn_mask.unsqueeze(1)
            
            # Apply attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
            
            # Reshape output
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        
        elif self.attention_mode == "flash_attn":
            # This would be implemented with flash attention
            # For now, fall back to SDPA
            logger.warning("Flash attention not implemented, falling back to SDPA")
            v = v.transpose(1, 2)  # [B, H, L, D]
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        
        # Project back to output dimension
        out = self.o(out)
        
        return out

class WanCrossAttention(nn.Module):
    """
    Cross-attention mechanism for conditioning on text embeddings.
    
    This allows the diffusion model to incorporate text guidance
    from the T5 encoder outputs.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        qk_norm: bool = True,
        eps: float = 1e-6,
        attention_mode: str = "sdpa",
    ):
        """
        Initialize cross-attention.
        
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            head_dim: Dimension per head (if None, calculated from dim/num_heads)
            qk_norm: Whether to apply normalization to Q and K
            eps: Small constant for numerical stability
            attention_mode: Type of attention implementation
        """
        super().__init__()
        
        # Calculate head dimension if not provided
        head_dim = head_dim or dim // num_heads
        inner_dim = num_heads * head_dim
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode
        
        # Projection matrices
        self.q = nn.Linear(dim, inner_dim)
        self.k = nn.Linear(dim, inner_dim)
        self.v = nn.Linear(dim, inner_dim)
        self.o = nn.Linear(inner_dim, dim)
        
        # Optional QK normalization
        self.norm_q = WanRMSNorm(inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(inner_dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention between input and context.
        
        Args:
            x: Input tensor of shape [B, L1, C]
            context: Context tensor (e.g., text embeddings) of shape [B, L2, C]
            context_lens: Length of each context sequence in batch
        
        Returns:
            Attended tensor
        """
        batch_size, seq_length, _ = x.shape
        
        # Project query from input, key and value from context
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [B, H, L1, D]
        k = k.transpose(1, 2)  # [B, H, L2, D]
        v = v.transpose(1, 2)  # [B, H, L2, D]
        
        # Apply attention
        if self.attention_mode == "sdpa":
            # Create attention mask for varying context lengths
            attn_mask = None
            if context_lens is not None:
                max_ctx_len = k.size(2)
                attn_mask = torch.zeros(
                    batch_size, 1, seq_length, max_ctx_len,
                    device=x.device, dtype=torch.bool
                )
                for i, length in enumerate(context_lens):
                    attn_mask[i, :, :, :length] = True
            
            # Apply attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
        
        elif self.attention_mode == "flash_attn":
            # This would be implemented with flash attention
            # For now, fall back to SDPA
            logger.warning("Flash attention not implemented, falling back to SDPA")
            out = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        
        # Project back to output dimension
        out = self.o(out)
        
        return out

class WanFeedForward(nn.Module):
    """
    Feed-forward network used in WanVideo transformer blocks.
    
    This is a standard FFN with GELU activation, similar to those
    used in modern transformers like GPT models.
    """
    
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        """
        Initialize feed-forward network.
        
        Args:
            dim: Input/output dimension
            ffn_dim: Hidden dimension (usually 4x larger than dim)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dim = dim
        self.ffn_dim = ffn_dim
        
        # MLP layers
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.
        
        Args:
            x: Input tensor of shape [B, L, C]
        
        Returns:
            Transformed tensor
        """
        # First projection and activation
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        
        # Second projection
        out = self.fc2(h)
        out = self.dropout(out)
        
        return out

class WanAttentionBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and FFN.
    
    This follows the standard transformer architecture with:
    1. Self-attention for spatial/temporal relationships
    2. Cross-attention for incorporating text guidance
    3. Feed-forward network for additional transformations
    
    Each component has residual connections and layer normalization.
    """
    
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
        attention_mode: str = "sdpa",
    ):
        """
        Initialize transformer block.
        
        Args:
            dim: Hidden dimension size
            ffn_dim: Feed-forward network dimension
            num_heads: Number of attention heads
            window_size: Size of local attention window (-1 for global)
            qk_norm: Whether to apply normalization to Q and K
            cross_attn_norm: Whether to apply normalization before cross-attention
            dropout: Dropout probability
            eps: Small constant for numerical stability
            attention_mode: Type of attention implementation
        """
        super().__init__()
        
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        # Layer normalization
        self.norm1 = WanLayerNorm(dim, eps=eps)
        self.norm2 = WanLayerNorm(dim, eps=eps)
        self.norm3 = WanLayerNorm(dim, eps=eps, elementwise_affine=cross_attn_norm)
        
        # Self-attention
        self.self_attn = WanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            eps=eps,
            attention_mode=attention_mode,
        )
        
        # Cross-attention
        self.cross_attn = WanCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            eps=eps,
            attention_mode=attention_mode,
        )
        
        # Feed-forward network
        self.ffn = WanFeedForward(dim, ffn_dim, dropout=dropout)
        
        # Modulation parameters (used for conditioning)
        # These act as learned scale and shift factors for the model
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply transformer block.
        
        Args:
            x: Input tensor of shape [B, L, C]
            e: Time embedding of shape [B, 6, C]
            seq_lens: Length of each sequence in batch
            freqs: Pre-computed frequencies for RoPE
            context: Context tensor (text embeddings) of shape [B, L2, C]
            context_lens: Length of each context sequence in batch
        
        Returns:
            Transformed tensor
        """
        # Extract modulation factors
        # e contains the time embeddings that modulate the block's behavior
        e = (self.modulation + e).chunk(6, dim=1)
        
        # Self-attention with modulation
        # Apply scale and shift from time embeddings
        h = self.norm1(x).float() * (1 + e[1]) + e[0]
        h = self.self_attn(h, seq_lens, freqs)
        x = x + (h * e[2])
        
        # Cross-attention
        h = self.norm3(x)
        h = self.cross_attn(h, context, context_lens)
        x = x + h
        
        # Feed-forward with modulation
        h = self.norm2(x).float() * (1 + e[4]) + e[3]
        h = self.ffn(h)
        x = x + (h * e[5])
        
        return x

class PatchEmbedding(nn.Module):
    """
    3D patch embedding for video input.
    
    This converts raw pixel values into embeddings by dividing
    the video into small 3D patches and projecting each patch.
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
    ):
        """
        Initialize patch embedding.
        
        Args:
            in_channels: Number of input channels (typically 16 for latents)
            embed_dim: Dimension of output embeddings
            patch_size: Size of each patch (temporal, height, width)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Projection with 3D convolution
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patch embedding to video latents.
        
        Args:
            x: Video latent tensor of shape [B, C, T, H, W]
        
        Returns:
            Embedded patches of shape [B, L, D]
            where L = T*H*W / (pt*ph*pw)
        """
        # Project patches with convolution
        x = self.proj(x)
        
        # Reshape to sequence format
        batch_size, channels, frames, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, L, C]
        
        return x

class Head(nn.Module):
    """
    Output head for the diffusion model.
    
    This projects from the model dimension back to the output
    space (typically the same as the input latent space).
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: int,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
    ):
        """
        Initialize model head.
        
        Args:
            dim: Input dimension (model's hidden dim)
            out_dim: Output dimension (typically input channels)
            patch_size: Size of each patch for proper reshaping
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        
        # Calculate output dimension including patch size
        # Each output vector will be unpatchified later
        output_dim = out_dim * patch_size[0] * patch_size[1] * patch_size[2]
        
        # Layers
        self.norm = WanLayerNorm(dim, eps=eps)
        self.head = nn.Linear(dim, output_dim)
        
        # Modulation parameters (similar to transformer blocks)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
    
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Apply model head.
        
        Args:
            x: Input tensor of shape [B, L, C]
            e: Time embedding of shape [B, C]
        
        Returns:
            Output tensor with correct shape for unpatchify
        """
        # Add batch dimension to e if needed
        e = e.unsqueeze(1) if e.dim() == 2 else e
        
        # Extract modulation factors
        e = (self.modulation + e).chunk(2, dim=1)
        
        # Apply normalization and modulation
        x = self.norm(x) * (1 + e[1]) + e[0]
        
        # Project to output dimension
        x = self.head(x)
        
        return x

class TimeEmbedding(nn.Module):
    """
    Time embedding for diffusion models.
    
    This converts timesteps to embeddings that modulate the
    transformer blocks, allowing the model to condition on
    the denoising timestep.
    """
    
    def __init__(self, dim: int, freq_dim: int = 256):
        """
        Initialize time embedding.
        
        Args:
            dim: Model dimension
            freq_dim: Frequency dimension for sinusoidal embeddings
        """
        super().__init__()
        
        self.dim = dim
        self.freq_dim = freq_dim
        
        # Embedding layers
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        
        # Projection for modulation factors
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),  # 6 modulation factors per block
        )
    
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time embeddings.
        
        Args:
            t: Timestep tensor of shape [B]
        
        Returns:
            Tuple of (base embedding, modulation factors)
        """
        # Create sinusoidal embeddings (similar to positional embeddings)
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Project to model dimension
        embedding = self.mlp(embedding)
        
        # Create modulation factors
        modulation = self.projection(embedding).unflatten(1, (6, self.dim))
        
        return embedding, modulation

class WanDiT(nn.Module):
    """
    WanVideo Diffusion Transformer (DiT) model.
    
    This is the main diffusion model that predicts the noise
    (or velocity field in flow matching) at each timestep. It uses
    a transformer architecture with self and cross attention.
    
    Features:
    - 3D patch embedding for video
    - Transformer blocks with self and cross attention
    - Time conditioning via embeddings
    - Text conditioning via cross-attention
    - Memory-efficient design with optional block swapping
    """
    
    def __init__(
        self,
        model_type: str = "t2v",
        in_dim: int = 16,   # Input channels (latent dimension)
        dim: int = 2048,    # Model dimension
        ffn_dim: int = 8192,  # Feed-forward dimension
        freq_dim: int = 256,  # Frequency dimension for time embeddings
        text_dim: int = 4096,  # Text embedding dimension
        out_dim: int = 16,   # Output channels
        num_heads: int = 16,  # Number of attention heads
        num_layers: int = 32,  # Number of transformer blocks
        patch_size: Tuple[int, int, int] = (1, 2, 2),  # Patch size
        window_size: Tuple[int, int] = (-1, -1),  # Window size for local attention
        qk_norm: bool = True,  # Normalize Q and K
        cross_attn_norm: bool = True,  # Normalize before cross-attention
        dropout: float = 0.0,  # Dropout probability
        eps: float = 1e-6,  # Epsilon for layer norm
        attention_mode: str = "sdpa",  # Attention implementation
        main_device: torch.device = torch.device("cuda"),  # Main computation device
        offload_device: torch.device = torch.device("cpu"),  # Offload device
    ):
        """
        Initialize WanVideo DiT model.
        
        Args:
            model_type: Model variant ("t2v" or "i2v")
            in_dim: Input latent channels
            dim: Model hidden dimension
            ffn_dim: Feed-forward network dimension
            freq_dim: Frequency dimension for time embeddings
            text_dim: Text embedding dimension
            out_dim: Output latent channels
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            patch_size: Size of each patch (temporal, height, width)
            window_size: Size of local attention window (-1 for global)
            qk_norm: Whether to apply normalization to Q and K
            cross_attn_norm: Whether to apply normalization before cross-attention
            dropout: Dropout probability
            eps: Small constant for numerical stability
            attention_mode: Type of attention implementation
            main_device: Main computation device
            offload_device: Device for offloading blocks
        """
        super().__init__()
        
        self.logger = logger.bind(component="WanDiT")
        
        # Store configuration
        self.model_type = model_type
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.window_size = window_size
        self.attention_mode = attention_mode
        self.main_device = main_device
        self.offload_device = offload_device
        
        # Block offloading configuration
        self.blocks_to_swap = -1  # Disabled by default
        self.use_non_blocking = True  # For async transfers
        
        # Embeddings
        self.patch_embedding = PatchEmbedding(in_dim, dim, patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.time_embedding = TimeEmbedding(dim, freq_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                dropout=dropout,
                eps=eps,
                attention_mode=attention_mode,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.head = Head(dim, out_dim, patch_size, eps)
        
        # Initialize RoPE frequencies
        d = dim // num_heads
        self.freq_cis = None  # Will be computed on first forward pass
        
        self.logger.info(f"Initialized WanDiT model",
                        model_type=model_type,
                        dim=dim,
                        num_heads=num_heads,
                        num_layers=num_layers)
    
    def configure_block_swap(self, blocks_to_swap: int, non_blocking: bool = True):
        """
        Configure block swapping for memory efficiency.
        
        This allows offloading a portion of the transformer blocks to 
        CPU to reduce VRAM usage. The blocks are loaded back to GPU
        as needed during inference.
        
        Args:
            blocks_to_swap: Number of transformer blocks to swap (-1 to disable)
            non_blocking: Whether to use non-blocking transfers
        """
        self.blocks_to_swap = blocks_to_swap
        self.use_non_blocking = non_blocking
        
        if blocks_to_swap > 0:
            self.logger.info(f"Configuring block swapping",
                           blocks_to_swap=blocks_to_swap,
                           non_blocking=non_blocking)
            
            # Move swap blocks to offload device
            for i, block in enumerate(self.blocks):
                if i < blocks_to_swap:
                    block.to(self.offload_device)
                    self.logger.debug(f"Moved block {i} to {self.offload_device}")
                else:
                    block.to(self.main_device)
    
    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> List[torch.Tensor]:
        """
        Reconstruct video tensors from patch embeddings.
        
        Args:
            x: Tensor of shape [B, L, C*prod(patch_size)]
            grid_sizes: Original spatial-temporal grid dimensions,
                        shape [B, 3] (3 dimensions: F, H, W)
        
        Returns:
            List of video tensors with shape [C, F, H, W]
        """
        output_dim = self.out_dim
        patch_size = self.patch_size
        
        # Process each item in batch
        result = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            # Get patches for this item, limiting to the actual grid size
            num_patches = f * h * w
            patches = x[i, :num_patches]
            
            # Reshape to reconstruct the video from patches
            # [L, C*pf*ph*pw] -> [F, H, W, pf, ph, pw, C]
            reshaped = patches.view(f, h, w, *patch_size, output_dim)
            
            # Permute to get [C, F*pf, H*ph, W*pw]
            permuted = reshaped.permute(6, 0, 3, 1, 4, 2, 5).contiguous()
            
            # Reshape to final dimensions [C, F', H', W']
            final_shape = (
                output_dim,
                f * patch_size[0],
                h * patch_size[1],
                w * patch_size[2]
            )
            video = permuted.view(final_shape)
            
            result.append(video)
        
        return result
    
    def forward(
        self,
        x: List[torch.Tensor],  # List of video latents, each [C, T, H, W]
        t: torch.Tensor,        # Timesteps [B]
        context: List[torch.Tensor],  # Text embeddings
        seq_len: int,           # Maximum sequence length
        is_uncond: bool = False,  # Whether using unconditional guidance
        current_step_percentage: float = 0.0,  # Progress through sampling
        clip_fea: Optional[torch.Tensor] = None,  # CLIP features for I2V
        y: Optional[List[torch.Tensor]] = None,  # Conditional inputs for I2V
        device = None,           # Override device if needed
        freqs: Optional[torch.Tensor] = None,  # Pre-computed RoPE frequencies
        current_step: int = 0,   # Current sampling step
    ) -> Tuple[List[torch.Tensor], None]:  # Output video latents
        """
        Forward pass through the diffusion model.
        
        This prediction process takes noisy latents and predicts the 
        flow field that moves toward the clean data distribution.
        
        Args:
            x: List of video latent tensors, each with shape [C, T, H, W]
            t: Diffusion timesteps tensor of shape [B]
            context: List of text embeddings each with shape [L, C]
            seq_len: Maximum sequence length
            is_uncond: Whether this is an unconditional guidance pass
            current_step_percentage: Percentage of sampling completed
            clip_fea: CLIP image features for image-to-video mode
            y: Conditional video inputs for image-to-video mode
            device: Override device if needed
            freqs: Pre-computed frequencies for RoPE
            current_step: Current sampling step index
            
        Returns:
            Tuple of (denoised video latents, None)
        """
        # Set device if provided
        device = device or self.main_device
        
        # For I2V models, combine inputs
        if self.model_type == 'i2v' and y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # Extract grid sizes (F, H, W) for each item
        grid_sizes = torch.stack([
            torch.tensor(u.shape[1:], dtype=torch.long) for u in x
        ]).to(device)
        
        # Patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        # Get sequence lengths for each item
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        max_seq_len = seq_lens.max().item()
        
        # Pad sequences to same length and combine into batch
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ]).to(device)
        
        # Time embeddings
        e, e0 = self.time_embedding(t.to(device))
        
        # Process text embeddings
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(seq_len - u.size(0), u.size(1))]) 
            for u in context
        ]).to(device))
        
        # Add CLIP features for I2V models
        if clip_fea is not None and self.model_type == 'i2v':
            # Process CLIP features (implementation would depend on CLIP model)
            self.logger.debug("Processing CLIP features for I2V model")
            # This is a placeholder - actual implementation would process clip_fea
        
        # Process through transformer blocks
        for b, block in enumerate(self.blocks):
            # Load block to main device if using block swapping
            if self.blocks_to_swap > 0 and b < self.blocks_to_swap:
                block.to(self.main_device)
            
            # Apply transformer block
            x = block(
                x=x, 
                e=e0,
                seq_lens=seq_lens,
                freqs=freqs,
                context=context,
                context_lens=None,  # Context lengths would be handled here if needed
            )
            
            # Offload block after use if using block swapping
            if self.blocks_to_swap > 0 and b < self.blocks_to_swap:
                block.to(self.offload_device, non_blocking=self.use_non_blocking)
        
        # Apply output head
        x = self.head(x, e)
        
        # Unpatchify to reconstruct video latents
        output = self.unpatchify(x, grid_sizes)
        
        return (output, None)

    def to(self, device, *args, **kwargs):
        """
        Custom `to` method for controlling device placement.
        
        This allows more fine-grained control over which parts
        of the model are on which devices.
        
        Args:
            device: Target device
            *args, **kwargs: Additional arguments for torch.nn.Module.to()
        
        Returns:
            Self
        """
        # If block swapping is enabled, we only want to move non-swapped parts
        if self.blocks_to_swap > 0:
            # Move non-block parameters
            for name, param in self.named_parameters():
                if 'blocks.' not in name:
                    param.data = param.data.to(device, *args, **kwargs)
            
            # Move non-swapped blocks
            for i, block in enumerate(self.blocks):
                if i >= self.blocks_to_swap:
                    block.to(device, *args, **kwargs)
        else:
            # Standard behavior if block swapping is disabled
            super().to(device, *args, **kwargs)
        
        return self