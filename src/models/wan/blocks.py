# src/models/wan/blocks.py
"""
Transformer blocks for WanModel implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .attention import WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention

logger = logging.getLogger(__name__)

# Registry of cross-attention types
WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class LayerNorm(nn.LayerNorm):
    """Layer normalization with dtype casting support."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        """Apply layer norm with float32 precision."""
        return super().forward(x.float()).type_as(x)


class WanAttentionBlock(nn.Module):
    """
    Attention block for WanModel.
    
    This combines self-attention, cross-attention, and feed-forward
    network with modulation based on time embeddings.
    """
    
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps
        )
        self.norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
        
        # Cross attention
        if cross_attn_type not in WAN_CROSSATTENTION_CLASSES:
            raise ValueError(f"Unknown cross attention type: {cross_attn_type}")
        
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # Modulation parameters
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        """
        Forward pass through attention block.
        
        Args:
            x (torch.Tensor): Input tensor [B, L, C]
            e (torch.Tensor): Time embeddings [B, 6, C]
            seq_lens (torch.Tensor): Sequence lengths [B]
            grid_sizes (torch.Tensor): Grid sizes [B, 3]
            freqs (torch.Tensor): Frequencies for rotary embeddings
            context (torch.Tensor): Context embeddings [B, L, C]
            context_lens (torch.Tensor): Context lengths [B]
            
        Returns:
            torch.Tensor: Output tensor [B, L, C]
        """
        # Split modulation parameters
        with torch.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)

        # Self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs
        )
        with torch.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # Cross-attention & FFN
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        with torch.autocast(dtype=torch.float32):
            x = x + y * e[5]

        return x