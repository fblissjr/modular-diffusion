# src/models/wan/attention.py
"""
Attention mechanisms for WanModel implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Optional

from .utils import rope_apply

logger = logging.getLogger(__name__)


class WanRMSNorm(nn.Module):
    """RMS normalization layer."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Apply RMS normalization."""
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanSelfAttention(nn.Module):
    """Self-attention with rotary position embeddings."""

    def __init__(
        self, 
        dim, 
        num_heads, 
        window_size=(-1, -1), 
        qk_norm=True, 
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        """
        Forward pass with rotary position embedding.
        
        Args:
            x (torch.Tensor): Input tensor [B, L, C]
            seq_lens (torch.Tensor): Sequence lengths [B]
            grid_sizes (torch.Tensor): Grid sizes [B, 3]
            freqs (torch.Tensor): Frequencies for rotary embeddings
            
        Returns:
            torch.Tensor: Self-attention output
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # Query, key, value projection
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply rotary embeddings
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # Compute attention with window size constraint
        # Note: We're using a simplified version that always uses global attention
        # If window_size needs to be implemented, additional code would be needed
        attn_bias = None
        if self.window_size[0] > 0 or self.window_size[1] > 0:
            logger.warning("Window size constraints not implemented - using global attention")

        # Flash attention or scaled dot-product attention
        try:
            # Try to use torch's built-in efficient attention
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2), 
                k.transpose(1, 2), 
                v.transpose(1, 2),
                attn_mask=attn_bias
            ).transpose(1, 2)
        except:
            # Fallback to manual implementation
            q = q.transpose(1, 2)  # [B, N, L, D]
            k = k.transpose(1, 2)  # [B, N, L, D]
            v = v.transpose(1, 2)  # [B, N, L, D]
            
            # Compute attention scores
            scale = 1.0 / math.sqrt(d)
            attn = torch.matmul(q, k.transpose(-1, -2)) * scale
            
            # Apply softmax
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            x = torch.matmul(attn, v)  # [B, N, L, D]
            x = x.transpose(1, 2)  # [B, L, N, D]

        # Output projection
        x = x.reshape(b, s, n * d)
        x = self.o(x)
        return x


class WanT2VCrossAttention(nn.Module):
    """Cross-attention for text-to-video conditioning."""
    
    def __init__(
        self, 
        dim, 
        num_heads, 
        window_size=(-1, -1), 
        qk_norm=True, 
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens=None):
        """
        Cross-attention forward pass.
        
        Args:
            x (torch.Tensor): Query tensor [B, L1, C]
            context (torch.Tensor): Key/value context tensor [B, L2, C]
            context_lens (torch.Tensor): Context lengths
            
        Returns:
            torch.Tensor: Cross-attention output
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # Flash attention or scaled dot-product attention
        try:
            # Try to use torch's built-in efficient attention
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2), 
                k.transpose(1, 2), 
                v.transpose(1, 2)
            ).transpose(1, 2)
        except:
            # Fallback to manual implementation
            q = q.transpose(1, 2)  # [B, N, L1, D]
            k = k.transpose(1, 2)  # [B, N, L2, D]
            v = v.transpose(1, 2)  # [B, N, L2, D]
            
            # Compute attention scores
            scale = 1.0 / math.sqrt(d)
            attn = torch.matmul(q, k.transpose(-1, -2)) * scale
            
            # Apply softmax
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            x = torch.matmul(attn, v)  # [B, N, L1, D]
            x = x.transpose(1, 2)  # [B, L1, N, D]

        # Output projection
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanT2VCrossAttention):
    """Cross-attention for image-to-video conditioning."""
    
    def __init__(
        self, 
        dim, 
        num_heads, 
        window_size=(-1, -1), 
        qk_norm=True, 
        eps=1e-6
    ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        
        # Additional projections for image features
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens=None):
        """
        Cross-attention with image and text conditioning.
        
        Args:
            x (torch.Tensor): Query tensor [B, L1, C]
            context (torch.Tensor): Combined image+text context [B, L2, C]
            context_lens (torch.Tensor): Context lengths
            
        Returns:
            torch.Tensor: Cross-attention output
        """
        # Split image and text context
        # Image context is the first 257 tokens (CLIP image embeddings)
        context_img = context[:, :257]
        context_txt = context[:, 257:]
        
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Compute query, key, value for text
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k_txt = self.norm_k(self.k(context_txt)).view(b, -1, n, d)
        v_txt = self.v(context_txt).view(b, -1, n, d)
        
        # Compute key, value for image
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        # Process image context
        img_out = self._process_attention(q, k_img, v_img)
        
        # Process text context
        txt_out = self._process_attention(q, k_txt, v_txt)
        
        # Combine outputs
        x = img_out + txt_out
        
        # Final projection
        x = self.o(x)
        return x
        
    def _process_attention(self, q, k, v):
        """Helper method to compute attention."""
        b, n, d = q.size(0), self.num_heads, self.head_dim
        
        try:
            # Try to use torch's built-in efficient attention
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2), 
                k.transpose(1, 2), 
                v.transpose(1, 2)
            ).transpose(1, 2)
        except:
            # Fallback to manual implementation
            q_t = q.transpose(1, 2)  # [B, N, L1, D]
            k_t = k.transpose(1, 2)  # [B, N, L2, D]
            v_t = v.transpose(1, 2)  # [B, N, L2, D]
            
            # Compute attention scores
            scale = 1.0 / math.sqrt(d)
            attn = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale
            
            # Apply softmax
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention to values
            x = torch.matmul(attn, v_t)  # [B, N, L1, D]
            x = x.transpose(1, 2)  # [B, L1, N, D]

        # Reshape to original dimensions
        x = x.reshape(b, -1, n * d)
        return x