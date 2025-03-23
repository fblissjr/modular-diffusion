# src/models/wan/model.py
"""
WanModel implementation based on WanVideo architecture.

This is our controlled version of the WanModel architecture,
allowing us to modify and improve it as needed.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional, Tuple

from .blocks import WanAttentionBlock
from .utils import sinusoidal_embedding_1d, rope_params

logger = logging.getLogger(__name__)

class WanModel(nn.Module):
    """
    WanModel implementation based on WanVideo architecture.
    
    This model serves as the backbone for the diffusion transformer
    used in WanVideo for text-to-video and image-to-video generation.
    """
    
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6
    ):
        """
        Initialize the diffusion model backbone.

        Args:
            model_type (str): Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (tuple): 3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (int): Fixed length for text embeddings
            in_dim (int): Input video channels (C_in)
            dim (int): Hidden dimension of the transformer
            ffn_dim (int): Intermediate dimension in feed-forward network
            freq_dim (int): Dimension for sinusoidal time embeddings
            text_dim (int): Input dimension for text embeddings
            out_dim (int): Output video channels (C_out)
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer blocks
            window_size (tuple): Window size for local attention (-1 indicates global attention)
            qk_norm (bool): Enable query/key normalization
            cross_attn_norm (bool): Enable cross-attention normalization
            eps (float): Epsilon value for normalization layers
        """
        super().__init__()

        # Store configuration
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)
        )

        # Blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type=cross_attn_type,
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                window_size=window_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps
            )
            for _ in range(num_layers)
        ])

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Initialize frequency parameters for rotary embeddings
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # Image embedding for I2V mode
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # Initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        """
        Forward pass through the diffusion model

        Args:
            x (List[torch.Tensor]): List of input video tensors, each with shape [C_in, F, H, W]
            t (torch.Tensor): Diffusion timesteps tensor
            context (List[torch.Tensor]): List of text embeddings each with shape [L, C]
            seq_len (int): Maximum sequence length for positional encoding
            clip_fea (torch.Tensor, optional): CLIP image features for image-to-video mode
            y (List[torch.Tensor], optional): Conditional video inputs for image-to-video mode

        Returns:
            List[torch.Tensor]: List of denoised video tensors with original input shapes
        """
        # Move freqs to correct device if needed
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Concatenate conditional inputs if provided
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
        ])

        # Time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # Context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        # Add CLIP features for image-to-video mode
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Process through transformer blocks
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
        )

        for block in self.blocks:
            x = block(x, **kwargs)

        # Head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (torch.Tensor): Patchified features [B, L, C_out * prod(patch_size)]
            grid_sizes (torch.Tensor): Original spatial-temporal grid dimensions

        Returns:
            List[torch.Tensor]: Reconstructed video tensors
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        """Initialize model parameters using Xavier initialization."""
        # Basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # Init output layer
        nn.init.zeros_(self.head.head.weight)


class Head(nn.Module):
    """Output head for the diffusion model."""
    
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # Layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # Modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        """
        Apply output head to transform features to patches.
        
        Args:
            x (torch.Tensor): Input features [B, L, C]
            e (torch.Tensor): Conditioning embeddings
            
        Returns:
            torch.Tensor: Output patches
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class LayerNorm(nn.LayerNorm):
    """Layer normalization with dtype casting support."""

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class MLPProj(nn.Module):
    """Projection MLP for image features."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)