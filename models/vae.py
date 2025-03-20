"""
WanVideo VAE (Variational Autoencoder) for video processing.

This module implements the VAE used in WanVideo for encoding videos to latent 
space and decoding latents back to pixel space. Key features include:
- 3D convolutional architecture for temporal processing
- Memory-efficient tiling for working with high-resolution videos
- Support for different precision formats

The VAE downsamples videos spatially by a factor of 8 and temporally
by a factor of 4, significantly reducing the computational load for
the diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog
from typing import List, Dict, Optional, Tuple, Union, Any
from einops import rearrange, repeat
import math
from tqdm import tqdm

logger = structlog.get_logger()

class WanRMSNorm(nn.Module):
    """RMS normalization layer."""
    
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class CausalConv3d(nn.Conv3d):
    """
    3D convolution with causal padding in the temporal dimension.
    
    This ensures that each frame only depends on itself and previous frames,
    not future frames, preserving temporal causality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store original padding
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        # Use no padding in forward, we'll handle it manually
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        """
        Apply causal 3D convolution.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
            cache_x: Optional cached state from previous time steps
            
        Returns:
            Convolved tensor
        """
        padding = list(self._padding)
        
        # Use cached state if available for causal convolution
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            
        # Apply padding manually
        x = F.pad(x, padding)
        
        # Apply standard convolution
        return super().forward(x)


class Upsample(nn.Upsample):
    """
    Custom upsampling layer with improved precision handling.
    
    This wrapper ensures proper handling of different precision formats
    during upsampling operations.
    """
    
    def forward(self, x):
        """
        Upsample with precision handling.
        
        Args:
            x: Input tensor
            
        Returns:
            Upsampled tensor
        """
        # Upsampling works better in float32, then convert back
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """
    Resample layer for changing spatial and temporal resolution.
    
    This handles both upsampling and downsampling in both spatial
    and temporal dimensions.
    """
    
    def __init__(self, dim, mode):
        """
        Initialize resample layer.
        
        Args:
            dim: Channel dimension
            mode: Resampling mode ('upsample2d', 'upsample3d', 
                  'downsample2d', 'downsample3d', or 'none')
        """
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode
        
        # Define different resampling operations based on mode
        if mode == 'upsample2d':
            # Spatial upsampling only (2x)
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            # Spatial upsampling (2x)
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            # Temporal upsampling (2x)
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            # Spatial downsampling only (2x)
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # Padding for stride 2
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            # Spatial downsampling (2x)
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),  # Padding for stride 2
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            # Temporal downsampling (2x)
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), 
                                          stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            # No resampling
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        Apply resampling operation.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
            feat_cache: Optional feature cache for temporal processing
            feat_idx: Feature index tracker
            
        Returns:
            Resampled tensor
        """
        b, c, t, h, w = x.size()
        
        # Handle temporal upsampling
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                
                # First time through this layer
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    # Cache the recent frames for temporal continuity
                    cache_x = x[:, :, -2:, :, :].clone()
                    
                    # Handle boundary cases
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        # Add last frame from previous chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x
                        ], dim=2)
                    
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        # No previous frame, pad with zeros
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ], dim=2)
                    
                    # Apply temporal convolution
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    
                    # Update cache
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    
                    # Reshape to double temporal resolution
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        
        # Apply spatial resampling
        t = x.shape[2]  # Current temporal dimension
        # Reshape to process each frame independently
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        # Reshape back to video format
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        
        # Handle temporal downsampling
        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                
                # First time through this layer
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    # Cache the last frame for temporal continuity
                    cache_x = x[:, :, -1:, :, :].clone()
                    
                    # Apply temporal convolution with cached frame
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    
                    # Update cache
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for VAE.
    
    This combines normalization, activation, and convolution
    with a residual connection for better gradient flow.
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Main branch
        self.residual = nn.Sequential(
            WanRMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            WanRMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1)
        )
        
        # Skip connection with optional projection
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        Apply residual block.
        
        Args:
            x: Input tensor
            feat_cache: Optional feature cache for temporal processing
            feat_idx: Feature index tracker
            
        Returns:
            Output tensor with residual connection
        """
        # Apply shortcut
        h = self.shortcut(x)
        
        # Apply main branch
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                # Handle causal convolution with caching
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # Add last frame from previous chunk for continuity
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x
                    ], dim=2)
                
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                # Apply other layers normally
                x = layer(x)
        
        # Add residual connection
        return x + h


class AttentionBlock(nn.Module):
    """
    Self-attention block for VAE.
    
    This adds a spatial self-attention mechanism to capture
    long-range dependencies within each frame.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Normalization and projections
        self.norm = WanRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        # Initialize to zero for stable training
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        """
        Apply attention block.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
            
        Returns:
            Attended tensor
        """
        # Store original for residual
        identity = x
        b, c, t, h, w = x.size()
        
        # Process each frame independently
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        
        # Compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        
        # Apply scaled dot-product attention
        x = F.scaled_dot_product_attention(q, k, v)
        
        # Project back to original dimension
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        
        # Reshape back to video format
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        
        # Add residual connection
        return x + identity


class Encoder3d(nn.Module):
    """
    3D encoder for the VAE.
    
    This encodes video frames into latent representations
    using a hierarchical architecture with residual blocks,
    attention, and downsampling.
    """
    
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0
    ):
        """
        Initialize 3D encoder.
        
        Args:
            dim: Base dimension
            z_dim: Output latent dimension
            dim_mult: Dimension multipliers for each level
            num_res_blocks: Number of residual blocks per level
            attn_scales: Scales at which to apply attention
            temperal_downsample: Whether to downsample temporally at each level
            dropout: Dropout probability
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        
        # Calculate dimensions at each level
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0  # Track scale for attention
        
        # Initial convolution
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        
        # Downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Residual blocks at current resolution
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                
                # Add attention if needed
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                
                in_dim = out_dim
            
            # Downsample if not at final level
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        
        self.downsamples = nn.Sequential(*downsamples)
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout)
        )
        
        # Output convolution
        self.head = nn.Sequential(
            WanRMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        Encode video to latent space.
        
        Args:
            x: Input video tensor of shape [B, C, T, H, W]
            feat_cache: Optional feature cache for temporal processing
            feat_idx: Feature index tracker
            
        Returns:
            Latent representation
        """
        # Apply initial convolution
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()
            
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # Add last frame from previous chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x
                ], dim=2)
            
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        
        # Apply downsample blocks
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        # Apply middle blocks
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        # Apply output head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # Add last frame from previous chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x
                    ], dim=2)
                
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        
        return x


class Decoder3d(nn.Module):
    """
    3D decoder for the VAE.
    
    This decodes latent representations back to video frames
    using a hierarchical architecture with residual blocks,
    attention, and upsampling.
    """
    
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0
    ):
        """
        Initialize 3D decoder.
        
        Args:
            dim: Base dimension
            z_dim: Input latent dimension
            dim_mult: Dimension multipliers for each level
            num_res_blocks: Number of residual blocks per level
            attn_scales: Scales at which to apply attention
            temperal_upsample: Whether to upsample temporally at each level
            dropout: Dropout probability
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        
        # Calculate dimensions at each level (reversed from encoder)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)  # Track scale for attention
        
        # Initial convolution
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout)
        )
        
        # Upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Adjust input dim for certain levels due to skip connections
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            
            # Residual blocks at current resolution
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                
                # Add attention if needed
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                
                in_dim = out_dim
            
            # Upsample if not at final level
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        
        self.upsamples = nn.Sequential(*upsamples)
        
        # Output convolution
        self.head = nn.Sequential(
            WanRMSNorm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        Decode latent representation to video.
        
        Args:
            x: Input latent tensor of shape [B, C, T, H, W]
            feat_cache: Optional feature cache for temporal processing
            feat_idx: Feature index tracker
            
        Returns:
            Reconstructed video
        """
        # Apply initial convolution
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()
            
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # Add last frame from previous chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x
                ], dim=2)
            
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        
        # Apply middle blocks
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        # Apply upsample blocks
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        # Apply output head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # Add last frame from previous chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x
                    ], dim=2)
                
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        
        return x


def count_conv3d(model):
    """Count the number of CausalConv3d modules in a model."""
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVideoVAE(nn.Module):
    """
    Video VAE for WanVideo.
    
    This combines the encoder and decoder with normalization
    and other utilities to process videos efficiently.
    """
    
    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        dtype=torch.float32
    ):
        """
        Initialize Video VAE.
        
        Args:
            dim: Base dimension
            z_dim: Latent dimension
            dim_mult: Dimension multipliers for each level
            num_res_blocks: Number of residual blocks per level
            attn_scales: Scales at which to apply attention
            temperal_downsample: Whether to downsample temporally at each level
            dropout: Dropout probability
            dtype: Data type for computation
        """
        super().__init__()
        self.logger = logger.bind(component="WanVideoVAE")
        
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]  # Reverse for decoder
        self.dtype = dtype
        self.upsampling_factor = 8  # Spatial upsampling factor
        
        # Define mean and std for latent normalization
        # These are empirically determined for stable training
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]
        
        # Initialize encoder and decoder
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks,
            attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim, z_dim, dim_mult, num_res_blocks,
            attn_scales, self.temperal_upsample, dropout
        )
        
        self.logger.info("Initialized WanVideoVAE", 
                      z_dim=z_dim, 
                      spatial_upsampling=self.upsampling_factor,
                      temporal_upsampling=4)
    
    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        """
        Build a 1D blending mask for tiled processing.
        
        Args:
            length: Length of the mask
            left_bound: Whether this is a left boundary
            right_bound: Whether this is a right boundary
            border_width: Width of the blending border
            
        Returns:
            1D blending mask
        """
        x = torch.ones((length,))
        
        # Apply ramp on left edge if not at boundary
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        
        # Apply ramp on right edge if not at boundary
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        
        return x
    
    def build_mask(self, data, is_bound, border_width):
        """
        Build a 3D blending mask for tiled processing.
        
        Args:
            data: Data tensor to match shape
            is_bound: Tuple of (left, right, top, bottom) boundary flags
            border_width: Tuple of (height, width) border widths
            
        Returns:
            Blending mask for smooth tiling
        """
        _, _, _, H, W = data.shape
        
        # Create 1D masks for height and width
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])
        
        # Expand to 2D
        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)
        
        # Combine by taking minimum
        mask = torch.stack([h, w]).min(dim=0).values
        
        # Add batch, channel and time dimensions
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        
        return mask
    
    def tiled_encode(self, video, device, tile_size, tile_stride):
        """
        Encode video using tiling for memory efficiency.
        
        This processes the video in spatial tiles to reduce peak memory usage.
        
        Args:
            video: Video tensor of shape [B, C, T, H, W]
            device: Device for computation
            tile_size: Tuple of (height, width) tile sizes
            tile_stride: Tuple of (height, width) tile strides
            
        Returns:
            Encoded latent representation
        """
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        
        # Split into tasks (one per tile)
        tasks = []
        for h in range(0, H, stride_h):
            # Skip if this would duplicate coverage
            if (h-stride_h >= 0 and h-stride_h+size_h >= H):
                continue
            
            for w in range(0, W, stride_w):
                # Skip if this would duplicate coverage
                if (w-stride_w >= 0 and w-stride_w+size_w >= W):
                    continue
                
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))
        
        # Data storage device (CPU) and computation device
        data_device = "cpu"
        computation_device = device
        
        # Calculate output dimensions
        out_T = (T + 3) // 4  # Temporal downsampling
        weight = torch.zeros((1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor), 
                            dtype=video.dtype, device=data_device)
        values = torch.zeros((1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor), 
                           dtype=video.dtype, device=data_device)
        
        # Process each tile
        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            # Extract and process tile
            tile = video[:, :, :, h:h_, w:w_].to(computation_device)
            encoded_tile = self.encode(tile, self.scale).to(data_device)
            
            # Create blending mask
            mask = self.build_mask(
                encoded_tile,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) // self.upsampling_factor, 
                              (size_w - stride_w) // self.upsampling_factor)
            ).to(dtype=video.dtype, device=data_device)
            
            # Calculate target position in output
            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            
            # Add weighted tile to output
            values[
                :, :, :,
                target_h:target_h + encoded_tile.shape[3],
                target_w:target_w + encoded_tile.shape[4],
            ] += encoded_tile * mask
            
            weight[
                :, :, :,
                target_h:target_h + encoded_tile.shape[3],
                target_w:target_w + encoded_tile.shape[4],
            ] += mask
        
        # Normalize by weights
        values = values / weight
        values = values.float()
        
        return values
    
    def tiled_decode(self, hidden_states, device, tile_size, tile_stride):
        """
        Decode latents using tiling for memory efficiency.
        
        This processes the latents in spatial tiles to reduce peak memory usage.
        
        Args:
            hidden_states: Latent tensor of shape [B, C, T, H, W]
            device: Device for computation
            tile_size: Tuple of (height, width) tile sizes
            tile_stride: Tuple of (height, width) tile strides
            
        Returns:
            Decoded video
        """
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        
        # Split into tasks (one per tile)
        tasks = []
        for h in range(0, H, stride_h):
            # Skip if this would duplicate coverage
            if (h-stride_h >= 0 and h-stride_h+size_h >= H):
                continue
            
            for w in range(0, W, stride_w):
                # Skip if this would duplicate coverage
                if (w-stride_w >= 0 and w-stride_w+size_w >= W):
                    continue
                
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))
        
        # Data storage device (CPU) and computation device
        data_device = "cpu"
        computation_device = device
        
        # Calculate output dimensions
        out_T = T * 4 - 3  # Temporal upsampling
        weight = torch.zeros((1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor), 
                            dtype=hidden_states.dtype, device=data_device)
        values = torch.zeros((1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor), 
                           dtype=hidden_states.dtype, device=data_device)
        
        # Process each tile
        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            # Extract and process tile
            tile = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            decoded_tile = self.decode(tile, self.scale).to(data_device)
            
            # Create blending mask
            mask = self.build_mask(
                decoded_tile,
                is_bound=(h==0, h_>=H, w==0, w_>=W),
                border_width=((size_h - stride_h) * self.upsampling_factor, 
                              (size_w - stride_w) * self.upsampling_factor)
            ).to(dtype=hidden_states.dtype, device=data_device)
            
            # Calculate target position in output
            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            
            # Add weighted tile to output
            values[
                :, :, :,
                target_h:target_h + decoded_tile.shape[3],
                target_w:target_w + decoded_tile.shape[4],
            ] += decoded_tile * mask
            
            weight[
                :, :, :,
                target_h:target_h + decoded_tile.shape[3],
                target_w:target_w + decoded_tile.shape[4],
            ] += mask
        
        # Normalize by weights
        values = values / weight
        
        # Clamp to valid pixel range
        values = values.float().clamp_(-1, 1)
        
        return values
    
    def encode(self, videos, scale=None):
        """
        Encode videos to latent space.
        
        Args:
            videos: Video tensor or list of tensors
            scale: Optional scale parameters
            
        Returns:
            Latent representation
        """
        scale = scale or self.scale
        
        # Clear cache for causal convolution
        self.clear_cache()
        
        # Process the video through encoder
        t = videos.shape[2]
        iter_ = 1 + (t - 1) // 4  # Process in chunks of 4 frames
        
        for i in range(iter_):
            self._enc_conv_idx = [0]
            
            if i == 0:
                # Process first frame(s)
                out = self.encoder(
                    videos[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx
                )
            else:
                # Process subsequent chunks
                chunk = videos[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :]
                out_ = self.encoder(
                    chunk,
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx
                )
                out = torch.cat([out, out_], 2)
        
        # Split into mean and log_var
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        
        # Apply normalization
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale[0].to(dtype=mu.dtype, device=mu.device) if isinstance(scale[0], torch.Tensor) else scale[0]
            scale_1 = scale[1].to(dtype=mu.dtype, device=mu.device) if isinstance(scale[1], torch.Tensor) else scale[1]
            mu = (mu - scale) * scale_1
        
        return mu
    
    def decode(self, z, scale=None):
        """
        Decode latents to video.
        
        Args:
            z: Latent tensor
            scale: Optional scale parameters
            
        Returns:
            Reconstructed video
        """
        scale = scale or self.scale
        
        # Clear cache for causal convolution
        self.clear_cache()
        
        # Denormalize latents
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            scale_0 = scale[0].to(dtype=z.dtype, device=z.device) if isinstance(scale[0], torch.Tensor) else scale[0]
            scale_1 = scale[1].to(dtype=z.dtype, device=z.device) if isinstance(scale[1], torch.Tensor) else scale[1]
            z = z / scale_1 + scale_0
        
        # Process through decoder
        iter_ = z.shape[2]  # Number of latent frames
        x = self.conv2(z)
        
        for i in range(iter_):
            self._conv_idx = [0]
            
            if i == 0:
                # Process first frame
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx
                )
            else:
                # Process subsequent frames
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx
                )
                out = torch.cat([out, out_], 2)
        
        return out
    
    def clear_cache(self):
        """Clear caches used for temporal processing."""
        # Count the number of causal convolutions
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        
        # Cache for encoder
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
    
    def forward(self, x):
        """
        Forward pass (not typically used for inference).
        
        Args:
            x: Input video tensor
            
        Returns:
            Reconstructed video and latent variables
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean tensor
            log_var: Log variance tensor
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu