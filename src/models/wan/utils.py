# src/models/wan/utils.py
"""
Utility functions for WanModel implementation.
"""
import torch
import torch.nn as nn
import math
from typing import Tuple
import torch.cuda.amp as amp

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Create sinusoidal embeddings for 1D positions.
    
    Args:
        dim (int): Embedding dimension (must be even)
        position (torch.Tensor): Position tensor
        
    Returns:
        torch.Tensor: Sinusoidal embeddings
    """
    # Preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # Calculate embeddings
    sinusoid = torch.outer(
        position, 
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    """
    Calculate rotary position embedding parameters.
    
    Args:
        max_seq_len (int): Maximum sequence length
        dim (int): Dimension of embedding (must be even)
        theta (int): Base for frequency calculation
        
    Returns:
        torch.Tensor: Complex-valued frequency parameters
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.
    
    Args:
        x (torch.Tensor): Input tensor [B, L, N, C]
        grid_sizes (torch.Tensor): Grid sizes [B, 3]
        freqs (torch.Tensor): Frequencies for embeddings
        
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
    """
    n, c = x.size(2), x.size(3) // 2

    # Split freqs for different dimensions
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # Append to collection
        output.append(x_i)
    return torch.stack(output).float()