# src/models/diffusion/wandit.py
import torch
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import torch.nn as nn

from src.core.registry import register_component
from src.models.diffusion.base import DiffusionModel
from src.models.wan.model import WanModel
from src.core.dtype import DtypeManager

logger = logging.getLogger(__name__)

@register_component("WanDiT", DiffusionModel)
class WanDiT(DiffusionModel):
    """
    WanDiT diffusion transformer model.
    
    This implements the WanVideo DiT architecture for
    text-to-video and image-to-video generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WanDiT model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Get model path and validate
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for WanDiT")
        
        # Load model configuration
        model_config = config.get("model_config")
        if not model_config:
            from src.configs import get_model_config
            model_config = get_model_config(model_path, "wanvideo")
        
        # Set model parameters from config
        self.in_channels = config.get("in_channels", 16)
        self.out_channels = config.get("out_channels", 16)
        self.model_type = config.get("model_type", "t2v")
        
        # Create WanModel with parameters from model_config
        self.model = WanModel(
            model_type=self.model_type,
            patch_size=model_config.patch_size,
            in_dim=self.in_channels,
            out_dim=self.out_channels,
            dim=model_config.dim,
            ffn_dim=model_config.ffn_dim,
            freq_dim=getattr(model_config, "freq_dim", 256),
            text_dim=getattr(model_config, "text_dim", 4096),
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            window_size=model_config.window_size,
            qk_norm=model_config.qk_norm,
            cross_attn_norm=model_config.cross_attn_norm,
        )
        
        # Load weights
        self._load_weights(model_path)
        
        # Set to eval mode
        self.model.eval()
        
        logger.info(f"Initialized WanDiT with in_channels={self.in_channels}, out_channels={self.out_channels}")
    
    def _load_weights(self, model_path):
        """load model weights and ensure they're on the right device."""
        import os
        from pathlib import Path
        
        model_path = Path(model_path)

        # find checkpoint file
        if model_path.is_dir():
            # look for safetensors first
            safetensor_files = list(model_path.glob("*.safetensors"))
            if safetensor_files:
                checkpoint_path = safetensor_files[0]
            else:
                # fall back to .pth files
                pth_files = list(model_path.glob("*.pth"))
                if not pth_files:
                    raise FileNotFoundError(
                        f"no model checkpoint files found in {model_path}"
                    )
                checkpoint_path = pth_files[0]
        else:
            checkpoint_path = model_path

        # load weights
        logger.info(f"loading model weights from {checkpoint_path}")
        try:
            if str(checkpoint_path).endswith('.safetensors'):
                import safetensors.torch
                state_dict = safetensors.torch.load_file(checkpoint_path, device='cpu')
            else:
                state_dict = torch.load(checkpoint_path, map_location='cpu')

            # handle 'model.' prefix if present
            adjusted_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    adjusted_state_dict[k[6:]] = v
                else:
                    adjusted_state_dict[k] = v

            # load weights with non-strict matching
            missing, unexpected = self.model.load_state_dict(adjusted_state_dict, strict=False)

            # debug weight mismatch - log some examples to understand pattern
            if len(missing) > 0 and len(unexpected) > 0:
                # group keys by prefix pattern
                missing_prefixes = {}
                for k in list(missing)[:5]:
                    prefix = k.split(".")[0]
                    missing_prefixes[prefix] = missing_prefixes.get(prefix, 0) + 1

                unexpected_prefixes = {}
                for k in list(unexpected)[:5]:
                    prefix = k.split(".")[0]
                    unexpected_prefixes[prefix] = unexpected_prefixes.get(prefix, 0) + 1

                logger.warning(f"missing key examples: {list(missing)[:3]}")
                logger.warning(f"unexpected key examples: {list(unexpected)[:3]}")
                logger.warning(f"missing prefixes: {missing_prefixes}")
                logger.warning(f"unexpected prefixes: {unexpected_prefixes}")

            # debug weight mismatch
            if len(missing) > 10 and len(unexpected) > 10:
                # sample a few keys to spot patterns
                for i in range(3):
                    m_key = list(missing)[i]
                    u_key = list(unexpected)[i]
                    logger.info(f"missing vs unexpected: {m_key} vs {u_key}")

            if len(missing) > 100 and len(unexpected) > 100:
                logger.info("attempting to fix key mismatches with remapping...")

                # create new state dict with remapped keys
                remapped_dict = {}

                # map 'diffusion_model.X' keys to our format
                for k, v in state_dict.items():
                    # remove diffusion_model prefix
                    if k.startswith("diffusion_model."):
                        new_key = k.replace("diffusion_model.", "")

                        # map block structure
                        if "blocks." in new_key:
                            parts = new_key.split(".")
                            if len(parts) > 3:
                                # try different mappings
                                block_num = parts[1]
                                component = parts[2]
                                rest = ".".join(parts[3:])
                                new_key = f"blocks.{block_num}.{component}.{rest}"

                        remapped_dict[new_key] = v

                    # try loading with remapped dict
                    missing2, unexpected2 = self.model.load_state_dict(
                        remapped_dict, strict=False
                    )

                    if len(missing2) < len(missing):
                        logger.info(
                            f"remapping reduced missing keys from {len(missing)} to {len(missing2)}"
                        )

            # move model to the correct device after loading weights
            self.model = self.model.to(self.device)

            if missing:
                logger.warning(
                    f"missing keys when loading weights: {len(missing)} keys"
                )
            if unexpected:
                logger.warning(
                    f"unexpected keys when loading weights: {len(unexpected)} keys"
                )

        except Exception as e:
            logger.error(f"failed to load weights: {e}")
            raise

    # add explicit dtype conversion
    def forward(self, latents, timestep, text_embeds, **kwargs):
        # ensure timestep is the right dtype before processing
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(device=self.device, dtype=self.dtype)

        # get sequence length from shape or kwargs
        seq_len = kwargs.get("seq_len")
        if not seq_len:
            latent = latents[0]
            seq_len = (
                latent.shape[1] * latent.shape[2] * latent.shape[3]
                if len(latent.shape) == 4
                else latent.shape[2] * latent.shape[3] * latent.shape[4]
            )

        # run forward with correct dtypes
        with torch.no_grad():
            outputs = self.model(
                latents,
                timestep,  # now in correct dtype
                text_embeds,
                seq_len=seq_len,
                clip_fea=kwargs.get("clip_fea"),
                y=kwargs.get("y"),
            )
        
        return outputs, None

        def to(self, device=None, dtype=None):
            """make sure model components move to the right device and dtype"""
            super().to(device, dtype)

            # update internal model if device/dtype changes
            if device is not None:
                self.model = self.model.to(device)
            if dtype is not None:
                self.model = self.model.to(dtype)

            return self
    
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
        #with torch.cuda.amp.autocast(dtype=torch.float32):
        with DtypeManager.autocast():
            x = x + y * e[2]
        
        # Cross-attention
        y = self.cross_attn(self.norm2(x), context, context_lens)
        #with torch.cuda.amp.autocast(dtype=torch.float32):
        with DtypeManager.autocast():
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