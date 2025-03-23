# src/models/vae/__init__.py
from .base import VAE
from .wanvae import WanVAEAdapter  # explicitly meed to import this to trigger registration

__all__ = ['VAE', 'WanVAEAdapter']