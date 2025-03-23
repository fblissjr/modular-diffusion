# src/models/diffusion/__init__.py
from .base import DiffusionModel
from .wandit import WanDiT  # explicitly meed to import this to trigger registration

__all__ = ['DiffusionModel', 'WanDiT']