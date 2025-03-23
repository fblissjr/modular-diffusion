# src/models/text_encoders/__init__.py
from .base import TextEncoder
from .t5 import T5TextEncoder  # explicitly meed to import this to trigger registration

__all__ = ['TextEncoder', 'T5TextEncoder']