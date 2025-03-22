# src/models/text_encoders/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import torch
from src.core.component import Component

class TextEncoder(Component, ABC):
    """
    Base interface for text encoders.
    
    This provides a common interface for text encoding components,
    similar to how LLM tokenizers and embedding models work.
    """
    
    @abstractmethod
    def encode(
        self, 
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Encode text prompt(s) into embeddings.
        
        Args:
            prompt: Text prompt(s) to encode
            negative_prompt: Optional negative prompt(s)
            
        Returns:
            Dictionary with prompt_embeds and negative_prompt_embeds
        """
        pass