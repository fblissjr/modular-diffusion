# src/models/text_encoders/t5.py
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from transformers import T5EncoderModel, AutoTokenizer

from src.core.component import Component
from src.models.text_encoders.base import TextEncoder

logger = logging.getLogger(__name__)

class T5TextEncoder(TextEncoder):
    """
    T5-based text encoder for WanVideo.
    
    This loads a pre-trained T5 encoder model and provides methods
    to encode text prompts into embeddings, similar to how LLM
    tokenizers and embedding modules work together.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize T5 text encoder.
        
        Args:
            config: Text encoder configuration
        """
        super().__init__(config)
        
        # Get paths and parameters
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for T5TextEncoder")
            
        tokenizer_name = config.get("tokenizer", "google/umt5-xxl")
        max_length = config.get("max_length", 512)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model
        logger.info(f"Loading T5 encoder from {model_path}")
        self.t5_model = T5EncoderModel.from_pretrained(model_path)
        
        # Move to device and set dtype
        self.t5_model.to(self.device, self.dtype)
        
        # Make sure model is in eval mode
        self.t5_model.eval()
        
        # Store config
        self.max_length = max_length
        
        logger.info(f"Initialized T5TextEncoder with tokenizer={tokenizer_name}, max_length={max_length}")
        
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
        # Convert to list if single string
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Process negative prompts if provided
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
                
            # Make sure lengths match
            if len(negative_prompt) == 1 and len(prompt) > 1:
                negative_prompt = negative_prompt * len(prompt)
                
            # Encode negative prompts
            neg_embeddings = self._encode_text(negative_prompt)
            
            # Return both prompt and negative prompt embeddings
            return {
                "prompt_embeds": self._encode_text(prompt),
                "negative_prompt_embeds": neg_embeddings
            }
        else:
            # Return only prompt embeddings
            return {
                "prompt_embeds": self._encode_text(prompt)
            }
    
    def _encode_text(self, text_list: List[str]) -> List[torch.Tensor]:
        """
        Encode list of text prompts.
        
        Args:
            text_list: List of text prompts
            
        Returns:
            List of embedding tensors
        """
        embeddings = []
        
        for text in text_list:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                output = self.t5_model(**inputs)
                
            # Get last hidden states
            text_embeds = output.last_hidden_state
            
            # Store embedding
            embeddings.append(text_embeds)
            
        return embeddings
        
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> "T5TextEncoder":
        """
        Move encoder to specified device and dtype.
        
        Args:
            device: Target device
            dtype: Target dtype
            
        Returns:
            Self for chaining
        """
        super().to(device, dtype)
        
        if hasattr(self, "t5_model"):
            if device is not None:
                self.t5_model = self.t5_model.to(device)
            if dtype is not None:
                self.t5_model = self.t5_model.to(dtype)
                
        return self