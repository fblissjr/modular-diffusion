# In src/models/text_encoders/t5.py
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from transformers import T5EncoderModel, AutoTokenizer, AutoConfig
import safetensors.torch

from src.core.component import Component
from src.core.registry import register_component
from src.models.text_encoders.base import TextEncoder

logger = logging.getLogger(__name__)

@register_component("T5TextEncoder", TextEncoder)
class T5TextEncoder(TextEncoder):
    """
    T5-based text encoder for WanVideo.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize T5 text encoder."""
        super().__init__(config)

        if 'test_mode' in config and config['test_mode']:
            # Create dummy tokenizer and model for testing
            logger.warning("Running T5TextEncoder in test mode with dummy model")
            self.tokenizer = None
            self.t5_model = nn.Module()  # Empty module
            self.max_length = max_length
            return
        
        # Get paths and parameters
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for T5TextEncoder")
            
        tokenizer_name = config.get("tokenizer", "google/umt5-xxl")
        max_length = config.get("max_length", 512)
        
        # Load tokenizer directly from HuggingFace
        logger.info(f"Loading tokenizer from {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model - special handling for safetensors
        logger.info(f"Loading T5 encoder from {model_path}")
        model_path = Path(model_path)
        
        if model_path.is_file() and str(model_path).endswith('.safetensors'):
            # Create a model config for UMT5-XXL
            logger.info("Creating model from safetensors file")
            model_config = AutoConfig.from_pretrained(tokenizer_name)
            
            # Create empty model with the right config
            self.t5_model = T5EncoderModel(config=model_config)
            
            # Load weights from safetensors
            logger.info(f"Loading weights from {model_path}")
            state_dict = safetensors.torch.load_file(model_path)
            
            # Load state dict
            self.t5_model.load_state_dict(state_dict, strict=False)
        else:
            # Standard loading
            self.t5_model = T5EncoderModel.from_pretrained(model_path)
        
        # Move to device and set dtype
        self.t5_model.to(self.device, self.dtype)
        
        # Make sure model is in eval mode
        self.t5_model.eval()
        
        # Store config
        self.max_length = max_length
        
        logger.info(f"Initialized T5TextEncoder with max_length={max_length}")
        
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