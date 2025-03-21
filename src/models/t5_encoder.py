"""
T5 Text Encoder for handling text conditioning in diffusion models.

This module provides a high-level interface for encoding text prompts
using T5 models (specifically UMT5-XXL for WanVideo). It supports:
- Running on CPU or GPU
- Different precision formats
- Potential for integration with other inference engines

The text encoder converts text prompts into embeddings that guide
the diffusion process during image generation.
"""

import os
import torch
import structlog
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer, T5EncoderModel
from safetensors.torch import load_file

logger = structlog.get_logger()

class T5TextEncoder:
    """
    Text encoder using T5 (specifically UMT5) for diffusion model conditioning.
    
    WanVideo uses a T5 encoder-only model to convert text into embeddings that
    guide the diffusion process. These embeddings are used during cross-attention
    in the DiT model to condition the generation on the text prompt.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_cpu: bool = False,
        quantization: str = "disabled",
    ):
        """
        Initialize T5 text encoder.
        
        Args:
            model_path: Path to model directory containing the text encoder
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type to use for model weights and computation
            use_cpu: Whether to force the model to stay on CPU
            quantization: Quantization method to use ("disabled", "fp8_e4m3fn", etc.)
        """
        self.logger = logger.bind(component="T5TextEncoder")
        self.device = torch.device("cpu") if use_cpu else torch.device(device)
        self.dtype = dtype
        
        self.logger.info("Loading T5 text encoder", 
                         model_path=model_path, 
                         device=str(self.device),
                         dtype=str(dtype))
        
        # Find tokenizer path - looks in standard locations
        if os.path.exists(os.path.join(model_path, "tokenizer")):
            tokenizer_path = os.path.join(model_path, "tokenizer")
            self.logger.info("Found tokenizer at model path", path=tokenizer_path)
        elif os.path.exists(os.path.join(model_path, "text_encoder/tokenizer")):
            tokenizer_path = os.path.join(model_path, "text_encoder/tokenizer")
            self.logger.info("Found tokenizer at model path", path=tokenizer_path)
        else:
            # Fallback to HuggingFace's hosted tokenizer
            tokenizer_path = "google/umt5-xxl"
            self.logger.warning("Tokenizer not found locally, using HuggingFace hosted version", 
                               path=tokenizer_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model based on quantization settings
        if quantization != "disabled" and quantization.lower() != "none":
            self._load_quantized(model_path, quantization)
        else:
            self._load_standard(model_path)

        self.logger.info("T5 text encoder loaded successfully")

    def _load_standard(self, model_path: str):
        """
        Load T5 model with standard precision (no quantization).

        Args:
            model_path: Path to the model directory
        """
        # Check for sharded models first (standard HF format with index)
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            self.logger.info("Found sharded model weights with index", path=index_path)
            # For sharded weights, use HF's from_pretrained which handles the indexing
            self.model = T5EncoderModel.from_pretrained(
                model_path,
                device_map=self.device if not self.device.type == "cpu" else None,
                torch_dtype=self.dtype
            )
            self.model.eval().requires_grad_(False)
            return

        # Check different possible paths for non-sharded models
        possible_paths = [
            os.path.join(model_path, "model.safetensors"),
            os.path.join(model_path, "text_encoder.safetensors"),
            os.path.join(model_path, "umt5-xxl-enc-bf16.pth")
        ]

        model_file = None
        for path in possible_paths:
            if os.path.exists(path):
                model_file = path
                self.logger.info("Found model weights", path=model_file)
                break

        if model_file is None:
            self.logger.error("Could not find T5 model weights in any expected location")
            raise FileNotFoundError(f"T5 model weights not found in {model_path}")

        # Load model weights
        # Note: The weights can be in either PyTorch's native format or safetensors
        if model_file.endswith(".safetensors"):
            state_dict = load_file(model_file)
        else:
            state_dict = torch.load(model_file, map_location="cpu")
        
        # Create an empty T5 encoder model
        self.model = T5EncoderModel.from_pretrained("google/umt5-xxl")
        
        # Load our weights into the model
        # Note: Some models might need parameter name mapping if the state dict keys don't match
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            self.logger.warning("Missing keys when loading T5 model", keys=missing_keys)
        if unexpected_keys:
            self.logger.warning("Unexpected keys when loading T5 model", keys=unexpected_keys)
        
        # Set model to eval mode and disable gradients
        self.model.eval().requires_grad_(False)
        
        # Handle precision and device placement
        # Note: We keep certain layers (normalization) at higher precision for stability
        params_to_keep = {"layer_norm", "relative_attention_bias", "norm"}
        for name, param in self.model.named_parameters():
            # Keep normalization layers at higher precision
            if any(keyword in name for keyword in params_to_keep):
                param.data = param.data.to(dtype=torch.float32)
            else:
                param.data = param.data.to(dtype=self.dtype)
        
        # Move to target device
        if not self.device.type == "cpu":
            self.model.to(self.device)
    
    def _load_quantized(self, model_path: str, quantization: str):
        """
        Load T5 model with quantization.
        
        Args:
            model_path: Path to the model directory
            quantization: Quantization method to use
        """
        # Note: This would implement quantized loading (FP8, Int8, etc.)
        # For now, we'll just call the standard loader and log a warning
        self.logger.warning("Quantization requested but not fully implemented yet", 
                           quantization=quantization)
        self._load_standard(model_path)
    
    def encode(
        self, 
        prompts: List[str], 
        negative_prompt: str = ""
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Encode text prompts into embeddings for diffusion conditioning.
        
        This method converts text prompts into embedding tensors that guide
        the diffusion process. Both positive and negative prompts are encoded.
        
        Args:
            prompts: List of text prompts to encode
            negative_prompt: Negative prompt text to encode
            
        Returns:
            Dictionary with 'prompt_embeds' and 'negative_prompt_embeds',
            each containing a list of embedding tensors
        """
        # Log information about the encoding process
        self.logger.debug("Encoding text prompts", 
                         num_prompts=len(prompts))
        
        # Process positive prompts
        prompt_embeds = []
        for i, prompt in enumerate(prompts):
            self.logger.debug(f"Processing prompt {i+1}/{len(prompts)}", 
                             prompt=prompt[:50] + ("..." if len(prompt) > 50 else ""))
            
            # Tokenize text
            # Note: For T5, we use padding to ensure consistent sequence length
            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=512,  # WanVideo uses 512 token context
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Run the model to get embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask
                )
                hidden_states = outputs.last_hidden_state
            
            # Extract embeddings up to the actual sequence length (not padding)
            seq_len = tokens.attention_mask.sum(dim=1).long()
            hidden_states = [h[:s] for h, s in zip(hidden_states, seq_len)]
            
            prompt_embeds.append(hidden_states[0])
        
        # Process negative prompt (same process as positive)
        if negative_prompt:
            tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask
                )
                hidden_states = outputs.last_hidden_state
            
            seq_len = tokens.attention_mask.sum(dim=1).long()
            negative_embeds = [h[:s] for h, s in zip(hidden_states, seq_len)]
        else:
            # If no negative prompt, create an empty embedding
            # Note: This is just a placeholder - the actual diffusion model will handle this
            negative_embeds = [torch.zeros(1, self.model.config.d_model, device=self.device)]
        
        self.logger.debug("Text encoding complete",
                         prompt_embed_shapes=[e.shape for e in prompt_embeds],
                         negative_embed_shape=negative_embeds[0].shape)
        
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_embeds
        }