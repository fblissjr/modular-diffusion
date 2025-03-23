# src/models/text_encoders/t5.py
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
from transformers import T5EncoderModel, AutoTokenizer, AutoConfig

from src.core.component import Component
from src.models.text_encoders.base import TextEncoder
from src.core.registry import register_component

logger = logging.getLogger(__name__)

@register_component("T5TextEncoder", TextEncoder)
class T5TextEncoder(TextEncoder):
    """
    T5-based text encoder adapter. 
    This wraps Hugging Face's T5 implementation to provide a consistent
    interface for text encoding across different models.
    """
    
    def __init__(s,c):
        """
        Initialize T5 text encoder.
        
        Args:
            config: Encoder configuration
        """
        #super().__init__(config)

        super().__init__(c)
        s.m_path=c.get("model_path")or exit("model_path required")
        s.t_name=c.get("tokenizer","google/umt5-xxl")
        s.m_len=c.get("max_length",512)
        s.cpu_off=c.get("cpu_offload",False)
        s.on_dem=c.get("load_on_demand",False)
        
        s.tkz=AutoTokenizer.from_pretrained(s.t_name)
        if not s.on_dem:s._load_model()
        logger.info(f"Init T5 with m_len={s.m_len},cpu_off={s.cpu_off},on_dem={s.on_dem}")

    def _load_model(s):
        p=Path(s.m_path)
        if p.is_file() and str(p).endswith('.safetensors'):
            cfg=AutoConfig.from_pretrained(s.t_name)
            s.t5=T5EncoderModel(config=cfg)
            import safetensors.torch as st
            s.t5.load_state_dict(st.load_file(p),strict=False)
        else:s.t5=T5EncoderModel.from_pretrained(p)
        
        dev='cpu' if s.cpu_off else s.device
        s.t5.to(dev,s.dtype).eval()
        return s.t5

    def _encode_text(s,txt):
        e=[]
        if hasattr(s,'t5') or s._load_model():
            for t in txt:
                i=s.tkz(t,return_tensors="pt",padding="max_length",truncation=True,max_length=s.m_len)
                i={k:v.to(s.device) for k,v in i.items()}
                with torch.no_grad():
                    # Move to target device only if offloaded
                    if s.cpu_off:s.t5=s.t5.to(s.device)
                    o=s.t5(**i).last_hidden_state
                    if s.cpu_off:
                        s.t5=s.t5.to('cpu')
                        torch.cuda.empty_cache()
                    e.append(o)
                return e        
        # # Get paths and parameters
        # model_path = config.get("model_path")
        # if not model_path:
        #     raise ValueError("model_path is required for T5TextEncoder")
            
        # tokenizer_name = config.get("tokenizer", "google/umt5-xxl")
        # max_length = config.get("max_length", 512)
        
        # # Load tokenizer directly from HuggingFace
        # logger.info(f"Loading tokenizer from {tokenizer_name}")
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # # Load model with safetensors support
        # logger.info(f"Loading T5 encoder from {model_path}")
        # model_path = Path(model_path)
        
        # if model_path.is_file() and str(model_path).endswith('.safetensors'):
        #     # Create a model config for UMT5-XXL
        #     model_config = AutoConfig.from_pretrained(tokenizer_name)
            
        #     # Create empty model with the right config
        #     self.t5_model = T5EncoderModel(config=model_config)
            
        #     # Load weights from safetensors
        #     import safetensors.torch
        #     state_dict = safetensors.torch.load_file(model_path)
            
        #     # Load state dict
        #     self.t5_model.load_state_dict(state_dict, strict=False)
        # else:
        #     # Standard loading
        #     self.t5_model = T5EncoderModel.from_pretrained(model_path)
        
        # # Move to device and set dtype
        # self.t5_model.to(self.device, self.dtype)
        
        # # Make sure model is in eval mode
        # self.t5_model.eval()
        
        # # Store config
        # self.max_length = max_length
        
        # logger.info(f"Initialized T5TextEncoder with max_length={max_length}")
        
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
                
            # Return both prompt and negative prompt embeddings
            return {
                "prompt_embeds": self._encode_text(prompt),
                "negative_prompt_embeds": self._encode_text(negative_prompt)
            }
        else:
            # Return only prompt embeddings
            return {
                "prompt_embeds": self._encode_text(prompt)
            }
    
    # def _encode_text(self, text_list: List[str]) -> List[torch.Tensor]:
    #     """
    #     Encode list of text prompts.
        
    #     Args:
    #         text_list: List of text prompts
            
    #     Returns:
    #         List of embedding tensors
    #     """
    #     embeddings = []
        
    #     for text in text_list:
    #         # Tokenize text
    #         inputs = self.tokenizer(
    #             text,
    #             return_tensors="pt",
    #             padding="max_length",
    #             truncation=True,
    #             max_length=self.max_length
    #         )
            
    #         # Move to device
    #         inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
    #         # Get embeddings
    #         with torch.no_grad():
    #             output = self.t5_model(**inputs)
                
    #         # Get last hidden states
    #         text_embeds = output.last_hidden_state
            
    #         # Store embedding
    #         embeddings.append(text_embeds)
            
    #     return embeddings
        
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