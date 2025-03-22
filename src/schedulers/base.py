# src/schedulers/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple, Optional
import torch

class Scheduler(ABC):
    """
    Base interface for diffusion schedulers.
    
    This provides a common interface for different scheduler implementations,
    analogous to how LLM samplers/decoders have standard interfaces.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize scheduler with configuration.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config or {}
        self.timesteps = None
        self.num_inference_steps = None
    
    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu"):
        """
        Set timesteps for inference.
        
        Args:
            num_inference_steps: Number of steps for inference
            device: Device to place timesteps on
        """
        pass
    
    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Perform a single denoising step.
        
        Args:
            model_output: Output from diffusion model
            timestep: Current timestep
            sample: Current sample (noisy latent)
            return_dict: Whether to return as dict
            
        Returns:
            Updated sample after denoising step
        """
        pass
    
    @property
    def init_noise_sigma(self) -> float:
        """
        Get initial noise sigma.
        
        Returns:
            Initial noise level
        """
        # Default implementation
        return 1.0