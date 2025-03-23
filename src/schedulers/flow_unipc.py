# src/schedulers/flow_unipc.py
import torch
import logging
from typing import Dict, List, Union, Tuple, Optional, Any

from src.schedulers.base import Scheduler
from src.core.registry import register_component

# Import original implementation (assuming it's available in our path)
# We could alternatively copy specific functions/classes we need
from src.models.wan import FlowUniPCMultistepScheduler as OriginalFlowUniPC

logger = logging.getLogger(__name__)

@register_component("FlowUniPCScheduler", Scheduler)
class FlowUniPCScheduler(Scheduler):
    """
    Adapter for the WanVideo FlowUniPCMultistepScheduler.
    
    This wraps the original implementation while providing our standard
    scheduler interface, preserving the mathematical precision of the
    original algorithm.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize scheduler with configuration.
        
        Args:
            config: Scheduler configuration
        """
        super().__init__(config or {})
        
        # Extract parameters for the original scheduler
        orig_config = {
            "num_train_timesteps": config.get("num_train_timesteps", 1000),
            "solver_order": config.get("solver_order", 2),
            "prediction_type": config.get("prediction_type", "flow_prediction"),
            "shift": config.get("shift", 1.0),
            "use_dynamic_shifting": config.get("use_dynamic_shifting", False),
            "thresholding": config.get("thresholding", False),
            "dynamic_thresholding_ratio": config.get("dynamic_thresholding_ratio", 0.995),
            "sample_max_value": config.get("sample_max_value", 1.0),
            "predict_x0": config.get("predict_x0", True),
            "solver_type": config.get("solver_type", "bh2"),
            "lower_order_final": config.get("lower_order_final", True),
            "disable_corrector": config.get("disable_corrector", []),
            "solver_p": config.get("solver_p", None),
            "timestep_spacing": config.get("timestep_spacing", "linspace"),
            "steps_offset": config.get("steps_offset", 0),
            "final_sigmas_type": config.get("final_sigmas_type", "zero"),
        }
        
        # Create original scheduler
        self.original_scheduler = OriginalFlowUniPC(**orig_config)
        
        # Store config for our API
        self.num_train_timesteps = config.get("num_train_timesteps", 1000)
        
        logger.info(f"Initialized FlowUniPCScheduler with {self.num_train_timesteps} timesteps")
    
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu", **kwargs):
        """
        Set timesteps for inference.
        
        Args:
            num_inference_steps: Number of steps for inference
            device: Device to place timesteps on
            **kwargs: Additional parameters passed to original scheduler
        """
        # Map our parameters to original scheduler
        self.original_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            **kwargs
        )
        
        # Store for our interface
        self.timesteps = self.original_scheduler.timesteps
        self.num_inference_steps = num_inference_steps
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Perform a single denoising step using the original scheduler.
        
        Args:
            model_output: Output from diffusion model
            timestep: Current timestep
            sample: Current sample (noisy latent)
            return_dict: Whether to return as dict
            **kwargs: Additional parameters
            
        Returns:
            Updated sample after denoising step
        """
        # Delegate to original implementation
        return self.original_scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            return_dict=return_dict,
            **kwargs
        )
    
    @property
    def init_noise_sigma(self) -> float:
        """
        Get initial noise sigma from original scheduler.
        
        Returns:
            Initial noise level
        """
        return getattr(self.original_scheduler, "init_noise_sigma", 1.0)