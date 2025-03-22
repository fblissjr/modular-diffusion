# src/models/base/scheduler.py
from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    """Base interface for all schedulers in the modular-diffusion project."""
    
    @abstractmethod
    def set_timesteps(self, num_inference_steps, device="cpu"):
        """Set up timesteps for inference."""
        pass
    
    @abstractmethod
    def step(self, model_output, timestep, sample, return_dict=True):
        """Perform a single denoising step."""
        pass

# we can make all our schedulers inherit from this