# schedulers/flow_dpm_scheduler.py

import torch
import numpy as np
import math
from typing import List, Optional, Tuple, Union
import structlog

logger = structlog.get_logger()

class FlowDPMScheduler:
    """
    DPM++ scheduler adapted for flow matching diffusion models.
    
    DPM++ offers faster convergence than Euler methods with higher
    quality results. This is similar to the solver in LLM architecture
    search when optimizing step count vs quality.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "flow_prediction",
        shift: float = 5.0,
        algorithm_type: str = "dpmsolver++",
        solver_order: int = 2,
        lower_order_final: bool = True,
        final_sigmas_type: str = "zero",
    ):
        """
        Initialize DPM++ scheduler for flow matching.
        
        Args:
            num_train_timesteps: Number of diffusion steps used in training
            prediction_type: Type of model prediction ("flow_prediction")
            shift: Flow matching shift parameter
            algorithm_type: DPM solver algorithm ("dpmsolver++", "dpmsolver", "sde")
            solver_order: Order for the solver (1, 2, or 3)
            lower_order_final: Whether to use lower order for final steps
            final_sigmas_type: Type of final sigma ("zero" or "sigma_min")
        """
        self.logger = logger.bind(component="FlowDPMScheduler")
        
        # Store configuration
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.shift = shift
        self.algorithm_type = algorithm_type
        self.solver_order = solver_order
        self.lower_order_final = lower_order_final
        self.final_sigmas_type = final_sigmas_type
        
        # Initialize timesteps and sigmas (noise levels)
        self.timesteps = None
        self.sigmas = None
        
        # Storage for multi-step solver
        self.model_outputs = None
        self.timestep_list = None
        self.counter = 0
        
        # Create the base sigmas (noise levels)
        self._init_sigmas()
        
        self.logger.info(
            "Initialized FlowDPM scheduler", 
            prediction_type=prediction_type,
            shift=shift,
            algorithm=algorithm_type,
            order=solver_order
        )
    
    def _init_sigmas(self):
        """Initialize base sigmas for the noise schedule."""
        # Create alpha values
        alphas = np.linspace(1, 1 / self.num_train_timesteps, self.num_train_timesteps)[::-1].copy()
        self.base_sigmas = 1.0 - alphas
        
        # Convert to torch tensor
        self.base_sigmas = torch.from_numpy(self.base_sigmas).to(dtype=torch.float32)
    
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
        shift: Optional[float] = None,
    ):
        """
        Set up the timestep schedule for inference.
        
        Args:
            num_inference_steps: Number of steps for inference
            device: Device to place tensors on
            shift: Optional override for shift parameter
        """
        shift = shift or self.shift
        
        self.logger.info(
            "Setting up DPM++ timesteps", 
            num_steps=num_inference_steps,
            shift=shift
        )
        
        # Create sigmas for DPM schedule - use log space for better spacing
        sigma_min, sigma_max = 0.02, 0.98  # Reasonable range
        
        # Calculate sigmas in log space
        if self.algorithm_type == "dpmsolver++":
            # More uniform spacing in log space for better quality
            sigmas = torch.linspace(
                math.log(sigma_max), math.log(sigma_min), num_inference_steps + 1
            ).exp()
        else:
            # Traditional linear spacing
            sigmas = torch.linspace(sigma_max, sigma_min, num_inference_steps + 1)
        
        # Apply shift to sigmas based on flow matching
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
        # Set final sigma based on config
        if self.final_sigmas_type == "sigma_min":
            sigma_last = self.base_sigmas[-1].item()
        elif self.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(f"Unknown final_sigmas_type: {self.final_sigmas_type}")
        
        # Convert sigmas to timesteps
        timesteps = sigmas * self.num_train_timesteps
        
        # Add final value and convert to tensors
        sigmas = torch.cat([sigmas, torch.tensor([sigma_last])]).to(
            device=device, dtype=torch.float32
        )
        
        self.timesteps = timesteps.to(device=device, dtype=torch.int64)
        self.sigmas = sigmas
        self.num_inference_steps = len(timesteps)
        
        # Initialize storage for multi-step solver
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.counter = 0
        
        self.logger.debug(
            "DPM++ timesteps configured",
            timesteps_range=[self.timesteps[0].item(), self.timesteps[-1].item()], 
            sigmas_range=[self.sigmas[0].item(), self.sigmas[-1].item()]
        )
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator = None,
        return_dict: bool = True,
    ) -> Union[dict, Tuple[torch.Tensor]]:
        """
        Implement a DPM++ solver step.
        
        Args:
            model_output: Flow prediction from diffusion model
            timestep: Current timestep
            sample: Current latent sample
            generator: Optional random generator for stochastic processes
            return_dict: Whether to return as dict
            
        Returns:
            Updated sample with less noise
        """
        # Get index for current timestep
        step_index = (self.timesteps == timestep).nonzero().item()
        
        # Get current and next sigma
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
        
        # Store model output for multi-step
        self.model_outputs[self.counter % self.solver_order] = model_output
        self.timestep_list[self.counter % self.solver_order] = step_index
        
        # Determine which order to use
        if self.counter < self.solver_order - 1:
            # Not enough previous steps - use Euler
            order = 1
        elif self.lower_order_final and step_index == self.num_inference_steps - 1:
            # Final step with lower order option
            order = 1
        else:
            # Regular case - use configured order
            order = self.solver_order
        
        # Apply the solver step
        if order == 1:
            # First-order (Euler) step
            derivative = model_output
            sample = sample - (sigma - sigma_next) * derivative
        else:
            # Higher-order DPM++ step
            if self.algorithm_type == "dpmsolver++":
                sample = self._dpmpp_solver_step(
                    step_index, order, sample, sigma, sigma_next, model_output
                )
            else:
                # Fallback to simpler method for other algorithm types
                sample = self._simple_solver_step(
                    step_index, order, sample, sigma, sigma_next, model_output
                )
        
        # Increment counter
        self.counter += 1
        
        # Return the output
        if return_dict:
            return {"prev_sample": sample}
        else:
            return (sample,)
    
    def _dpmpp_solver_step(
        self, step_index, order, sample, sigma, sigma_next, model_output
    ):
        """
        DPM++ solver step implementation.
        
        Args:
            step_index: Current step index
            order: Current solver order to use
            sample: Current sample
            sigma: Current sigma value
            sigma_next: Next sigma value
            model_output: Current model output
            
        Returns:
            Updated sample
        """
        if order == 2:
            # Second-order DPM++ step
            sigma_mid = (sigma + sigma_next) / 2
            
            # First prediction
            x_mid = sample - 0.5 * (sigma - sigma_next) * model_output
            
            # Second model evaluation at midpoint
            model_output_mid = self.model_outputs[(self.counter - 1) % self.solver_order]
            
            # Final update
            sample = sample - (sigma - sigma_next) * (
                0.5 * model_output + 0.5 * model_output_mid
            )
        elif order == 3:
            # Third-order DPM++ step
            sigma_mid = (sigma + sigma_next) / 2
            
            # Get multiple model outputs
            m_0 = model_output
            m_1 = self.model_outputs[(self.counter - 1) % self.solver_order]
            m_2 = self.model_outputs[(self.counter - 2) % self.solver_order]
            
            # Calculate coefficients
            h_0 = sigma - sigma_next
            r_0 = sigma / sigma_next
            D_0 = m_0
            D_1 = (1 / h_0) * (m_0 - m_1)
            D_2 = (2 / h_0**2) * (m_0 - 2 * m_1 + m_2)
            
            # Calculate update using Taylor expansion
            sample = sample - h_0 * (D_0 + 0.5 * h_0 * D_1 + (1/6) * h_0**2 * D_2)
        
        return sample
        
    def _simple_solver_step(
        self, step_index, order, sample, sigma, sigma_next, model_output
    ):
        """
        Simpler solver step for other algorithm types.
        
        Args:
            step_index: Current step index
            order: Current solver order to use
            sample: Current sample
            sigma: Current sigma value
            sigma_next: Next sigma value
            model_output: Current model output
            
        Returns:
            Updated sample
        """
        # Simple weighted average of model outputs
        if order == 1:
            derivative = model_output
        elif order == 2:
            m_0 = model_output
            m_1 = self.model_outputs[(self.counter - 1) % self.solver_order]
            derivative = (3 * m_0 - m_1) / 2
        elif order == 3:
            m_0 = model_output
            m_1 = self.model_outputs[(self.counter - 1) % self.solver_order]
            m_2 = self.model_outputs[(self.counter - 2) % self.solver_order]
            derivative = (23 * m_0 - 16 * m_1 + 5 * m_2) / 12
        
        # Simple Euler step with the calculated derivative
        sample = sample - (sigma - sigma_next) * derivative
        
        return sample