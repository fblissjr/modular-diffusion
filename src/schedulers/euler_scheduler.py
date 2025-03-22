import torch

# src/schedulers/euler_scheduler.py
class EulerDiscreteScheduler:
    """
    Simple Euler discretization scheduler for diffusion models.
    
    This is a straightforward implementation that's functionally 
    equivalent to the diffusers version but integrated with our codebase.
    """
    
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        
        # Set up beta schedule
        if beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        
        # Compute crucial diffusion parameters
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Additional parameters
        self.prediction_type = prediction_type
        self.timesteps = None
        
    def set_timesteps(self, num_inference_steps, device="cpu"):
        """Set timesteps for inference."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = self.num_train_timesteps - timesteps - 1
        
        self.timesteps = timesteps.to(device=device, dtype=torch.long)
        self.num_inference_steps = len(timesteps)
        
    def step(self, model_output, timestep, sample, return_dict=True):
        """Perform a single denoising step using Euler method."""
        t_index = (self.timesteps == timestep).nonzero().item()
        prev_t_index = t_index + 1
        
        # Handle edge case for last step
        if prev_t_index >= len(self.timesteps):
            prev_timestep = 0
        else:
            prev_timestep = self.timesteps[prev_t_index]
        
        # Get alphas for current and previous step
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        # Compute step parameters
        beta_prod_t = 1 - alpha_cumprod_t
        beta_prod_prev = 1 - alpha_cumprod_prev
        
        # Apply Euler step based on prediction type
        if self.prediction_type == "epsilon":
            # Predict noise
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_cumprod_t.sqrt()
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + beta_prod_prev.sqrt() * model_output
        elif self.prediction_type == "v_prediction":
            # Predict velocity
            pred_original_sample = alpha_cumprod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            pred_prev_sample = alpha_cumprod_prev.sqrt() * pred_original_sample + beta_prod_prev.sqrt() * model_output
        else:
            # Flow prediction (similar to what we use in our other schedulers)
            pred_prev_sample = sample - (alpha_cumprod_prev / alpha_cumprod_t - 1) * model_output
        
        # Return in expected format
        if return_dict:
            return {"prev_sample": pred_prev_sample}
        else:
            return (pred_prev_sample,)