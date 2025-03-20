# utils/teacache.py
import torch
import structlog
from typing import Dict, Any, Optional

logger = structlog.get_logger()

class TeaCache:
    """
    TeaCache for diffusion models to speed up inference.
    
    TeaCache works by caching intermediate model outputs and skipping
    diffusion model forward passes when the expected output would be
    similar enough to the previous step. This is conceptually similar
    to kv-caching in LLMs, but applied to diffusion models.
    
    For details see: https://github.com/ali-vilab/TeaCache
    """
    
    def __init__(
        self,
        rel_l1_thresh: float = 0.15,
        start_step: int = 0,
        end_step: int = -1,
        cache_device: Optional[torch.device] = None,
        use_polynomial_coefficients: bool = True
    ):
        """
        Initialize TeaCache.
        
        Args:
            rel_l1_thresh: Threshold for relative L1 distance to trigger caching
            start_step: First step to apply caching (to avoid early skipping)
            end_step: Last step to apply caching (-1 for all steps)
            cache_device: Device to store cache on (None for same as model)
            use_polynomial_coefficients: Whether to use polynomial rescaling
        """
        self.logger = logger.bind(component="TeaCache")
        self.rel_l1_thresh = rel_l1_thresh
        self.start_step = start_step
        self.end_step = end_step
        self.cache_device = cache_device
        self.use_polynomial_coefficients = use_polynomial_coefficients
        
        # Polynomial coefficients for different model variants (empirically determined)
        self.coefficients = {
            "14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
            "1_3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
            "i2v_480": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
            "i2v_720": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
        }
        
        # Cache states for different prediction streams
        self.states = {}
        self._next_pred_id = 0
        
        self.logger.info(
            "TeaCache initialized",
            threshold=rel_l1_thresh,
            start_step=start_step,
            end_step=end_step,
            polynomial=use_polynomial_coefficients
        )
    
    def new_prediction(self):
        """Create new prediction state and return its ID."""
        pred_id = self._next_pred_id
        self._next_pred_id += 1
        self.states[pred_id] = {
            'previous_residual': None,
            'accumulated_rel_l1_distance': torch.tensor(0.0),
            'previous_modulated_input': None,
            'skipped_steps': 0
        }
        return pred_id
    
    def update(self, pred_id, **kwargs):
        """Update state for specific prediction stream."""
        if pred_id not in self.states:
            return None
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and self.cache_device is not None:
                value = value.to(self.cache_device)
            self.states[pred_id][key] = value
    
    def get(self, pred_id):
        """Get state for specific prediction stream."""
        return self.states.get(pred_id, {})
    
    def should_compute(
        self, 
        pred_id: int, 
        current_step: int, 
        current_input: torch.Tensor,
        model_variant: str = "14B"
    ) -> bool:
        """
        Determine if computation should be done or skipped.
        
        Args:
            pred_id: Prediction ID for this stream
            current_step: Current denoising step
            current_input: Current model input (time embedding)
            model_variant: Model variant for coefficient selection
            
        Returns:
            Boolean indicating whether computation should be done
        """
        # Always compute if outside valid range
        if not self.start_step <= current_step <= (self.end_step if self.end_step > 0 else float('inf')):
            return True
            
        # Always compute first step for a prediction stream
        if pred_id not in self.states or self.states[pred_id]['previous_modulated_input'] is None:
            return True
            
        # Get previous state
        state = self.states[pred_id]
        prev_input = state['previous_modulated_input'].to(current_input.device)
        accumulated_dist = state['accumulated_rel_l1_distance'].to(current_input.device)
        
        # Calculate relative L1 distance
        if self.use_polynomial_coefficients:
            # Use polynomial transformation for better scaling
            coeffs = self.coefficients.get(model_variant, self.coefficients["14B"])
            rel_diff = ((current_input - prev_input).abs().mean() / 
                        prev_input.abs().mean()).cpu().item()
                        
            # Apply polynomial transformation
            import numpy as np
            poly = np.poly1d(coeffs)
            accumulated_dist = accumulated_dist + torch.tensor(poly(rel_diff), 
                                                              device=accumulated_dist.device)
        else:
            # Direct relative L1 calculation
            rel_diff = ((current_input - prev_input).abs().mean() / 
                        prev_input.abs().mean())
            accumulated_dist = accumulated_dist + rel_diff
        
        # Decide whether to compute or skip
        should_calc = accumulated_dist >= self.rel_l1_thresh
        
        # Reset accumulation if computing
        if should_calc:
            accumulated_dist = torch.tensor(0.0, device=accumulated_dist.device)
        
        # Update state
        self.update(
            pred_id,
            accumulated_rel_l1_distance=accumulated_dist
        )
        
        return should_calc
    
    def apply_cached_residual(self, pred_id: int, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cached residual to input tensor.
        
        Args:
            pred_id: Prediction ID
            x: Input tensor to apply residual to
            
        Returns:
            Updated tensor with residual applied
        """
        if pred_id not in self.states or self.states[pred_id]['previous_residual'] is None:
            return x
            
        residual = self.states[pred_id]['previous_residual'].to(x.device)
        self.update(
            pred_id,
            skipped_steps=self.states[pred_id]['skipped_steps'] + 1
        )
        
        return x + residual
    
    def store_residual(self, pred_id: int, original_x: torch.Tensor, updated_x: torch.Tensor, 
                      current_input: torch.Tensor):
        """
        Store residual between original and updated tensor.
        
        Args:
            pred_id: Prediction ID
            original_x: Original input tensor
            updated_x: Updated output tensor
            current_input: Current model input (time embedding)
        """
        residual = updated_x - original_x
        self.update(
            pred_id,
            previous_residual=residual,
            previous_modulated_input=current_input
        )
    
    def report(self):
        """Report TeaCache statistics."""
        total_skipped = sum(state['skipped_steps'] for state in self.states.values())
        total_preds = len(self.states)
        
        self.logger.info(
            "TeaCache statistics",
            prediction_streams=total_preds,
            total_skipped_steps=total_skipped,
            skipped_per_stream=[state['skipped_steps'] for state in self.states.values()]
        )
    
    def clear_prediction(self, pred_id: int):
        """Clear cache for specific prediction."""
        if pred_id in self.states:
            del self.states[pred_id]
    
    def clear_all(self):
        """Clear all cache states."""
        self.states.clear()
        self._next_pred_id = 0