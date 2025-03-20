# utils/teacache.py
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Pre-defined polynomial coefficients for different model variants
TEACACHE_COEFFICIENTS = {
    "14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    "1_3B": [
        2.39676752e03,
        -1.31110545e03,
        2.01331979e02,
        -8.29855975e00,
        1.37887774e-01,
    ],
    "i2v_480": [
        -3.02331670e02,
        2.23948934e02,
        -5.25463970e01,
        5.87348440e00,
        -2.01973289e-01,
    ],
    "i2v_720": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
}

class TeaCache:
    """
    TeaCache for diffusion models to speed up inference.

    TeaCache works by caching intermediate model outputs and skipping
    diffusion model forward passes when the expected output would be
    similar enough to the previous step.

    Conceptually similar to kv-caching in LLMs, but applied to diffusion models.
    """
    
    def __init__(
        self,
        threshold: float = 0.15,
        start_step: int = 5,
        end_step: int = -1,
        cache_device: Optional[torch.device] = None,
        use_coefficients: bool = True,
        model_variant: str = "14B",
    ):
        """
        Initialize TeaCache.

        Args:
            threshold: Threshold for relative L1 distance to skip computation
            start_step: First step to apply caching (to avoid early skipping)
            end_step: Last step to apply caching (-1 for all remaining steps)
            cache_device: Device to store cache on (None for same as model)
            use_coefficients: Whether to use polynomial rescaling coefficients
            model_variant: Model variant for coefficient selection
        """
        self.threshold = threshold
        self.start_step = start_step
        self.end_step = end_step
        self.cache_device = cache_device
        self.use_coefficients = use_coefficients

        # Get polynomial coefficients for this model variant
        self.coefficients = TEACACHE_COEFFICIENTS.get(
            model_variant, TEACACHE_COEFFICIENTS["14B"]
        )

        # Initialize prediction cache
        self.states = {}
        self._next_pred_id = 0

        # Statistics
        self.total_skipped = 0

        logger.info(
            f"TeaCache initialized: "
            f"threshold={threshold}, "
            f"start_step={start_step}, "
            f"end_step={end_step}, "
            f"use_coefficients={use_coefficients}, "
            f"model_variant={model_variant}"
        )

    def new_prediction(self) -> int:
        """
        Create new prediction stream.

        Returns:
            Prediction ID for this stream
        """
        pred_id = self._next_pred_id
        self._next_pred_id += 1

        # Initialize state for this prediction
        self.states[pred_id] = {
            'previous_residual': None,
            'accumulated_rel_l1_distance': torch.tensor(0.0),
            'previous_modulated_input': None,
            'skipped_steps': 0
        }

        return pred_id

    def should_compute(
        self, pred_id: int, step_idx: int, time_modulation: torch.Tensor
    ) -> bool:
        """
        Determine if computation should be performed or skipped.

        Args:
            pred_id: Prediction ID
            step_idx: Current step index
            time_modulation: Current time modulation tensor

        Returns:
            True if computation should be performed, False if skipped
        """
        # Skip check if outside valid range
        if (
            not self.start_step
            <= step_idx
            <= (self.end_step if self.end_step > 0 else float("inf"))
        ):
            return True

        # Always compute first step for a prediction
        if pred_id not in self.states or self.states[pred_id]['previous_modulated_input'] is None:
            return True

        # Get previous state
        state = self.states[pred_id]
        prev_modulation = state["previous_modulated_input"].to(time_modulation.device)
        accumulated_dist = state["accumulated_rel_l1_distance"].to(
            time_modulation.device
        )
        
        # Calculate relative L1 distance
        if self.use_coefficients:
            # Get relative difference
            rel_diff = (
                (
                    (time_modulation - prev_modulation).abs().mean()
                    / prev_modulation.abs().mean()
                )
                .cpu()
                .item()
            )

            # Apply polynomial transformation
            poly = np.poly1d(self.coefficients)
            increment = torch.tensor(poly(rel_diff), device=accumulated_dist.device)
            accumulated_dist = accumulated_dist + increment
        else:
            # Direct L1 calculation
            rel_diff = (
                time_modulation - prev_modulation
            ).abs().mean() / prev_modulation.abs().mean()
            accumulated_dist = accumulated_dist + rel_diff
        
        # Decide whether to compute or skip
        should_compute = accumulated_dist >= self.threshold
        
        # Reset accumulation if computing
        if should_compute:
            accumulated_dist = torch.tensor(0.0, device=accumulated_dist.device)
        else:
            # Track skipped steps
            self.states[pred_id]["skipped_steps"] += 1
            self.total_skipped += 1
        
        # Update state
        self.states[pred_id]["accumulated_rel_l1_distance"] = accumulated_dist

        return should_compute

    def apply_cached_residual(
        self, pred_id: int, latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cached residual to skip computation.

        Args:
            pred_id: Prediction ID
            latents: Input latent tensor

        Returns:
            Updated latent tensor with residual applied
        """
        if pred_id not in self.states or self.states[pred_id]['previous_residual'] is None:
            return latents

        # Apply residual
        residual = self.states[pred_id]["previous_residual"].to(latents.device)
        return latents + residual

    def store_residual(
        self,
        pred_id: int,
        original_latents: torch.Tensor,
        updated_latents: torch.Tensor,
        time_modulation: torch.Tensor,
    ):
        """
        Store residual for future use.

        Args:
            pred_id: Prediction ID
            original_latents: Original input latents
            updated_latents: Updated output latents
            time_modulation: Current time modulation tensor
        """
        # Calculate residual
        residual = updated_latents - original_latents

        # Store in cache
        if self.cache_device is not None:
            residual = residual.to(self.cache_device)
            time_modulation = time_modulation.to(self.cache_device)

        # Update state
        self.states[pred_id].update(
            {"previous_residual": residual, "previous_modulated_input": time_modulation}
        )

    def report_statistics(self):
        """Report TeaCache statistics."""
        total_preds = len(self.states)
        skipped_per_stream = [state["skipped_steps"] for state in self.states.values()]

        logger.info(
            f"TeaCache statistics: "
            f"prediction_streams={total_preds}, "
            f"total_skipped_steps={self.total_skipped}, "
            f"avg_skipped_per_stream={sum(skipped_per_stream) / max(1, len(skipped_per_stream)):.1f}"
        )
    
    def clear_prediction(self, pred_id: int):
        """Clear cache for specific prediction."""
        if pred_id in self.states:
            del self.states[pred_id]
    
    def clear_all(self):
        """Clear all cache states."""
        self.states.clear()
        self._next_pred_id = 0
        self.total_skipped = 0