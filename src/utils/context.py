"""
Context Window Utilities for WanVideo Pipeline

This module provides utilities for processing videos in overlapping
windows, which allows handling longer videos with limited memory.
The approach divides the video into windows, processes each separately,
and then blends them together for a seamless result.
"""

import torch
import logging
import math
import numpy as np
from typing import List, Dict, Optional, Union, Callable, Any, Generator, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def ordered_halving(val: int) -> float:
    """
    Generate a deterministic fraction with bit-reversed ordering.

    This creates a sequence that appears random but is deterministic,
    similar to quasi-random sampling techniques in Monte Carlo methods.

    Args:
        val: Integer value to convert

    Returns:
        Fraction between 0 and 1
    """
    # Convert to binary and reverse
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]

    # Convert back to integer and normalize
    as_int = int(bin_flip, 2)
    return as_int / (1 << 64)


class WindowTracker:
    """
    Track and manage context window states.

    This maintains state for each window across denoising steps,
    similar to how LLMs track key-value caches for different contexts.
    """

    def __init__(self, verbose: bool = False):
        """Initialize window tracker."""
        self.window_states = {}
        self.teacache_states = {}
        self.verbose = verbose

    def get_window_id(self, frame_indices: List[int]) -> str:
        """Get a unique ID for a window based on its frames."""
        return f"window_{min(frame_indices)}_{max(frame_indices)}"

    def get_teacache_streams(self, window_id: str, global_streams: Dict) -> Dict:
        """
        Get TeaCache streams for a specific window.

        Args:
            window_id: Unique window identifier
            global_streams: Global TeaCache streams

        Returns:
            Window-specific TeaCache streams
        """
        # If we don't have streams for this window yet, create them
        if window_id not in self.teacache_states:
            self.teacache_states[window_id] = {}
            for stream_name, stream_id in global_streams.items():
                # Each window gets its own TeaCache prediction stream
                self.teacache_states[window_id][stream_name] = stream_id

        return self.teacache_states[window_id]


class ContextStrategy:
    """
    Base class for context window strategies.

    This is the strategy pattern for handling different ways
    of processing video frames, similar to how LLMs have different
    attention patterns for different context lengths.
    """

    def __init__(self, config, device):
        """
        Initialize strategy.

        Args:
            config: Context configuration
            device: Computation device
        """
        self.config = config
        self.device = device
        self.tracker = WindowTracker()

    def process_frames(
        self,
        model,
        latents,
        timestep,
        text_embeds,
        step_idx,
        total_steps,
        guidance_scale=1.0,
        teacache_streams=None,
    ):
        """
        Process frames with the strategy.

        Args:
            model: Diffusion model
            latents: Latent tensor
            timestep: Current timestep
            text_embeds: Text embeddings
            step_idx: Current step index
            total_steps: Total number of steps
            guidance_scale: Classifier-free guidance scale
            teacache_streams: TeaCache streams

        Returns:
            Processed latent tensor
        """
        raise NotImplementedError("Subclasses must implement process_frames")


class StandardContextStrategy(ContextStrategy):
    """
    Standard strategy - process all frames at once.

    This is the simplest approach, similar to how small LLMs
    can process the entire context in one go.
    """

    def process_frames(
        self,
        model,
        latents,
        timestep,
        text_embeds,
        step_idx,
        total_steps,
        guidance_scale=1.0,
        teacache_streams=None,
    ):
        """Process all frames at once."""
        # Handle conditional and unconditional paths for classifier-free guidance
        with torch.no_grad():
            # Current percentage through sampling process
            step_percentage = step_idx / total_steps

            # Unconditional path (if guidance scale > 1)
            if guidance_scale > 1.0:
                uncond_output = model(
                    [latents.clone()],
                    timestep,
                    [text_embeds["negative_prompt_embeds"][0]],
                    seq_len=latents.shape[2]
                    * latents.shape[3]
                    * latents.shape[4],  # Calculate sequence length
                    is_uncond=True,
                    current_step_percentage=step_percentage,
                )[0][0]

            # Conditional path
            cond_output = model(
                [latents],
                timestep,
                [text_embeds["prompt_embeds"][0]],
                seq_len=latents.shape[2]
                * latents.shape[3]
                * latents.shape[4],  # Calculate sequence length
                is_uncond=False,
                current_step_percentage=step_percentage,
            )[0][0]

        # Combine outputs for classifier-free guidance
        if guidance_scale > 1.0:
            model_output = uncond_output + guidance_scale * (
                cond_output - uncond_output
            )
        else:
            model_output = cond_output

        return model_output


class WindowedContextStrategy(ContextStrategy):
    """
    Windowed strategy - process frames in overlapping windows.

    This allows processing longer videos with limited memory,
    similar to how LLMs use sliding window attention for long contexts.
    """

    def __init__(self, config, device):
        """Initialize windowed strategy."""
        super().__init__(config, device)

        # Get appropriate context scheduler
        self.scheduler = get_context_scheduler(config.schedule_type)

        logger.info(
            f"Using windowed context strategy - "
            f"schedule={config.schedule_type}, "
            f"size={config.latent_size}, "
            f"stride={config.latent_stride}, "
            f"overlap={config.latent_overlap}"
        )

    def get_windows(self, step_idx, total_steps, num_frames):
        """
        Get context windows for the current step.

        Args:
            step_idx: Current step index
            total_steps: Total number of steps
            num_frames: Total number of frames

        Returns:
            List of window frame indices
        """
        return list(
            self.scheduler(
                step_idx,
                total_steps,
                num_frames,
                self.config.latent_size,
                self.config.latent_stride,
                self.config.latent_overlap,
                self.config.closed_loop,
            )
        )

    def process_frames(
        self,
        model,
        latents,
        timestep,
        text_embeds,
        step_idx,
        total_steps,
        guidance_scale=1.0,
        teacache_streams=None,
    ):
        """
        Process frames in overlapping windows.

        Args:
            model: Diffusion model
            latents: Latent tensor
            timestep: Current timestep
            text_embeds: Text embeddings
            step_idx: Current step index
            total_steps: Total number of steps
            guidance_scale: Classifier-free guidance scale
            teacache_streams: TeaCache streams

        Returns:
            Processed latent tensor
        """
        # Get windows for this step
        windows = self.get_windows(step_idx, total_steps, latents.shape[2])

        # Prepare accumulators for blending
        noise_pred_combined = torch.zeros_like(latents)
        weight_combined = torch.zeros(
            (1, 1, latents.shape[2], 1, 1), device=self.device, dtype=latents.dtype
        )

        # Current percentage through sampling process
        step_percentage = step_idx / total_steps

        # Process each window
        for window_idx, frame_indices in enumerate(windows):
            # Get window ID for TeaCache
            window_id = self.tracker.get_window_id(frame_indices)

            # Select frames for this window
            window_latents = latents[:, :, frame_indices, :, :]

            # Select appropriate text embeddings
            window_text_embeds = self._select_text_embeds(
                text_embeds, frame_indices, latents.shape[2]
            )

            # Get TeaCache streams for this window
            window_teacache = None
            if teacache_streams:
                window_teacache = self.tracker.get_teacache_streams(
                    window_id, teacache_streams
                )

            # Process window
            window_output = self._process_window(
                model,
                window_latents,
                timestep,
                window_text_embeds,
                step_percentage,
                guidance_scale,
                window_teacache,
            )

            # Create blending mask
            window_mask = self._create_blending_mask(
                window_output,
                frame_indices,
                latents.shape[2],
                self.config.latent_overlap,
            )

            # Blend into combined output
            noise_pred_combined[:, :, frame_indices, :, :] += (
                window_output * window_mask
            )
            weight_combined[:, :, frame_indices, :, :] += window_mask

        # Normalize by weights
        epsilon = 1e-8  # Avoid division by zero
        result = noise_pred_combined / (weight_combined + epsilon)

        return result

    def _process_window(
        self,
        model,
        window_latents,
        timestep,
        window_text_embeds,
        step_percentage,
        guidance_scale,
        teacache_streams,
    ):
        """
        Process a single window.

        Args:
            model: Diffusion model
            window_latents: Latent tensor for this window
            timestep: Current timestep
            window_text_embeds: Text embeddings for this window
            step_percentage: Current percentage through sampling
            guidance_scale: Classifier-free guidance scale
            teacache_streams: TeaCache streams for this window

        Returns:
            Window model output
        """
        # Calculate sequence length
        seq_len = (
            window_latents.shape[2] * window_latents.shape[3] * window_latents.shape[4]
        )

        with torch.no_grad():
            # Unconditional pass (if using guidance)
            if guidance_scale > 1.0:
                uncond_latents = window_latents.clone()
                uncond_output = model(
                    [uncond_latents],
                    timestep,
                    [window_text_embeds["negative_prompt_embeds"][0]],
                    seq_len=seq_len,
                    is_uncond=True,
                    current_step_percentage=step_percentage,
                )[0][0]

            # Conditional pass
            cond_output = model(
                [window_latents],
                timestep,
                [window_text_embeds["prompt_embeds"][0]],
                seq_len=seq_len,
                is_uncond=False,
                current_step_percentage=step_percentage,
            )[0][0]

        # Combine outputs for classifier-free guidance
        if guidance_scale > 1.0:
            window_output = uncond_output + guidance_scale * (
                cond_output - uncond_output
            )
        else:
            window_output = cond_output

        return window_output

    def _select_text_embeds(self, text_embeds, frame_indices, total_frames):
        """
        Select appropriate text embeddings for a window.

        For multiple prompts, this selects based on window position.

        Args:
            text_embeds: All text embeddings
            frame_indices: Window frame indices
            total_frames: Total number of frames

        Returns:
            Text embeddings for this window
        """
        # If only one prompt, use it
        if len(text_embeds["prompt_embeds"]) == 1:
            return text_embeds

        # Otherwise select based on window position
        num_prompts = len(text_embeds["prompt_embeds"])
        window_center = sum(frame_indices) / len(frame_indices)
        prompt_idx = min(
            int(window_center / total_frames * num_prompts), num_prompts - 1
        )

        return {
            "prompt_embeds": [text_embeds["prompt_embeds"][prompt_idx]],
            "negative_prompt_embeds": text_embeds["negative_prompt_embeds"],
        }

    def _create_blending_mask(self, tensor, frame_indices, total_frames, overlap):
        """
        Create a blending mask for smooth transitions between windows.

        Args:
            tensor: Tensor to create mask for
            frame_indices: Frame indices in this window
            total_frames: Total number of frames
            overlap: Number of frames to overlap

        Returns:
            Blending mask tensor
        """
        # Create base mask - all ones
        mask = torch.ones_like(tensor)

        # No blending needed if window covers all frames or no overlap
        if len(frame_indices) >= total_frames or overlap <= 0:
            return mask

        # Apply left-side blending if not starting at first frame
        if min(frame_indices) > 0:
            # Create ramp from 0 to 1 over overlap frames
            ramp_up = torch.linspace(0, 1, overlap, device=tensor.device)
            # Add dimensions to match tensor shape
            ramp_up = ramp_up.view(1, 1, -1, 1, 1)
            # Apply to beginning of mask (up to overlap frames)
            overlap_idx = min(overlap, mask.shape[2])
            mask[:, :, :overlap_idx] = ramp_up[:, :, :overlap_idx]

        # Apply right-side blending if not ending at last frame
        if max(frame_indices) < total_frames - 1:
            # Create ramp from 1 to 0 over overlap frames
            ramp_down = torch.linspace(1, 0, overlap, device=tensor.device)
            # Add dimensions to match tensor shape
            ramp_down = ramp_down.view(1, 1, -1, 1, 1)
            # Apply to end of mask (last overlap frames)
            mask[:, :, -overlap:] = ramp_down

        return mask


def get_context_scheduler(name: str) -> Callable:
    """
    Get the appropriate context scheduling function.

    Args:
        name: Scheduler name

    Returns:
        Scheduling function
    """
    if name == "uniform_looped":
        return uniform_looped
    elif name == "uniform_standard":
        return uniform_standard
    elif name == "static_standard":
        return static_standard
    else:
        logger.warning(f"Unknown context scheduler: {name}, using uniform_standard")
        return uniform_standard


def uniform_standard(
    step: int,
    num_steps: Optional[int] = None,
    num_frames: int = 0,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
) -> List[List[int]]:
    """
    Generate context windows using uniform standard method.

    This creates windows with deterministic spacing that changes
    based on the current step, similar to how LLMs use different
    attention patterns at different layers.

    Args:
        step: Current denoising step
        num_steps: Total number of steps
        num_frames: Total number of frames
        context_size: Size of context window
        context_stride: Stride between steps
        context_overlap: Overlap between windows
        closed_loop: Whether video is treated as looping

    Returns:
        List of windows, each a list of frame indices
    """
    windows = []

    # If video fits in one window, return all frames
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    # Adjust context_stride to avoid too many windows
    max_stride = int(np.ceil(np.log2(num_frames / context_size))) + 1
    context_stride = min(context_stride, max_stride)

    # Generate windows at different strides
    for context_step in 1 << np.arange(context_stride):
        # Calculate offset based on current step
        pad = int(round(num_frames * ordered_halving(step)))

        # Generate windows with this stride
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            # Create window and handle wrapping if closed loop
            window = [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]
            windows.append(window)

    # Check for windows that wrap around the end
    for i, window in enumerate(windows):
        # Check if window decreases (wraps around)
        wraps = False
        for j in range(1, len(window)):
            if window[j] < window[j - 1]:
                wraps = True
                break

        # Adjust wrapped windows
        if wraps and closed_loop:
            # Shift to make continuous
            min_val = min(window)
            windows[i] = [(x - min_val) % num_frames for x in window]

    # Remove duplicate windows
    unique_windows = []
    window_strs = set()
    for window in windows:
        window_str = ",".join(map(str, window))
        if window_str not in window_strs:
            window_strs.add(window_str)
            unique_windows.append(window)

    return unique_windows


def static_standard(
    step: int = 0,
    num_steps: Optional[int] = None,
    num_frames: int = 0,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
) -> List[List[int]]:
    """
    Generate context windows using static standard method.

    This creates a fixed set of evenly spaced windows,
    similar to how some LLMs use fixed-stride sparse attention.

    Args:
        step: Current denoising step (unused)
        num_steps: Total number of steps (unused)
        num_frames: Total number of frames
        context_size: Size of context window
        context_stride: Stride between steps (unused)
        context_overlap: Overlap between windows
        closed_loop: Whether video is treated as looping (unused)

    Returns:
        List of windows, each a list of frame indices
    """
    windows = []

    # If video fits in one window, return all frames
    if num_frames <= context_size:
        windows.append(list(range(num_frames)))
        return windows

    # Calculate step size between windows
    delta = context_size - context_overlap

    # Generate evenly spaced windows
    for start_idx in range(0, num_frames, delta):
        # Check if this window extends past the end
        ending = start_idx + context_size

        if ending >= num_frames:
            # Adjust window to fit
            final_delta = ending - num_frames
            final_start_idx = max(0, start_idx - final_delta)
            windows.append(list(range(final_start_idx, final_start_idx + context_size)))
            break

        # Normal window
        windows.append(list(range(start_idx, start_idx + context_size)))

    return windows


def uniform_looped(
    step: int = 0,
    num_steps: Optional[int] = None,
    num_frames: int = 0,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
) -> Generator[List[int], None, None]:
    """
    Generate context windows using uniform looped method.

    This is a generator version that yields windows one at a time,
    which can be more memory efficient for very long videos.

    Args:
        step: Current denoising step
        num_steps: Total number of steps
        num_frames: Total number of frames
        context_size: Size of context window
        context_stride: Stride between steps
        context_overlap: Overlap between windows
        closed_loop: Whether video is treated as looping

    Yields:
        Lists of frame indices for each window
    """
    # If video fits in one window, yield a single window
    if num_frames <= context_size:
        yield list(range(num_frames))
        return
    
    # Adjust context_stride to avoid too many windows
    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)
    
    # Generate windows at different strides
    for context_step in 1 << np.arange(context_stride):
        # Calculate offset based on current step
        pad = int(round(num_frames * ordered_halving(step)))
        
        # Generate windows with this stride
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            # Create window and handle wrapping
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]


def create_context_strategy(config, device):
    """
    Create the appropriate context strategy based on configuration.

    Args:
        config: Context configuration
        device: Computation device

    Returns:
        Context strategy instance
    """
    if not config.enabled:
        logger.info("Using standard context strategy (no windowing)")
        return StandardContextStrategy(config, device)
    else:
        logger.info(
            f"Using windowed context strategy with scheduler: {config.schedule_type}"
        )
        return WindowedContextStrategy(config, device)