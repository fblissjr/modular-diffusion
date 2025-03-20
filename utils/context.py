"""
Context Window Utilities for WanVideo Pipeline

This module provides utilities for processing videos in overlapping
windows, which allows handling longer videos with limited memory.
The approach divides the video into windows, processes each separately,
and then blends them together for a seamless result.
"""

import numpy as np
import torch
import math
from typing import List, Tuple, Callable, Generator, Optional

def ordered_halving(val: int) -> float:
    """
    Generate a deterministic fraction using bit-reversed ordering.
    
    This creates a sequence that appears random but is deterministic,
    which helps with better distribution of windows.
    
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

def does_window_roll_over(window: List[int], num_frames: int) -> Tuple[bool, int]:
    """
    Check if a window rolls over the end of the video.
    
    Args:
        window: List of frame indices
        num_frames: Total number of frames
        
    Returns:
        Tuple of (rolls_over, rollover_index)
    """
    prev_val = -1
    for i, val in enumerate(window):
        val = val % num_frames
        if val < prev_val:
            return True, i
        prev_val = val
    return False, -1

def shift_window_to_start(window: List[int], num_frames: int):
    """
    Shift a window to start at frame 0.
    
    This is used to handle windows that wrap around the end of the video.
    
    Args:
        window: List of frame indices
        num_frames: Total number of frames
    """
    start_val = window[0]
    for i in range(len(window)):
        # Normalize relative to start and handle wrapping
        window[i] = ((window[i] - start_val) + num_frames) % num_frames

def shift_window_to_end(window: List[int], num_frames: int):
    """
    Shift a window to end at the final frame.
    
    Args:
        window: List of frame indices
        num_frames: Total number of frames
    """
    # First shift to start
    shift_window_to_start(window, num_frames)
    
    # Calculate shift to end
    end_val = window[-1]
    end_delta = num_frames - end_val - 1
    
    # Apply shift
    for i in range(len(window)):
        window[i] = window[i] + end_delta

def get_missing_indexes(windows: List[List[int]], num_frames: int) -> List[int]:
    """
    Find frames not covered by any window.
    
    Args:
        windows: List of window frame indices
        num_frames: Total number of frames
        
    Returns:
        List of uncovered frame indices
    """
    all_indexes = list(range(num_frames))
    
    # Remove covered indices
    for w in windows:
        for val in w:
            try:
                all_indexes.remove(val)
            except ValueError:
                # Index might have been removed already
                pass
                
    return all_indexes

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
    
    This creates a set of overlapping windows that cover the entire
    video, with careful handling of wrapping and overlaps.
    
    Args:
        step: Current denoising step
        num_steps: Total number of denoising steps
        num_frames: Total number of frames
        context_size: Size of each context window
        context_stride: Stride between context steps
        context_overlap: Overlap between windows
        closed_loop: Whether to treat the video as looping
        
    Returns:
        List of windows, each a list of frame indices
    """
    windows = []
    
    # If video fits in one window, return a single window
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
            # Create window and handle wrapping
            window = [e % num_frames for e in range(j, j + context_size * context_step, context_step)]
            windows.append(window)
    
    # Handle windows that roll over
    delete_idxs = []
    win_i = 0
    
    while win_i < len(windows):
        # Check if window rolls over
        is_roll, roll_idx = does_window_roll_over(windows[win_i], num_frames)
        
        if is_roll:
            # Handle rollover
            roll_val = windows[win_i][roll_idx]
            shift_window_to_end(windows[win_i], num_frames=num_frames)
            
            # Add a new window if needed
            if roll_val not in windows[(win_i+1) % len(windows)]:
                windows.insert(win_i+1, list(range(roll_val, roll_val + context_size)))
        
        # Check for duplicate windows
        for pre_i in range(0, win_i):
            if windows[win_i] == windows[pre_i]:
                delete_idxs.append(win_i)
                break
                
        win_i += 1
    
    # Delete duplicate windows
    delete_idxs.reverse()
    for i in delete_idxs:
        windows.pop(i)
        
    return windows

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
    
    This creates a fixed set of windows regardless of step number,
    which can be more stable for some videos.
    
    Args:
        step: Current denoising step (unused)
        num_steps: Total number of denoising steps (unused)
        num_frames: Total number of frames
        context_size: Size of each context window
        context_stride: Stride between context steps (unused)
        context_overlap: Overlap between windows
        closed_loop: Whether to treat the video as looping (unused)
        
    Returns:
        List of windows, each a list of frame indices
    """
    windows = []
    
    # If video fits in one window, return a single window
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
            final_start_idx = start_idx - final_delta
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
    which can be more memory efficient.
    
    Args:
        step: Current denoising step
        num_steps: Total number of denoising steps
        num_frames: Total number of frames
        context_size: Size of each context window
        context_stride: Stride between context steps
        context_overlap: Overlap between windows
        closed_loop: Whether to treat the video as looping
        
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

def get_context_scheduler(name: str) -> Callable:
    """
    Get the appropriate context scheduling function.
    
    Args:
        name: Name of scheduling method
        
    Returns:
        Context scheduling function
    """
    if name == "uniform_looped":
        return uniform_looped
    elif name == "uniform_standard":
        return uniform_standard
    elif name == "static_standard":
        return static_standard
    else:
        raise ValueError(f"Unknown context scheduler: {name}")

def get_total_steps(
    scheduler: Callable,
    timesteps: List[int],
    num_frames: int = 0,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
) -> int:
    """
    Calculate total processing steps needed for all windows.
    
    Args:
        scheduler: Context scheduling function
        timesteps: List of timesteps
        num_frames: Total number of frames
        context_size: Size of each context window
        context_stride: Stride between context steps
        context_overlap: Overlap between windows
        closed_loop: Whether to treat the video as looping
        
    Returns:
        Total number of processing steps
    """
    return sum(
        len(
            list(
                scheduler(
                    i,
                    len(timesteps),
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )