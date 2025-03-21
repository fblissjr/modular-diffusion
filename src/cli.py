# cli.py
import os
import torch
import logging
import time
import argparse
import json
from pathlib import Path
import random
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WanVideo Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    model_group.add_argument(
        "--block_swap",
        type=int,
        default=20,
        help="Number of transformer blocks to offload (0 to disable)",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for computation",
    )
    model_group.add_argument(
        "--attention",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attn", "xformers"],
        help="Attention implementation",
    )

    # Generation options
    gen_group = parser.add_argument_group("Generation Options")
    gen_group.add_argument("--prompt", type=str, required=True, help="Text prompt")
    gen_group.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, distorted",
        help="Negative text prompt",
    )
    gen_group.add_argument("--height", type=int, default=480, help="Video height")
    gen_group.add_argument("--width", type=int, default=832, help="Video width")
    gen_group.add_argument(
        "--frames",
        type=int,
        default=25,
        help="Number of frames",
    )
    gen_group.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    gen_group.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale",
    )
    gen_group.add_argument(
        "--shift",
        type=float,
        default=5.0,
        help="Flow matching shift parameter",
    )
    gen_group.add_argument(
        "--seed", type=int, default=None, help="Random seed (None for random)"
    )
    gen_group.add_argument(
        "--scheduler",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++", "dpm++_sde", "euler"],
        help="Sampling scheduler to use",
    )

    # Context options
    ctx_group = parser.add_argument_group("Context Options")
    ctx_group.add_argument(
        "--context_windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use context windows for processing",
    )
    ctx_group.add_argument(
        "--context_size",
        type=int,
        default=81,
        help="Size of context window in frames",
    )
    ctx_group.add_argument(
        "--context_stride",
        type=int,
        default=4,
        help="Stride between context windows",
    )
    ctx_group.add_argument(
        "--context_overlap",
        type=int,
        default=16,
        help="Overlap between context windows",
    )

    # Memory options
    mem_group = parser.add_argument_group("Memory Options")
    mem_group.add_argument(
        "--vae_tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use VAE tiling to save memory",
    )
    mem_group.add_argument(
        "--teacache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TeaCache to speed up generation",
    )
    mem_group.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile for faster execution",
    )

    # Output options
    out_group = parser.add_argument_group("Output Options")
    out_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )
    out_group.add_argument(
        "--output_type",
        type=str,
        default="mp4",
        choices=["mp4", "gif", "png"],
        help="Output file type",
    )
    out_group.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for video output",
    )

    # Advanced options
    adv_group = parser.add_argument_group("Advanced Options")
    adv_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def auto_output_path(prompt: str) -> str:
    """
    Generate automatic output path based on prompt and settings.

    Args:
        prompt: Text prompt

    Returns:
        Auto-generated output path
    """
    # Clean prompt for filename (take first few words)
    prompt_words = prompt.split()[:5]
    prompt_part = "_".join(word.lower() for word in prompt_words if word.isalnum())

    # Format with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{prompt_part}_{timestamp}"

    # Create directory if needed
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)

    # Return full path
    return os.path.join(save_dir, f"{prefix}.mp4")


class ProgressCallback:
    """
    Simple progress callback with updates to console.

    Similar to tqdm but with customized output for the CLI.
    """

    def __init__(self, total_steps, display_interval=1):
        """
        Initialize progress callback.

        Args:
            total_steps: Total number of steps
            display_interval: How often to update display
        """
        self.total_steps = total_steps
        self.display_interval = display_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def __call__(self, step, total_steps=None, preview=None):
        """
        Update progress.

        Args:
            step: Current step
            total_steps: Total steps (overrides initialization value)
            preview: Optional preview image
        """
        if total_steps is not None:
            self.total_steps = total_steps

        # Calculate progress
        progress = (step + 1) / self.total_steps
        elapsed = time.time() - self.start_time

        # Only update at specified intervals to avoid console spam
        if step % self.display_interval == 0 or step == self.total_steps - 1:
            # Calculate speed and ETA
            it_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            eta = (self.total_steps - step - 1) / it_per_sec if it_per_sec > 0 else 0

            # Print progress bar and stats
            bar_length = 20
            filled_length = int(progress * bar_length)
            bar = "=" * filled_length + ">" + " " * (bar_length - filled_length - 1)

            print(
                f"\r[{bar}] {step + 1}/{self.total_steps} ({progress * 100:.1f}%) "
                f"- {it_per_sec:.2f} it/s - ETA: {eta:.1f}s",
                end="",
                flush=True,
            )

            # Update last time
            self.last_update_time = time.time()

        # Print newline at the end
        if step == self.total_steps - 1:
            total_time = time.time() - self.start_time
            print(
                f"\nGeneration complete in {total_time:.2f}s "
                f"({self.total_steps / total_time:.2f} steps/s)"
            )


def save_generation_settings(args, output_path):
    """
    Save generation settings alongside the output.

    Args:
        args: Arguments
        output_path: Output path

    Returns:
        Path to settings file
    """
    # Convert args to dict
    settings = vars(args).copy()
    settings["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Add paths
    settings["output_path"] = output_path

    # Save as JSON
    settings_path = os.path.splitext(output_path)[0] + "_settings.json"
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    return settings_path


def main():
    """Main entry point for CLI."""
    # Parse arguments and set up logging
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.root.setLevel(log_level)

    # Determine output path
    output_path = args.output or auto_output_path(args.prompt)
    logger.info(f"Output will be saved to: {output_path}")

    # Import our configuration
    from .utils.config import (
        WanVideoConfig,
        MemoryConfig,
        ContextConfig,
        TeaCacheConfig,
        GenerationConfig,
    )

    # Create configuration
    memory_config = MemoryConfig(
        block_swap_count=args.block_swap,
        vae_tiling=args.vae_tiling,
        efficient_attention=args.attention if args.attention != "sdpa" else None,
        use_torch_compile=args.compile,
    )

    context_config = ContextConfig(
        enabled=args.context_windows,
        size=args.context_size,
        stride=args.context_stride,
        overlap=args.context_overlap,
    )

    teacache_config = TeaCacheConfig(
        enabled=args.teacache,
        threshold=0.15,  # Default threshold
        start_step=5,  # Skip early critical steps
        end_step=-1,  # All steps after start
        cache_device="cpu",  # Store cache on CPU to save VRAM
    )

    generation_config = GenerationConfig(
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        scheduler_type=args.scheduler,
        shift=args.shift,
        seed=args.seed,
        fps=args.fps,
        output_type=args.output_type,
    )

    config = WanVideoConfig(
        model_path=args.model_path,
        device="cuda",
        dtype=args.dtype,
        memory=memory_config,
        context=context_config,
        teacache=teacache_config,
        generation=generation_config,
    )

    # Load the pipeline
    from modular_diffusion.pipelines.wanvideo_pipeline import WanVideoPipeline

    logger.info(f"Loading model from {args.model_path}")

    # Create the pipeline
    t_start = time.time()
    pipeline = WanVideoPipeline(
        model_path=args.model_path,
        config=config,
    )
    t_load = time.time() - t_start
    logger.info(f"Model loaded in {t_load:.2f}s")

    # Create progress callback
    callback = ProgressCallback(args.steps)

    # Run inference
    logger.info(f"Generating video for prompt: {args.prompt}")
    t_start = time.time()

    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        callback=callback,
    )

    t_inference = time.time() - t_start

    # Log results
    logger.info(f"Generation complete in {t_inference:.2f}s")
    logger.info(f"Steps per second: {args.steps / t_inference:.2f}")

    # Save the video
    pipeline.save_video(
        video=output.video[0],
        output_path=output_path,
        fps=args.fps,
        metadata=output.metadata,
    )

    # Save settings
    settings_path = save_generation_settings(args, output_path)
    logger.info(f"Settings saved to {settings_path}")

    print(f"\nOutput saved to {output_path}")

if __name__ == "__main__":
    main()