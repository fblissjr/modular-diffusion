# cli.py
import os
import torch
import structlog
import time
import argparse
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_logging():
    """Set up structured logging."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )
    return structlog.get_logger()


class ProgressCallback:
    """
    Simple progress callback with updates to console.

    Similar to tqdm but with customized output for the CLI.
    """

    def __init__(self, total_steps, display_interval=1):
        self.total_steps = total_steps
        self.display_interval = display_interval
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def __call__(self, step, latent_preview=None, **kwargs):
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


def get_default_config_path():
    """Get path to default config file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "configs", "default_config.json")


def load_config(config_path=None):
    """Load configuration from JSON file."""
    config_path = config_path or get_default_config_path()

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        # Create default config if it doesn't exist
        default_config = {
            "models": {
                "default": {
                    "path": "",
                    "block_swap": 20,
                    "attention_mode": "sdpa",
                    "use_teacache": True,
                }
            },
            "generation": {
                "width": 832,
                "height": 480,
                "frames": 25,
                "steps": 30,
                "guidance_scale": 6.0,
                "shift": 5.0,
                "context_windows": True,
            },
            "memory": {
                "use_vae_tiling": True,
                "vae_tile_size": [272, 272],
                "vae_tile_stride": [144, 128],
            },
            "output": {"type": "mp4", "save_dir": "outputs", "fps": 16},
        }

        # Create config directory if needed
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        return default_config


def parse_args():
    """Parse command-line arguments."""
    config = load_config()
    gen_config = config.get("generation", {})
    mem_config = config.get("memory", {})
    out_config = config.get("output", {})

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
    gen_group.add_argument(
        "--height", type=int, default=gen_config.get("height", 480), help="Video height"
    )
    gen_group.add_argument(
        "--width", type=int, default=gen_config.get("width", 832), help="Video width"
    )
    gen_group.add_argument(
        "--frames",
        type=int,
        default=gen_config.get("frames", 25),
        help="Number of frames",
    )
    gen_group.add_argument(
        "--steps",
        type=int,
        default=gen_config.get("steps", 30),
        help="Number of inference steps",
    )
    gen_group.add_argument(
        "--guidance_scale",
        type=float,
        default=gen_config.get("guidance_scale", 6.0),
        help="Classifier-free guidance scale",
    )
    gen_group.add_argument(
        "--shift",
        type=float,
        default=gen_config.get("shift", 5.0),
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
        default=gen_config.get("context_windows", True),
        help="Use context windows for processing",
    )
    ctx_group.add_argument(
        "--context_size",
        type=int,
        default=gen_config.get("context_size", 81),
        help="Size of context window in frames",
    )
    ctx_group.add_argument(
        "--context_stride",
        type=int,
        default=gen_config.get("context_stride", 4),
        help="Stride between context windows",
    )
    ctx_group.add_argument(
        "--context_overlap",
        type=int,
        default=gen_config.get("context_overlap", 16),
        help="Overlap between context windows",
    )

    # Memory options
    mem_group = parser.add_argument_group("Memory Options")
    mem_group.add_argument(
        "--vae_tiling",
        action=argparse.BooleanOptionalAction,
        default=mem_config.get("use_vae_tiling", True),
        help="Use VAE tiling to save memory",
    )
    mem_group.add_argument(
        "--teacache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TeaCache to speed up generation",
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
        default=out_config.get("type", "mp4"),
        choices=["mp4", "gif", "png"],
        help="Output file type",
    )
    out_group.add_argument(
        "--fps",
        type=int,
        default=out_config.get("fps", 16),
        help="Frames per second for video output",
    )

    # Advanced options
    adv_group = parser.add_argument_group("Advanced Options")
    adv_group.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    adv_group.add_argument(
        "--save_config",
        action="store_true",
        help="Save current settings as default config",
    )
    adv_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def auto_output_path(args):
    """Generate automatic output path based on prompt and settings."""
    # Clean prompt for filename (take first few words)
    prompt_words = args.prompt.split()[:5]
    prompt_part = "_".join(word.lower() for word in prompt_words if word.isalnum())

    # Format with settings
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{prompt_part}_{timestamp}"

    # Create directory if needed
    config = load_config()
    save_dir = config.get("output", {}).get("save_dir", "outputs")
    os.makedirs(save_dir, exist_ok=True)

    # Return full path
    extension = "." + args.output_type
    return os.path.join(save_dir, f"{prefix}{extension}")


def save_generation_settings(args, output_path):
    """Save generation settings alongside the output."""
    # Create settings dict
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
    logger = setup_logging()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.root.setLevel(log_level)

    # Save config if requested
    if args.save_config:
        config = load_config()

        # Update config with current settings
        config["generation"]["height"] = args.height
        config["generation"]["width"] = args.width
        config["generation"]["frames"] = args.frames
        config["generation"]["steps"] = args.steps
        config["generation"]["guidance_scale"] = args.guidance_scale
        config["generation"]["shift"] = args.shift
        config["generation"]["context_windows"] = args.context_windows

        # Save updated config
        config_path = args.config or get_default_config_path()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {config_path}")

    # Determine output path
    output_path = args.output or auto_output_path(args)
    logger.info(f"Output will be saved to: {output_path}")

    # Load the pipeline
    logger.info(f"Loading model from {args.model_path}")

    # Map dtype string to torch dtype
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Import at runtime to allow CLI to work without all dependencies
    try:
        from pipelines.wanvideo_pipeline import WanVideoPipeline
        from utils.memory import MemoryConfig
    except ImportError:
        from modular_diffusion.pipelines.wanvideo_pipeline import WanVideoPipeline
        from modular_diffusion.utils.memory import MemoryConfig

    # Configure memory settings
    memory_config = MemoryConfig(
        dtype=dtype,
        block_swap_count=args.block_swap,
        vae_tiling=args.vae_tiling,
    )

    # Configure TeaCache settings if enabled
    teacache_config = None
    if args.teacache:
        teacache_config = {
            "rel_l1_thresh": 0.15,  # Default threshold
            "start_step": 5,  # Skip early steps that are critical
            "end_step": -1,  # All steps after start
            "cache_device": "cpu",  # Store cache on CPU to save VRAM
            "use_polynomial_coefficients": True,
        }

    # Create the pipeline
    t_start = time.time()
    pipeline = WanVideoPipeline(
        model_path=args.model_path,
        device="cuda",
        dtype=dtype,
        memory_config=memory_config,
        enable_teacache=args.teacache,
        teacache_config=teacache_config,
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
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        scheduler=args.scheduler,
        seed=args.seed,
        use_context_windows=args.context_windows,
        context_size=args.context_size,
        context_stride=args.context_stride,
        context_overlap=args.context_overlap,
        enable_vae_tiling=args.vae_tiling,
        output_type=args.output_type,
        save_path=output_path,
        fps=args.fps,
        callback=callback,
    )

    t_inference = time.time() - t_start

    # Log results
    logger.info(f"Generation complete in {t_inference:.2f}s")
    logger.info(f"Steps per second: {args.steps / t_inference:.2f}")

    # Save settings
    settings_path = save_generation_settings(args, output_path)
    logger.info(f"Settings saved to {settings_path}")

    print(f"\nOutput saved to {output_path}")

if __name__ == "__main__":
    main()