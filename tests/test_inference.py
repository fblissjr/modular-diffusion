# test_inference.py

import os
import torch
import time
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inference_test")

# Import our components
from modular_diffusion.pipelines.wanvideo_pipeline import WanVideoPipeline
from modular_diffusion.utils.memory import MemoryTracker
from modular_diffusion.utils.memory_config import MemoryConfig

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test WanVideo inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean",
        help="Text prompt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, distorted",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output", type=str, default="output.mp4", help="Output file path"
    )
    parser.add_argument(
        "--steps", type=int, default=30, help="Number of inference steps"
    )
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--frames", type=int, default=25, help="Number of frames")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type",
    )
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Torch compile mode",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="none",
        choices=["none", "fp8_e4m3fn", "int8_dynamic"],
        help="Quantization method",
    )
    parser.add_argument(
        "--block_swap", type=int, default=0, help="Number of blocks to swap to CPU"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (None for random)"
    )
    parser.add_argument(
        "--context_windows",
        action="store_true",
        help="Use context windows for processing",
    )
    parser.add_argument(
        "--no_vae_tiling",
        action="store_false",
        dest="vae_tiling",
        help="Disable VAE tiling",
    )
    
    return parser.parse_args()

def main():
    """Main entry point for inference test."""
    args = parse_args()
    
    # Configure memory tracking
    memory_tracker = MemoryTracker()
    memory_tracker.log_memory_usage("Before model loading")
    
    # Map dtype string to torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize memory configuration
    memory_config = MemoryConfig(
        dtype=dtype,
        quantize_method=args.quantize,
        use_torch_compile=args.use_compile,
        compile_mode=args.compile_mode,
        block_swap_count=args.block_swap,
        vae_tiling=args.vae_tiling,
    )

    logger.info(f"Loading WanVideo pipeline from {args.model_path}")
    t0 = time.time()
    
    # Load pipeline
    pipeline = WanVideoPipeline(
        model_path=args.model_path,
        device="cuda",
        dtype=dtype,
        memory_config=memory_config,
    )

    load_time = time.time() - t0
    logger.info(f"Pipeline loaded in {load_time:.2f} seconds")
    memory_tracker.log_memory_usage("After model loading")
    
    # Run inference
    logger.info(f"Running inference with prompt: {args.prompt}")
    t0 = time.time()
    
    output = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_context_windows=args.context_windows,
        output_type="mp4",
        save_path=args.output,
    )

    inference_time = time.time() - t0
    
    # Log results
    memory_tracker.log_memory_usage("After inference")
    logger.info(f"Inference complete in {inference_time:.2f} seconds")
    logger.info(f"Steps per second: {args.steps / inference_time:.2f}")
    logger.info(f"Output saved to: {args.output}")
    
    # Basic output validation
    if hasattr(output, "video") and output.video is not None:
        video = output.video[0]
        logger.info(f"Video shape: {video.shape}")
        logger.info(
            f"Video range: {video.min().item():.3f} to {video.max().item():.3f}"
        )
    else:
        logger.warning("Warning: No video output generated")

if __name__ == "__main__":
    main()