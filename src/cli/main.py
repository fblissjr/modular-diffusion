# cli/main.py
import argparse
import logging
import os
import sys
import json
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.factory import ComponentFactory
from src.pipelines.wanvideo.pipeline import WanVideoPipeline

def main():
    """Main CLI entry point."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Modular Diffusion - Video generation with WanVideo"
    )
    
    # Add arguments
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--model_path", type=str, help="Base model path")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output file path")
    parser.add_argument("--height", type=int, default=320, help="Video height")
    parser.add_argument("--width", type=int, default=576, help="Video width")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--fps", type=int, default=8, help="Output video FPS")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu)")
    parser.add_argument("--dtype", type=str, default="bf16", 
                       choices=["fp32", "fp16", "bf16"], help="Model data type")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = load_config(args)
    
    # Create pipeline
    try:
        pipeline = create_pipeline(config)
        
        # Set up generator if seed provided
        generator = torch.Generator(device=args.device).manual_seed(args.seed) if args.seed else None
        
        # Generate video
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator
        )
        
        # Save output
        pipeline.save_output(output["video"], args.output, fps=args.fps)
        
        print(f"Generation complete! Output saved to {args.output}")
        print(f"Time taken: {output['generation_time']:.2f}s ({output['fps']:.2f} frames/sec)")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

def load_config(args):
    """
    Load and merge configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Load from file if specified
    if args.config:
        with open(args.config, "r") as f:
            if args.config.endswith(".json"):
                config = json.load(f)
            elif args.config.endswith((".yaml", ".yml")):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {args.config}")
    
    # Override with command line arguments
    if args.model_path:
        config["model_path"] = args.model_path
        
    # Set device and dtype
    config["device"] = args.device
    config["dtype"] = args.dtype
    
    # Set generation parameters
    config["height"] = args.height
    config["width"] = args.width
    config["num_frames"] = args.frames
    config["num_inference_steps"] = args.steps
    config["guidance_scale"] = args.guidance_scale
    config["fps"] = args.fps
    
    return config

def create_pipeline(config):
    """
    Create pipeline from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Pipeline instance
    """
    # Import torch here to avoid importing it if there's an error in args
    import torch
    
    # Create pipeline
    pipeline = WanVideoPipeline(config)
    
    return pipeline

if __name__ == "__main__":
    main()