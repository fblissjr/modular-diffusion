# tests/quick_test.py
import torch
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipelines.wanvideo.pipeline import WanVideoPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory tracking helper
def log_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def main():
    try:
        logger.info("Testing WanVideo pipeline...")
        
        # Set up configuration
        config = {
            "model_path": "./checkpoints",  # Base path
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dtype": "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32",
            # Minimal generation config for test
            "height": 320,
            "width": 576,
            "num_frames": 16,
            "num_inference_steps": 10
        }
        
        # Log initial memory
        log_memory()
        
        # Create pipeline
        logger.info("Initializing pipeline...")
        pipeline = WanVideoPipeline(config)
        logger.info("Pipeline initialized!")
        
        # Log memory after loading
        log_memory()
        
        # Run inference
        logger.info("Running inference...")
        output = pipeline(
            prompt="A beautiful sunset over the ocean",
            negative_prompt="worst quality, blurry"
        )
        logger.info("Inference complete!")
        
        # Log final memory
        log_memory()
        
        # Save output
        output_path = "test_output.mp4"
        logger.info(f"Saving to {output_path}...")
        pipeline.save_output(output["video"], output_path, fps=8)
        logger.info(f"Done! Output saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)