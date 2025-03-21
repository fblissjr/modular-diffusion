# quick_test.py
import torch
import logging

from src.utils.config import WanVideoConfig, GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory tracking helper
def log_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

try:
    logger.info("Importing modules...")
    from src.utils.config import WanVideoConfig
    from src.pipelines.wanvideo_pipeline import WanVideoPipeline
    logger.info("Imports successful!")
    
    # Create minimal config - adjust model_path!
    config = WanVideoConfig(
        model_path="./checkpoints/Wan2.1-T2V-1.3B-Diffusers",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        # Minimal generation config
        generation=GenerationConfig(
            height=320,  # Lower for testing
            width=576,  # Lower for testing
            num_frames=16,  # Fewer frames for testing
            num_inference_steps=10,  # Fewer steps for testing
        ),
    )
    
    logger.info("Initializing pipeline...")
    log_memory()
    pipeline = WanVideoPipeline(
        model_path=config.model_path,
        config=config
    )
    logger.info("Pipeline initialized!")
    log_memory()
    
    # Test with minimal parameters
    logger.info("Running inference...")
    output = pipeline(
        prompt="A beautiful sunset over the ocean",
    )
    logger.info("Inference complete!")
    log_memory()
    
    # Save output
    output_path = "test_output.mp4"
    logger.info(f"Saving to {output_path}...")
    pipeline.save_video(
        video=output.video[0],
        output_path=output_path,
        fps=16
    )
    logger.info(f"Done! Output saved to {output_path}")
    
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    logger.error(traceback.format_exc())