# utils/config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """
    Configuration for memory management strategies.
    
    Controls various techniques to reduce GPU memory usage:
    - Data type selection (FP32, FP16, BF16)
    - Quantization (FP8, INT8)
    - Block swapping between GPU and CPU
    - VAE tiling for efficient processing
    
    Similar to LLM quantization, these techniques help run larger models
    on limited hardware with controlled quality trade-offs.
    """
    block_swap_count: int = 0  # Number of transformer blocks to swap
    vae_tiling: bool = True  # Use tiling for VAE operations
    vae_tile_size: Tuple[int, int] = (272, 272)  # Size of tiles (height, width)
    vae_tile_stride: Tuple[int, int] = (144, 128)  # Stride between tiles
    text_encoder_offload: bool = True  # Offload text encoder when not in use
    efficient_attention: Optional[str] = None  # "sdpa", "flash_attn", etc.
    quantize_method: str = "none"  # "none", "fp8_e4m3fn", "int8_dynamic"
    use_torch_compile: bool = False  # Use torch.compile for faster execution
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    keep_norm_fp32: bool = True  # Keep normalization layers in FP32

@dataclass
class ContextConfig:
    """
    Configuration for context window processing.
    
    Controls how videos are processed in chunks to reduce memory usage:
    - Window size and overlap
    - Scheduling strategy for windows
    - Blending parameters between windows
    
    Similar to how LLMs use sliding window attention for long contexts.
    """
    enabled: bool = True  # Use context windows
    schedule_type: str = "uniform_standard"  # "uniform_standard", "uniform_looped", "static"
    size: int = 81  # Window size (frames)
    stride: int = 4  # Step size between windows
    overlap: int = 16  # Overlap between windows
    closed_loop: bool = True  # Treat video as looping for windows at the end
    
    def __post_init__(self):
        """Update dependent values to match latent space."""
        # Convert from pixel space to latent space (4x temporal downsampling)
        self.latent_size = (self.size - 1) // 4 + 1
        self.latent_stride = max(1, self.stride // 4)
        self.latent_overlap = max(1, self.overlap // 4)

@dataclass
class TeaCacheConfig:
    """
    Configuration for TeaCache optimization.
    
    Controls the adaptive skipping of diffusion steps to speed up inference:
    - Threshold for skipping calculations
    - When to start/stop using the cache
    - How to store cached results
    
    Conceptually similar to kv-caching in LLMs, but for diffusion steps.
    """
    enabled: bool = False  # Use TeaCache optimization
    threshold: float = 0.15  # Relative L1 distance threshold
    start_step: int = 5  # First step to apply caching (avoid early critical steps)
    end_step: int = -1  # Last step to apply caching (-1 for all remaining)
    use_coefficients: bool = True  # Use polynomial rescaling coefficients
    cache_device: str = "cpu"  # Device to store cache on
    model_variant: str = "14B"  # Model variant for coefficient selection

@dataclass
class GenerationConfig:
    """
    Configuration for video generation parameters.
    
    Controls the core aspects of the generation process:
    - Video dimensions
    - Number of steps
    - Guidance scale
    - Scheduler parameters
    """
    height: int = 480  # Video height
    width: int = 832  # Video width
    num_frames: int = 25  # Number of frames to generate
    fps: int = 16  # Frames per second for output
    num_inference_steps: int = 30  # Number of denoising steps
    guidance_scale: float = 6.0  # Classifier-free guidance scale
    scheduler_type: str = "unipc"  # "unipc", "dpm++", "euler"
    shift: float = 5.0  # Flow matching shift parameter
    seed: Optional[int] = None  # Random seed (None for random)
    output_type: str = "mp4"  # "mp4", "gif", "png"

@dataclass
class WanVideoConfig:
    """
    Master configuration for WanVideo pipeline.
    
    Combines all configuration aspects into a single structure for
    easy management and validation.
    """
    # Core settings
    model_path: str = ""  # Path to model directory
    model_type: str = "t2v"  # "t2v" or "i2v"
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: str = "bf16"  # "fp32", "fp16", "bf16"
    
    # Component configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    teacache: TeaCacheConfig = field(default_factory=TeaCacheConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    def to_torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch.dtype."""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        return dtype_map.get(self.dtype, torch.float32)
    
    def validate(self) -> List[str]:
        """Validate configuration for potential issues."""
        issues = []
        
        # Check for device compatibility
        if self.device == "cuda" and not torch.cuda.is_available():
            issues.append("CUDA requested but not available, will use CPU instead")
        
        # Check for BF16 compatibility
        if self.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            issues.append("BF16 requested but not supported, falling back to FP16")
        
        # Check for reasonable video dimensions
        if self.generation.height % 8 != 0 or self.generation.width % 8 != 0:
            issues.append("Video dimensions should be multiples of 8 for optimal processing")
        
        # Check for TeaCache + context window compatibility
        if self.teacache.enabled and self.context.enabled:
            issues.append("TeaCache with context windows may cause inconsistent results")
        
        # Check for model path
        if not self.model_path:
            issues.append("Model path not specified")
        
        return issues