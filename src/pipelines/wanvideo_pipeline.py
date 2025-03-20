# pipelines/wanvideo_pipeline.py
"""
WanVideo Inference Pipeline

This module provides the main pipeline for running inference with
WanVideo models. It coordinates the text encoder, diffusion model,
and VAE to generate videos from text prompts.

The pipeline implements the full diffusion process:
1. Convert text to embeddings
2. Initialize latent space
3. Iteratively denoise through the diffusion model
4. Decode the final latents to a video
"""

import os
import torch
import numpy as np
import random
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import math
from pathlib import Path

# Import our configuration
from ..utils.config import WanVideoConfig, MemoryConfig, ContextConfig, TeaCacheConfig

logger = logging.getLogger(__name__)


@dataclass
class WanVideoPipelineOutput:
    """
    Output from the WanVideo pipeline.

    Contains the generated video and optionally the latent
    representations and generation metadata.
    """

    video: List[torch.Tensor]
    latents: Optional[torch.Tensor] = None
    fps: int = 16
    metadata: Optional[Dict] = None


class WanVideoPipeline:
    """
    Pipeline for generating videos using WanVideo models.

    This pipeline coordinates:
    - Text encoder for processing prompts
    - Diffusion model for the denoising process
    - VAE for encoding/decoding between pixel and latent space
    - Scheduler for controlling the diffusion process
    - Memory management and optimizations

    The core concepts are similar to LLM text generation, but adapted
    for the diffusion process where we gradually denoise latents
    instead of generating tokens one by one.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[WanVideoConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the WanVideo pipeline.

        Args:
            model_path: Path to model directory
            config: Pipeline configuration (or None for defaults)
            device: Computation device
            dtype: Data type for computation
        """
        self.model_path = Path(model_path)

        # Set up configuration
        self.config = config or WanVideoConfig(
            model_path=str(model_path),
            device=device,
            dtype="bf16"
            if dtype == torch.bfloat16
            else "fp16"
            if dtype == torch.float16
            else "fp32",
        )

        # Validate configuration
        issues = self.config.validate()
        for issue in issues:
            logger.warning(f"Configuration issue: {issue}")

        # Set up device and dtype
        self.device = torch.device(self.config.device)
        self.dtype = self.config.to_torch_dtype()

        logger.info(
            f"Initializing WanVideo pipeline: "
            f"model_path={model_path}, "
            f"device={self.device}, "
            f"dtype={self.dtype}"
        )

        # Set up memory management
        from ..utils.memory import MemoryManager, MemoryTracker

        self.memory_tracker = MemoryTracker()
        self.memory_manager = MemoryManager(
            self.config.memory,
            main_device=self.device,
            offload_device=torch.device("cpu"),
        )

        # Determine model type from path or config
        self.model_type = self._determine_model_type()

        # Load components
        with self.memory_tracker.track_usage("Loading pipeline components"):
            # Load text encoder
            self.text_encoder = self._load_text_encoder()

            # Load diffusion model
            self.diffusion_model = self._load_diffusion_model()

            # Load VAE
            self.vae = self._load_vae()

            # Create scheduler
            self.scheduler = self._create_scheduler()

            # Initialize TeaCache if enabled
            self.teacache = (
                self._setup_teacache() if self.config.teacache.enabled else None
            )

        # Set up context strategy
        from ..utils.context import create_context_strategy

        self.context_strategy = (
            create_context_strategy(self.config.context, self.device)
            if self.config.context.enabled
            else None
        )

        logger.info(f"WanVideo pipeline initialized: model_type={self.model_type}")

    def _determine_model_type(self) -> str:
        """
        Determine whether the model is t2v (text-to-video) or i2v (image-to-video).

        Returns:
            Model type: "t2v" or "i2v"
        """
        # Use type from config if specified
        if self.config.model_type in ["t2v", "i2v"]:
            return self.config.model_type

        # Check for indicators in the path
        path_str = str(self.model_path).lower()
        if "t2v" in path_str or "text-to-video" in path_str:
            logger.info("Detected text-to-video model from path")
            return "t2v"
        elif "i2v" in path_str or "image-to-video" in path_str:
            logger.info("Detected image-to-video model from path")
            return "i2v"

        # Try to read from config file
        config_path = self.model_path / "config.json"
        if config_path.exists():
            import json

            with open(config_path, "r") as f:
                try:
                    config = json.load(f)
                    if "model_type" in config:
                        model_type = config["model_type"]
                        logger.info(f"Found model type in config: {model_type}")
                        return model_type
                except json.JSONDecodeError:
                    pass

        # Default to t2v if we can't determine
        logger.warning("Could not determine model type, defaulting to t2v")
        return "t2v"

    def _load_text_encoder(self):
        """
        Load the text encoder model.

        Returns:
            T5 text encoder model
        """
        from ..models.t5_encoder import T5TextEncoder

        logger.info("Loading T5 text encoder")

        # Find text encoder path
        text_encoder_path = self.model_path / "text_encoder"
        if not text_encoder_path.exists():
            text_encoder_path = self.model_path
            logger.warning(
                f"No dedicated text_encoder directory found, using {text_encoder_path}"
            )

        # Load text encoder
        return T5TextEncoder(
            model_path=str(text_encoder_path),
            device=self.device,
            dtype=self.dtype,
            use_cpu=self.config.memory.text_encoder_offload,
            quantization=self.config.memory.quantize_method,
        )

    def _load_diffusion_model(self):
        """
        Load the diffusion model.

        Returns:
            WanDiT diffusion model
        """
        from ..models.diffusion_model import WanDiT

        logger.info("Loading diffusion model")

        # Determine model parameters based on type
        if self.model_type == "t2v":
            in_dim = 16  # Standard latent dimension

            # Detect model size from path (14B or 1.3B)
            is_14b = "14B" in str(self.model_path) or "14b" in str(self.model_path)
            model_config = {
                "model_type": "t2v",
                "in_dim": in_dim,
                "dim": 5120 if is_14b else 1536,
                "ffn_dim": 13824 if is_14b else 8960,
                "freq_dim": 256,  # For time embeddings
                "text_dim": 4096,  # T5 output dimension
                "out_dim": 16,  # Output latent dimension
                "num_heads": 40 if is_14b else 12,
                "num_layers": 40 if is_14b else 30,
                "patch_size": (1, 2, 2),
                "main_device": self.device,
                "offload_device": torch.device("cpu"),
                "attention_mode": self.config.memory.efficient_attention or "sdpa",
            }
        else:  # i2v
            in_dim = 16
            is_14b = "14B" in str(self.model_path) or "14b" in str(self.model_path)
            model_config = {
                "model_type": "i2v",
                "in_dim": in_dim,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "patch_size": (1, 2, 2),
                "main_device": self.device,
                "offload_device": torch.device("cpu"),
                "attention_mode": self.config.memory.efficient_attention or "sdpa",
            }

        # Find diffusion model weights
        model_paths = [
            self.model_path / "diffusion_model.safetensors",
            self.model_path / "transformer.safetensors",
            self.model_path / "model.safetensors",
        ]

        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                f"Could not find diffusion model weights in {self.model_path}"
            )

        # Load weights
        logger.info(f"Loading diffusion model weights from {model_path}")
        try:
            from safetensors.torch import load_file

            state_dict = load_file(str(model_path))
        except ImportError:
            logger.warning("safetensors not available, falling back to torch.load")
            state_dict = torch.load(str(model_path), map_location="cpu")

        # Create model
        model = WanDiT(**model_config)

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(
                f"Missing keys when loading diffusion model: {len(missing_keys)} keys"
            )
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading diffusion model: {len(unexpected_keys)} keys"
            )

        # Apply memory optimizations
        model = self.memory_manager.optimize_model(model)

        return model

    def _load_vae(self):
        """
        Load the VAE model.

        Returns:
            WanVideoVAE model
        """
        from ..models.vae import WanVideoVAE

        logger.info("Loading VAE")

        # Find VAE weights
        vae_paths = [
            self.model_path / "vae.safetensors",
            self.model_path / "vae" / "model.safetensors",
            self.model_path / "Wan2.1_VAE.pth",
        ]

        vae_path = None
        for path in vae_paths:
            if path.exists():
                vae_path = path
                break

        if vae_path is None:
            raise FileNotFoundError(f"Could not find VAE weights in {self.model_path}")

        # Load weights
        logger.info(f"Loading VAE weights from {vae_path}")
        try:
            if str(vae_path).endswith(".safetensors"):
                from safetensors.torch import load_file

                vae_state_dict = load_file(str(vae_path))
            else:
                vae_state_dict = torch.load(str(vae_path), map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE weights: {e}")

        # Create model
        vae = WanVideoVAE(
            dim=96,  # Default for WanVideo VAE
            z_dim=16,  # Latent dimension
            dtype=self.dtype,
        )

        # Load weights
        vae.load_state_dict(vae_state_dict)

        # VAE initially on CPU to save memory if configured
        if self.config.memory.text_encoder_offload:
            vae.to("cpu")
        else:
            vae.to(self.device)

        return vae

    def _create_scheduler(self):
        """
        Create the scheduler for the diffusion process.

        Returns:
            Scheduler instance
        """
        scheduler_type = self.config.generation.scheduler_type
        logger.info(f"Creating scheduler: {scheduler_type}")

        # Create scheduler based on type
        if scheduler_type == "unipc":
            from ..schedulers.flow_scheduler import FlowUniPCScheduler

            scheduler = FlowUniPCScheduler(
                num_train_timesteps=1000,
                prediction_type="flow_prediction",
                shift=self.config.generation.shift,
            )
        elif scheduler_type == "dpm++" or scheduler_type == "dpm++_sde":
            from ..schedulers.flow_dpm_scheduler import FlowDPMScheduler

            scheduler = FlowDPMScheduler(
                num_train_timesteps=1000,
                prediction_type="flow_prediction",
                shift=self.config.generation.shift,
                algorithm_type="dpmsolver++"
                if scheduler_type == "dpm++"
                else "sde-dpmsolver++",
            )
        elif scheduler_type == "euler":
            # Import a simple Euler scheduler if available
            try:
                from diffusers import EulerDiscreteScheduler

                scheduler = EulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                )
            except ImportError:
                logger.warning(
                    "EulerDiscreteScheduler not available, falling back to FlowUniPCScheduler"
                )
                from ..schedulers.flow_scheduler import FlowUniPCScheduler

                scheduler = FlowUniPCScheduler(
                    num_train_timesteps=1000,
                    prediction_type="flow_prediction",
                    shift=self.config.generation.shift,
                )
        else:
            # Default to UniPC
            logger.warning(
                f"Unknown scheduler type: {scheduler_type}, falling back to FlowUniPCScheduler"
            )
            from ..schedulers.flow_scheduler import FlowUniPCScheduler

            scheduler = FlowUniPCScheduler(
                num_train_timesteps=1000,
                prediction_type="flow_prediction",
                shift=self.config.generation.shift,
            )

        return scheduler

    def _setup_teacache(self):
        """
        Set up TeaCache for optimization.

        Returns:
            TeaCache instance
        """
        from ..utils.teacache import TeaCache

        logger.info("Setting up TeaCache")

        # Determine model variant for coefficients
        if "14B" in str(self.model_path):
            model_variant = "14B"
        elif "1.3B" in str(self.model_path):
            model_variant = "1_3B"
        elif "i2v" in self.model_type.lower() and "480" in str(self.model_path):
            model_variant = "i2v_480"
        elif "i2v" in self.model_type.lower():
            model_variant = "i2v_720"
        else:
            model_variant = "14B"  # Default

        # Create TeaCache
        return TeaCache(
            threshold=self.config.teacache.threshold,
            start_step=self.config.teacache.start_step,
            end_step=self.config.teacache.end_step,
            cache_device=torch.device(self.config.teacache.cache_device),
            use_coefficients=self.config.teacache.use_coefficients,
            model_variant=model_variant,
        )

    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: str = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        shift: Optional[float] = None,
        seed: Optional[int] = None,
        callback: Optional[callable] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[WanVideoPipelineOutput, Dict]:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text prompt or list of prompts
            negative_prompt: Text to avoid in generation
            height: Video height (pixels)
            width: Video width (pixels)
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            shift: Flow matching shift parameter
            seed: Random seed (None for random)
            callback: Function to call for progress updates
            return_dict: Whether to return as a dictionary
            **kwargs: Additional arguments

        Returns:
            Generated video and metadata
        """
        # Make sure prompt is a list
        if isinstance(prompt, str):
            prompt = [prompt]

        # Get values from config if not specified
        height = height or self.config.generation.height
        width = width or self.config.generation.width
        num_frames = num_frames or self.config.generation.num_frames
        num_inference_steps = (
            num_inference_steps or self.config.generation.num_inference_steps
        )
        guidance_scale = guidance_scale or self.config.generation.guidance_scale
        shift = shift or self.config.generation.shift

        # Set random seed for reproducibility
        if seed is None:
            seed = self.config.generation.seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        logger.info(
            f"Generating video: "
            f"prompt='{prompt[0][:50]}{'...' if len(prompt[0]) > 50 else ''}', "
            f"{height}x{width}, {num_frames} frames, {num_inference_steps} steps, "
            f"guidance_scale={guidance_scale}, seed={seed}"
        )

        # 1. Encode the text prompts
        with self.memory_tracker.track_usage("Text encoding"):
            text_embeds = self._encode_text(prompt, negative_prompt)

        # 2. Initialize latent space
        with self.memory_tracker.track_usage("Latent initialization"):
            latents = self._initialize_latents(
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
            )

        # 3. Set up the scheduler
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=self.device,
            shift=shift,
        )
        timesteps = self.scheduler.timesteps

        # 4. Run the denoising loop
        with self.memory_tracker.track_usage("Denoising process"):
            latents = self._run_denoising_loop(
                latents=latents,
                timesteps=timesteps,
                text_embeds=text_embeds,
                guidance_scale=guidance_scale,
                callback=callback,
            )

        # 5. Decode latents to video
        with self.memory_tracker.track_usage("VAE decoding"):
            video = self._decode_latents(latents)

        # Format and return the output
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "num_frames": num_frames,
        }

        if return_dict:
            return WanVideoPipelineOutput(
                video=[video],
                latents=latents.detach().cpu()
                if kwargs.get("return_latents", False)
                else None,
                fps=self.config.generation.fps,
                metadata=metadata,
            )
        else:
            return {
                "video": video,
                "latents": latents.detach().cpu()
                if kwargs.get("return_latents", False)
                else None,
                "fps": self.config.generation.fps,
                "metadata": metadata,
            }

    def _encode_text(
        self,
        prompt: List[str],
        negative_prompt: str,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Encode text prompts to embeddings.

        Args:
            prompt: List of text prompts
            negative_prompt: Negative prompt text

        Returns:
            Dictionary with prompt and negative prompt embeddings
        """
        logger.debug(f"Encoding {len(prompt)} prompts and negative prompt")

        # Use the memory manager to handle device placement
        if hasattr(self.text_encoder, "encode"):
            # Using our custom T5TextEncoder
            return self.text_encoder.encode(prompt, negative_prompt=negative_prompt)
        else:
            # Fallback for other text encoders
            raise NotImplementedError("Only T5TextEncoder is currently supported")

    def _initialize_latents(
        self,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
    ) -> torch.Tensor:
        """
        Initialize random latent tensors for the diffusion process.

        Args:
            height: Video height in pixels
            width: Video width in pixels
            num_frames: Number of frames
            seed: Random seed

        Returns:
            Initial latent tensor
        """
        # Set up the VAE scale (downsampling factors)
        vae_scale = 8  # VAE downsamples by 8x spatially
        temporal_vae_scale = 4  # and 4x temporally

        # Calculate latent dimensions
        latent_height = height // vae_scale
        latent_width = width // vae_scale
        latent_frames = (num_frames - 1) // temporal_vae_scale + 1

        # Create random generator with seed
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Generate random latents
        latents = torch.randn(
            (1, 16, latent_frames, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.float32,  # Initial noise in float32 for stability
        )

        logger.debug(
            f"Initialized latents: "
            f"shape={latents.shape}, "
            f"device={latents.device}, "
            f"dtype={latents.dtype}, "
            f"seed={seed}"
        )

        return latents.to(dtype=self.dtype)

    def _run_denoising_loop(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeds: Dict[str, List[torch.Tensor]],
        guidance_scale: float = 1.0,
        callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Run the denoising diffusion process.

        This is the core of the diffusion model, where we gradually
        denoise the latent representation guided by the text embeddings.

        Args:
            latents: Initial noise tensor
            timesteps: Diffusion timesteps from scheduler
            text_embeds: Text embeddings for conditioning
            guidance_scale: Classifier-free guidance scale
            callback: Function to call for progress updates

        Returns:
            Denoised latent tensor
        """
        # Create progress indicator
        progress = tqdm(
            total=len(timesteps), desc="Generating video", disable=callback is not None
        )

        # Set up TeaCache streams if enabled
        teacache_streams = {}
        if self.teacache:
            teacache_streams["cond"] = self.teacache.new_prediction()
            if guidance_scale > 1.0:
                teacache_streams["uncond"] = self.teacache.new_prediction()

        # Main denoising loop
        for i, t in enumerate(timesteps):
            # Current timestep tensor
            timestep = torch.tensor([t], device=self.device)

            # Current percentage through sampling process
            step_percentage = i / len(timesteps)

            # Process with appropriate context strategy
            if self.context_strategy:
                # Process with context windows
                model_output = self.context_strategy.process_frames(
                    model=self.diffusion_model,
                    latents=latents,
                    timestep=timestep,
                    text_embeds=text_embeds,
                    step_idx=i,
                    total_steps=len(timesteps),
                    guidance_scale=guidance_scale,
                    teacache_streams=teacache_streams,
                )
            else:
                # Process without context windows (all frames at once)
                model_output = self._process_single_step(
                    latents=latents,
                    timestep=timestep,
                    text_embeds=text_embeds,
                    step_idx=i,
                    total_steps=len(timesteps),
                    guidance_scale=guidance_scale,
                    teacache_streams=teacache_streams,
                )

            # Step with scheduler
            latents = self.scheduler.step(
                model_output=model_output, timestep=t, sample=latents
            ).prev_sample

            # Update progress
            if callback is not None:
                # For video, just preview the first frame to save memory
                if hasattr(self, "decode_preview"):
                    preview = self.decode_preview(latents[:, :, 0:1])
                    callback(i, len(timesteps), preview)
                else:
                    callback(i, len(timesteps))
            else:
                progress.update(1)

        # Close progress bar
        progress.close()

        # Report TeaCache statistics if used
        if self.teacache:
            self.teacache.report_statistics()

        return latents

    def _process_single_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: Dict[str, List[torch.Tensor]],
        step_idx: int,
        total_steps: int,
        guidance_scale: float = 1.0,
        teacache_streams: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Process a single denoising step.

        Args:
            latents: Current latent tensor
            timestep: Current timestep tensor
            text_embeds: Text embeddings
            step_idx: Current step index
            total_steps: Total number of steps
            guidance_scale: Classifier-free guidance scale
            teacache_streams: TeaCache prediction streams

        Returns:
            Model output tensor
        """
        # Current percentage through sampling
        step_percentage = step_idx / total_steps

        # Calculate sequence length (used by transformer model)
        seq_len = latents.shape[2] * latents.shape[3] * latents.shape[4]

        # Check TeaCache first if enabled
        if self.teacache and "cond" in teacache_streams:
            # Get time modulation tensor
            time_embed, time_modulation = self.diffusion_model.time_embedding(timestep)

            # Check if we can skip conditional computation
            cond_should_compute = self.teacache.should_compute(
                teacache_streams["cond"], step_idx, time_modulation
            )

            # Check if we can skip unconditional computation
            uncond_should_compute = True
            if guidance_scale > 1.0 and "uncond" in teacache_streams:
                uncond_should_compute = self.teacache.should_compute(
                    teacache_streams["uncond"], step_idx, time_modulation
                )

            # If we can skip both, apply cached residuals
            if not cond_should_compute and (
                guidance_scale <= 1.0 or not uncond_should_compute
            ):
                # Apply cached residual for conditional
                return self.teacache.apply_cached_residual(
                    teacache_streams["cond"], latents
                )

        # Standard processing without TeaCache or when we can't skip
        with torch.no_grad():
            # Unconditional path (if guidance scale > 1)
            uncond_output = None
            if guidance_scale > 1.0:
                # Store original latents if using TeaCache
                uncond_latents = latents.clone()

                # Run unconditional forward pass
                uncond_output = self.diffusion_model(
                    [uncond_latents],
                    timestep,
                    [text_embeds["negative_prompt_embeds"][0]],
                    seq_len=seq_len,
                    is_uncond=True,
                    current_step_percentage=step_percentage,
                )[0][0]

                # Store for TeaCache if enabled
                if (
                    self.teacache
                    and "uncond" in teacache_streams
                    and uncond_should_compute
                ):
                    self.teacache.store_residual(
                        teacache_streams["uncond"],
                        uncond_latents,
                        uncond_output,
                        time_modulation,
                    )

            # Conditional path
            # Store original latents if using TeaCache
            cond_latents = latents.clone() if self.teacache else latents

            # Run conditional forward pass
            cond_output = self.diffusion_model(
                [cond_latents],
                timestep,
                [text_embeds["prompt_embeds"][0]],
                seq_len=seq_len,
                is_uncond=False,
                current_step_percentage=step_percentage,
            )[0][0]

            # Store for TeaCache if enabled
            if self.teacache and "cond" in teacache_streams and cond_should_compute:
                self.teacache.store_residual(
                    teacache_streams["cond"], cond_latents, cond_output, time_modulation
                )

        # Combine outputs for classifier-free guidance
        if guidance_scale > 1.0 and uncond_output is not None:
            return uncond_output + guidance_scale * (cond_output - uncond_output)
        else:
            return cond_output

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to video frames.

        Args:
            latents: Latent tensor of shape [B, C, T, H, W]

        Returns:
            Decoded video tensor of shape [T, H, W, 3]
        """
        # Use memory manager to handle VAE placement
        video = self.memory_manager.decode_with_tiling(self.vae, latents)

        # Convert to [T, H, W, 3] format with range [0, 1]
        # VAE output is [-1, 1] range
        video = (video + 1) / 2.0
        video = torch.clamp(video, 0.0, 1.0)
        video = video.permute(2, 3, 4, 1).cpu().float()

        logger.debug(
            f"Decoded video: shape={video.shape}, range=[{video.min():.2f}, {video.max():.2f}]"
        )

        return video

    def save_video(
        self,
        video: torch.Tensor,
        output_path: str,
        fps: int = 16,
        quality: int = 8,
        metadata: Optional[Dict] = None,
    ):
        """
        Save a video to a file.

        Args:
            video: Video tensor of shape [T, H, W, 3]
            output_path: Path to save the video
            fps: Frames per second
            quality: Video quality (0-10)
            metadata: Optional metadata to save alongside the video
        """
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy array in correct format for video
        video_np = (video * 255).numpy().astype(np.uint8)

        # Save using appropriate format based on extension
        if output_path.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov"]:
            try:
                import imageio

                logger.info(f"Saving video to {output_path}")
                imageio.mimsave(
                    str(output_path),
                    video_np,
                    fps=fps,
                    quality=quality,
                )
                
                # Save metadata alongside video
                if metadata:
                    import json

                    metadata_path = output_path.with_suffix(".json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"Saved metadata to {metadata_path}")
            except ImportError:
                logger.error("Cannot save video - imageio not installed")
                logger.info("Install with: pip install imageio[ffmpeg]")
        elif output_path.suffix.lower() == ".gif":
            try:
                import imageio

                logger.info(f"Saving GIF to {output_path}")
                imageio.mimsave(str(output_path), video_np, format="GIF", fps=fps)
            except ImportError:
                logger.error("Cannot save GIF - imageio not installed")
                logger.info("Install with: pip install imageio")
        else:
            logger.error(f"Unsupported output format: {output_path.suffix}")
            logger.info("Supported formats: .mp4, .mkv, .avi, .mov, .gif")