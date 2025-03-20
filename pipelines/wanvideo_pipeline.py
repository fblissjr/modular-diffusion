"""
WanVideo Inference Pipeline

This module provides the main pipeline for running inference with
WanVideo models. It coordinates the text encoder, diffusion model,
and VAE to generate videos from text prompts.

The pipeline implements the full diffusion process:
1. Convert text to embeddings
2. Initialize noise
3. Iteratively denoise through the diffusion model
4. Decode the final latents to a video
"""

import os
import torch
import numpy as np
import random
import structlog
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Import our model components
from ..models.t5_encoder import T5TextEncoder
# Note: We'll implement these next, but referencing them here for structure
# from ..models.diffusion_model import WanVideoDiT
# from ..models.vae import WanVideoVAE
from .schedulers.flow_schedulers import FlowUniPCScheduler
import imageio
from ..utils.memory import MemoryConfig, MemoryTracker, DeviceManager
import math

logger = structlog.get_logger()

@dataclass
class WanVideoPipelineOutput:
    """
    Output from the WanVideo pipeline.
    
    Contains the generated video and optionally the latent
    representations and generation metadata.
    """
    video: List[torch.Tensor]
    fps: int = 16
    latents: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None

class WanVideoPipeline:
    """
    Pipeline for generating videos using WanVideo models.
    
    This pipeline coordinates the different components (text encoder,
    diffusion model, VAE) to implement the full generation process.
    It handles memory management, context windowing, and other
    optimizations.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        offload_blocks: int = 0,
        t5_on_cpu: bool = False,
        attention_mode: str = "sdpa",
        use_compile: bool = False,
        compile_options: Optional[Dict] = None,
    ):
        """
        Initialize the WanVideo pipeline.
        
        Args:
            model_path: Path to the model directory
            device: Device to run inference on ("cuda" or "cpu")
            dtype: Data type for model weights and computation
            offload_blocks: Number of transformer blocks to offload to CPU
            t5_on_cpu: Whether to keep the T5 text encoder on CPU
            attention_mode: Type of attention implementation to use
            use_compile: Whether to use torch.compile
            compile_options: Options for torch.compile
        """
        self.logger = logger.bind(component="WanVideoPipeline")
        self.device = torch.device(device)
        self.dtype = dtype
        
        self.logger.info("Initializing WanVideo pipeline", 
                        model_path=model_path,
                        device=str(device),
                        offload_blocks=offload_blocks)
        
        # Determine the model type (t2v or i2v)
        self.model_type = self._determine_model_type(model_path)
        
        # Load the text encoder
        self.text_encoder = T5TextEncoder(
            model_path=model_path,
            device=device,
            dtype=dtype,
            use_cpu=t5_on_cpu
        )
        
        # TODO: Load the diffusion model
        # self.diffusion_model = WanVideoDiT(...)
        
        # TODO: Load the VAE
        # self.vae = WanVideoVAE(...)
        
        # Create the scheduler
        self.scheduler = FlowUniPCScheduler(
            prediction_type="flow_prediction",
            shift=5.0  # Default shift parameter
        )
        
        self.logger.info("WanVideo pipeline initialized",
                        model_type=self.model_type)
    
    def _determine_model_type(self, model_path: str) -> str:
        """
        Determine whether the model is text-to-video or image-to-video.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Model type ("t2v" or "i2v")
        """
        # Look for indicators in the path
        if "t2v" in model_path.lower() or "text-to-video" in model_path.lower():
            self.logger.info("Detected text-to-video model")
            return "t2v"
        elif "i2v" in model_path.lower() or "image-to-video" in model_path.lower():
            self.logger.info("Detected image-to-video model")
            return "i2v"
        
        # Try to read from config file
        config_path = os.path.join(model_path, "transformer/config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
                if "model_type" in config:
                    model_type = config["model_type"]
                    self.logger.info(f"Found model type in config: {model_type}")
                    return model_type
        
        # Default to t2v if we can't determine
        self.logger.warning("Could not determine model type, defaulting to t2v")
        return "t2v"
    
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: str = "",
        height: int = 480,
        width: int = 832,
        num_frames: int = 25,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        shift: float = 5.0,
        seed: Optional[int] = None,
        use_context_windows: bool = True,
        context_size: int = 81,
        context_stride: int = 4,
        context_overlap: int = 16,
        enable_vae_tiling: bool = True,
        vae_tile_size: Tuple[int, int] = (272, 272),
        vae_tile_stride: Tuple[int, int] = (144, 128),
        output_type: str = "mp4",
        save_path: Optional[str] = None,
        save_latents: bool = False,
        callback: Optional[callable] = None,
    ) -> WanVideoPipelineOutput:
        """
        Generate a video from a text prompt.

        This method implements the full diffusion pipeline:
        1. Encode text prompts to embeddings
        2. Initialize random noise
        3. Gradually denoise with the diffusion model
        4. Decode the final latents with the VAE
        """
        # Set random seed for reproducibility
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        self.logger.info("Using seed", seed=seed)

        # Make sure prompt is a list
        if isinstance(prompt, str):
            prompt = [prompt]

        self.logger.info(
            "Starting video generation",
            prompts=[p[:50] + "..." if len(p) > 50 else p for p in prompt],
            num_frames=num_frames,
            steps=num_inference_steps,
        )

        # 1. Encode the text prompts to embeddings
        self.logger.debug("Encoding text prompts")
        with self.memory_tracker.track_usage("Text encoding"):
            text_embeds = self.text_encoder.encode(
                prompt, negative_prompt=negative_prompt
            )

        # 2. Set up the latent space dimensions
        # VAE downsamples spatially by factor of 8 and temporally by factor of 4
        latent_height = height // 8
        latent_width = width // 8
        latent_frames = (num_frames - 1) // 4 + 1

        # Calculate sequence length for the transformer
        # This depends on the patch size used in the transformer
        patch_size = (
            self.diffusion_model.patch_size
            if hasattr(self.diffusion_model, "patch_size")
            else (1, 2, 2)
        )
        seq_len = math.ceil(
            (latent_height * latent_width)
            / (patch_size[1] * patch_size[2])
            * latent_frames
        )

        self.logger.debug(
            f"Latent dimensions: {latent_frames}x{latent_height}x{latent_width}, sequence length: {seq_len}"
        )

        # 3. Create initial random noise
        # This is the starting point for the diffusion process
        self.logger.debug(f"Initializing noise with seed {seed}")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            (1, 16, latent_frames, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

        # 4. Set up the noise scheduler
        self.logger.debug(f"Setting up scheduler with {num_inference_steps} steps")
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
        timesteps = self.scheduler.timesteps

        # 5. Adjust context parameters for latent space if using context windows
        context_queue = None
        if use_context_windows:
            # Convert from pixel-space to latent-space values
            latent_context_size = (context_size - 1) // 4 + 1
            latent_context_stride = context_stride // 4
            latent_context_overlap = context_overlap // 4

            self.logger.info(
                "Using context windows",
                context_size=latent_context_size,
                stride=latent_context_stride,
                overlap=latent_context_overlap,
            )

            # Import context scheduler
            from ..utils.context import (
                get_context_scheduler,
                uniform_looped,
                uniform_standard,
            )

            context_scheduler = get_context_scheduler("uniform_standard")

        # 6. Run the denoising loop
        # This is the core of the diffusion process
        self.logger.info("Starting denoising process")
        progress_bar = tqdm(
            total=len(timesteps), desc="Generating video", disable=callback is not None
        )

        # Move latents to device and correct dtype
        latents = latents.to(device=self.device, dtype=self.dtype)

        # Store for callback visualization
        latent_history = []

        with self.memory_tracker.track_usage("Denoising"):
            # Loop through each timestep
            for i, t in enumerate(timesteps):
                # Current position in sampling process (0-1)
                current_step_percentage = i / len(timesteps)

                # Get context windows for this step if using windowing
                if use_context_windows:
                    context_windows = context_scheduler(
                        i,
                        num_inference_steps,
                        latent_frames,
                        latent_context_size,
                        latent_context_stride,
                        latent_context_overlap,
                    )

                    # Process by window to save memory
                    noise_pred_combined = torch.zeros_like(latents)
                    weight_combined = torch.zeros(
                        (1, 1, latent_frames, 1, 1),
                        device=self.device,
                        dtype=self.dtype,
                    )

                    # Process each context window
                    for window_idx, window_frames in enumerate(context_windows):
                        # Get the appropriate text embedding based on window position
                        # For multiple prompts, select based on window position
                        prompt_idx = min(
                            int(len(window_frames) / latent_frames * len(prompt)),
                            len(prompt) - 1,
                        )
                        window_context = [text_embeds["prompt_embeds"][prompt_idx]]
                        window_uncond = text_embeds["negative_prompt_embeds"]

                        # Select relevant latent frames for this window
                        window_latents = latents[:, :, window_frames, :, :]

                        # Convert timestep to tensor
                        timestep = torch.tensor([t], device=self.device)

                        # For classifier-free guidance, do two forward passes
                        with torch.no_grad():
                            # Unconditional pass (if using guidance)
                            if guidance_scale > 1.0:
                                uncond_latents = latents[
                                    :, :, window_frames, :, :
                                ].clone()
                                uncond_output = self.diffusion_model(
                                    [uncond_latents],
                                    timestep,
                                    [window_uncond],
                                    seq_len=seq_len,
                                    is_uncond=True,
                                    current_step_percentage=current_step_percentage,
                                    device=self.device,
                                )[0][0]

                            # Conditional pass
                            cond_output = self.diffusion_model(
                                [window_latents],
                                timestep,
                                window_context,
                                seq_len=seq_len,
                                is_uncond=False,
                                current_step_percentage=current_step_percentage,
                                device=self.device,
                            )[0][0]

                        # Combine outputs for classifier-free guidance
                        if guidance_scale > 1.0:
                            model_output = uncond_output + guidance_scale * (
                                cond_output - uncond_output
                            )
                        else:
                            model_output = cond_output

                        # Create blending mask for window
                        # Smooth transition between windows
                        window_mask = self._create_window_mask(
                            model_output,
                            window_frames,
                            latent_frames,
                            latent_context_overlap,
                        )

                        # Add to combined output with weight mask
                        noise_pred_combined[:, :, window_frames, :, :] += (
                            model_output * window_mask
                        )
                        weight_combined[:, :, window_frames, :, :] += window_mask

                    # Normalize by weights to get final prediction
                    noise_pred = noise_pred_combined / (weight_combined + 1e-8)

                else:
                    # No windowing - process all frames at once
                    timestep = torch.tensor([t], device=self.device)

                    # For classifier-free guidance, do two forward passes
                    with torch.no_grad():
                        # Unconditional pass (if using guidance)
                        if guidance_scale > 1.0:
                            uncond_output = self.diffusion_model(
                                [latents.clone()],
                                timestep,
                                [text_embeds["negative_prompt_embeds"]],
                                seq_len=seq_len,
                                is_uncond=True,
                                current_step_percentage=current_step_percentage,
                                device=self.device,
                            )[0][0]

                        # Conditional pass
                        cond_output = self.diffusion_model(
                            [latents],
                            timestep,
                            [text_embeds["prompt_embeds"][0]],
                            seq_len=seq_len,
                            is_uncond=False,
                            current_step_percentage=current_step_percentage,
                            device=self.device,
                        )[0][0]

                    # Combine outputs for classifier-free guidance
                    if guidance_scale > 1.0:
                        noise_pred = uncond_output + guidance_scale * (
                            cond_output - uncond_output
                        )
                    else:
                        noise_pred = cond_output

                # Step with scheduler
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # Store latent for history if requested
                if save_latents and i % max(1, num_inference_steps // 10) == 0:
                    latent_history.append(latents.clone().cpu())

                # Update progress
                if callback is not None:
                    # Convert current latent to image for preview
                    with torch.no_grad():
                        # Just use the first frame for preview to save memory
                        preview_latent = latents[:, :, 0:1, :, :].clone()
                        preview = self._decode_preview(preview_latent)
                    callback(i, len(timesteps), preview)
                else:
                    progress_bar.update(1)

        # Close progress bar
        if not callback:
            progress_bar.close()

        # 7. Decode the latents to pixel space
        self.logger.info("Decoding latents to video frames")
        with self.memory_tracker.track_usage("VAE decoding"):
            video = self._decode_latents(
                latents,
                enable_tiling=enable_vae_tiling,
                tile_size=vae_tile_size,
                tile_stride=vae_tile_stride,
            )

        # 8. Format and save the output
        if save_path:
            self._save_output(
                video,
                save_path,
                output_type,
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "shift": shift,
                },
            )

        return WanVideoPipelineOutput(
            video=[video],
            fps=16,
            latents=latent_history if save_latents else None,
            metadata={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "shift": shift,
                "height": height,
                "width": width,
                "num_frames": num_frames,
            },
        )

    def _create_window_mask(self, tensor, frame_indices, total_frames, overlap):
        """
        Create a blending mask for context windows.

        This creates smooth transitions between windows to avoid seams.

        Args:
            tensor: Tensor to create mask for (gets shape from this)
            frame_indices: List of frame indices in this window
            total_frames: Total number of frames in the video
            overlap: Number of frames to overlap between windows

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
            # Apply to beginning of mask
            mask[:, :, :overlap] = ramp_up

        # Apply right-side blending if not ending at last frame
        if max(frame_indices) < total_frames - 1:
            # Create ramp from 1 to 0 over overlap frames
            ramp_down = torch.linspace(1, 0, overlap, device=tensor.device)
            # Add dimensions to match tensor shape
            ramp_down = ramp_down.view(1, 1, -1, 1, 1)
            # Apply to end of mask
            mask[:, :, -overlap:] = ramp_down

        return mask

    def _decode_latents(
        self, latents, enable_tiling=True, tile_size=(272, 272), tile_stride=(144, 128)
    ):
        """
        Decode latent representations to video frames.

        Args:
            latents: Latent tensor of shape [B, C, T, H, W]
            enable_tiling: Whether to use tiling for memory efficiency
            tile_size: Size of tiles for tiled processing
            tile_stride: Stride between tiles

        Returns:
            Decoded video tensor of shape [T, H, W, 3]
        """
        # Move VAE to device for decoding
        self.vae.to(self.device)

        # Clone and move latents to avoid modifying input
        latents = latents.clone().to(device=self.device, dtype=self.vae.dtype)

        # Scale latents if needed (depends on VAE implementation)
        # For WanVideo VAE, no scaling needed as it's handled internally

        # Decode latents to pixel space
        with torch.no_grad():
            if enable_tiling:
                # Use tiled decoding to save memory
                video = self.vae.tiled_decode(
                    latents, self.device, tile_size=tile_size, tile_stride=tile_stride
                )[0]
            else:
                # Decode directly
                video = self.vae.decode(latents, self.device)[0]

        # Move VAE back to CPU to save memory
        self.vae.to("cpu")

        # Convert to [T, H, W, 3] format with range [0, 1]
        # VAE output is [-1, 1] range
        video = (video + 1) / 2.0
        video = torch.clamp(video, 0.0, 1.0)
        video = video.permute(2, 3, 4, 1).cpu().float()

        return video

    def _decode_preview(self, latents):
        """
        Decode a single frame for preview/callback.

        This is a lightweight version of decode_latents for progress updates.

        Args:
            latents: Latent tensor containing a single frame

        Returns:
            Decoded image for preview
        """
        # Use a context manager to temporarily move VAE to device
        with torch.no_grad():
            # Move VAE to device
            self.vae.to(self.device)

            # Decode just the first frame to save memory
            image = self.vae.decode(latents, self.device)[0]

            # Move VAE back to CPU
            self.vae.to("cpu")

        # Convert to [H, W, 3] format with range [0, 1]
        image = (image + 1) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        image = image[0, :, 0].permute(1, 2, 0).cpu().float()

        return image
    
    def _save_output(
        self,
        video: torch.Tensor,
        save_path: str,
        output_type: str,
        metadata: Dict[str, Any],
    ):
        """
        Save the generated video to a file.
        
        Args:
            video: Video tensor to save
            save_path: Path to save the output file
            output_type: Type of output file ("mp4" or "gif")
            metadata: Metadata about the generation process
        """
        # Convert to numpy array in correct format for video
        video_np = video.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
        video_np = np.clip(video_np * 255, 0, 255).astype(np.uint8)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Save using appropriate format
        if output_type == "mp4":
            try:
                import imageio.v3 as imageio
                imageio.imwrite(
                    save_path, 
                    video_np, 
                    fps=16,
                    quality=7,  # Good quality with reasonable file size
                )
                
                # Save metadata alongside video
                metadata_path = os.path.splitext(save_path)[0] + ".json"
                import json
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info("Saved output", 
                                video_path=save_path,
                                metadata_path=metadata_path)
                
            except ImportError:
                self.logger.error("Cannot save video - imageio not installed")
                self.logger.info("Install with: pip install imageio[ffmpeg]")
        
        elif output_type == "gif":
            try:
                import imageio.v3 as imageio
                imageio.imwrite(
                    save_path,
                    video_np,
                    format="gif",
                    fps=16
                )
                self.logger.info("Saved GIF", path=save_path)
            except ImportError:
                self.logger.error("Cannot save GIF - imageio not installed")
                self.logger.info("Install with: pip install imageio")