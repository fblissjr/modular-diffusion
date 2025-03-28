# src/pipelines/wanvideo/pipeline.py
import torch
import numpy as np
import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
from tqdm import tqdm

from src.core.component import Component
from src.core.factory import ComponentFactory
from src.core.dtype import DtypeManager
from src.models.text_encoders import TextEncoder
from src.models.diffusion import DiffusionModel
from src.models.vae import VAE
from src.schedulers.base import Scheduler
from src.pipelines.base import Pipeline
from PIL import Image

logger = logging.getLogger(__name__)

class WanVideoPipeline(Pipeline):
    """
    Pipeline for text-to-video generation with WanVideo.
    
    This orchestrates the text encoder, diffusion model, VAE, and
    scheduler for video generation, similar to how LLM pipelines
    handle text generation workflows.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WanVideo pipeline.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__(config)
        
        # Initialize dtype manager
        self.dtype_manager = DtypeManager(
            compute_dtype=self.dtype,
            param_dtype=self.dtype,
            keep_modules_fp32=config.get("keep_fp32", ["norm", "ln", "layernorm", "rmsnorm"])
        )
        
        # Initialize components
        self._init_components()
        
        # Set up generation parameters
        self._init_generation_config()
        
        logger.info(f"Initialized WanVideoPipeline with dtype={self.dtype}, device={self.device}")
        
    def _init_components(s):
        """Init pipeline components."""
        # Track memory
        m=lambda:torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0
        m0=m()
        
        # Load text encoder
        logger.info("Loading text encoder...")
        s.text_encoder=s._load_text_encoder()
        m1=m()
        
        # Load diffusion model
        logger.info("Loading diffusion model...")
        s.diffusion_model=s._load_diffusion_model()
        m2=m()
        
        # Load VAE
        logger.info("Loading VAE...")
        s.vae=s._load_vae()
        m3=m()
        
        # Load scheduler
        logger.info("Loading scheduler...")
        s.scheduler=s._load_scheduler()
        
        # Log memory stats
        if torch.cuda.is_available():
            logger.info(f"Memory: init={m0:.1f}GB → T5={m1:.1f}GB → DiT={m2:.1f}GB → VAE={m3:.1f}GB")
        
    def _init_generation_config(self):
        """Initialize generation config from defaults."""
        # Set default generation parameters
        self.generation_config = {
            "height": self.config.get("height", 320),
            "width": self.config.get("width", 576),
            "num_frames": self.config.get("num_frames", 16),
            "num_inference_steps": self.config.get("num_inference_steps", 30),
            "guidance_scale": self.config.get("guidance_scale", 7.5),
            "negative_prompt": self.config.get("negative_prompt", "worst quality, blurry"),
            "fps": self.config.get("fps", 8)
        }
        
    def _load_text_encoder(self) -> TextEncoder:
        """
        Load text encoder component.
        
        Returns:
            Text encoder component
        """
        # Get text encoder config
        text_encoder_config = self.config.get("text_encoder", {})
        
        # Get model type and path
        model_type = text_encoder_config.get("type", "T5TextEncoder")
        model_path = text_encoder_config.get("model_path")
        
        # Default to a subfolder of model_path if not specified
        if not model_path and "model_path" in self.config:
            base_path = Path(self.config["model_path"])
            if base_path.is_dir():
                # Check for text_encoder subfolder
                text_encoder_path = base_path.parent / "text_encoder" / "umt5-xxl"
                if text_encoder_path.exists():
                    model_path = str(text_encoder_path)
                    logger.info(f"Using text encoder path: {model_path}")
        
        # Update config
        text_encoder_config["model_path"] = model_path
        text_encoder_config["device"] = self.device
        text_encoder_config["dtype"] = self.dtype
        
        # Create text encoder
        logger.info(f"Creating text encoder of type {model_type}")
        text_encoder = ComponentFactory.create_component(
            model_type,
            text_encoder_config,
            TextEncoder
        )
        
        return text_encoder
    
    def _load_diffusion_model(self) -> DiffusionModel:
        """
        Load diffusion model component.
        
        Returns:
            Diffusion model component
        """
        # Get diffusion model config
        diffusion_config = self.config.get("diffusion_model", {})
        
        # Get model type and path
        model_type = diffusion_config.get("type", "WanDiT")
        model_path = diffusion_config.get("model_path", self.config.get("model_path"))
        
        # Default to a subfolder of model_path if specified
        if not model_path and "model_path" in self.config:
            base_path = Path(self.config["model_path"])
            if base_path.is_dir():
                # Check for diffusion_model subfolder
                diffusion_path = base_path / "dit" / "Wan2.1-T2V-1.3B"
                if diffusion_path.exists():
                    model_path = str(diffusion_path)
                    logger.info(f"Using diffusion model path: {model_path}")
        
        # Update config
        diffusion_config["model_path"] = model_path
        diffusion_config["device"] = self.device
        diffusion_config["dtype"] = self.dtype
        diffusion_config["in_channels"] = self.config.get("latent_channels", 16)
        diffusion_config["out_channels"] = self.config.get("latent_channels", 16)
        
        # Create diffusion model
        logger.info(f"Creating diffusion model of type {model_type}")
        diffusion_model = ComponentFactory.create_component(
            model_type,
            diffusion_config,
            DiffusionModel
        )
        
        # Apply dtype management
        self.dtype_manager.apply_to_model(diffusion_model)
        
        return diffusion_model
    
    def _load_vae(self) -> VAE:
        """
        Load VAE component.
        
        Returns:
            VAE component
        """
        # Get VAE config
        vae_config = self.config.get("vae", {})
        
        # Get model type and path
        model_type = vae_config.get("type", "WanVAEAdapter")
        model_path = vae_config.get("model_path")
        
        # Default to a subfolder of model_path if not specified
        if not model_path and "model_path" in self.config:
            base_path = Path(self.config["model_path"])
            if base_path.is_dir():
                # Check for vae subfolder
                vae_path = base_path.parent / "vae" / "WanVideo" / "Wan2_1_VAE_bf16.safetensors"
                if vae_path.exists():
                    model_path = str(vae_path)
                    logger.info(f"Using VAE path: {model_path}")
        
        # Update config
        vae_config["model_path"] = model_path
        vae_config["device"] = vae_config.get("device", self.device)
        # VAEs are often kept in fp32 for stability
        vae_config["dtype"] = vae_config.get("dtype", torch.float32)
        vae_config["z_dim"] = vae_config.get("z_dim", 16)
        
        # Create VAE
        logger.info(f"Creating VAE of type {model_type}")
        vae = ComponentFactory.create_component(
            model_type,
            vae_config,
            VAE
        )
        
        return vae
    
    def _load_scheduler(self) -> Scheduler:
        """
        Load scheduler component.
        
        Returns:
            Scheduler component
        """
        # Get scheduler config
        scheduler_config = self.config.get("scheduler", {})
        
        # Get scheduler type
        scheduler_type = scheduler_config.get("type", "FlowUniPCScheduler")
        
        # Create scheduler
        logger.info(f"Creating scheduler of type {scheduler_type}")
        scheduler = ComponentFactory.create_component(
            scheduler_type,
            scheduler_config,
            Scheduler
        )
        
        return scheduler
    
    def __call__(self,**kw):
        """
        Generate video from prompt.
        
        Kwargs:
            prompt: Text prompt for generation
            negative_prompt: Optional negative prompt
            height: Video height (must be multiple of 8)
            width: Video width (must be multiple of 8) 
            num_frames: Number of frames to generate
            num_inference_steps: Number of diffusion steps
            guidance_scale: Scale for classifier-free guidance
            generator: Random number generator
            output_type: Output format (pil, pt, np, latent)
            
        Returns:
            Dictionary with generated video
        """
        # Start timer
        t0=time.time()

        # Get generation parameters (update defaults with kwargs)
        p = kw.pop("prompt")
        if not p:
            raise ValueError("prompt is required")

        # Get parameters with defaults from config
        n_p = kw.pop("negative_prompt", self.generation_config["negative_prompt"])
        h = kw.pop("height", self.generation_config["height"]) - (
            kw.pop("height", self.generation_config["height"]) % 8
        )
        w = kw.pop("width", self.generation_config["width"]) - (
            kw.pop("width", self.generation_config["width"]) % 8
        )
        nf = kw.pop("num_frames", self.generation_config["num_frames"])
        ns = kw.pop(
            "num_inference_steps", self.generation_config["num_inference_steps"]
        )
        gs = kw.pop("guidance_scale", self.generation_config["guidance_scale"])
        g = kw.pop("generator", None)
        ot = kw.pop("output_type", "pil")

        # # Ensure dimensions are divisible by 8
        # height = height - height % 8
        # width = width - width % 8

        # Encode text
        logger.info(f"Encoding: '{p}'")
        emb = self.text_encoder.encode(
            prompt=[p], negative_prompt=[n_p] if n_p else None
        )

        # Setup amd scale latents
        lh, lw = h // 8, w // 8
        logger.info(f"Gen latents: f={nf}, h={lh}, w={lw}")
        z = torch.randn(
            (self.diffusion_model.in_channels, nf, lh, lw),
            generator=g,
            device=self.device,
            dtype=self.dtype,
        )
        zs = [z * self.scheduler.init_noise_sigma]

        # Setup steps
        self.scheduler.set_timesteps(ns, device=self.device)

        ts=self.scheduler.timesteps
        # update to set timestep dtype from the start:
        ts = ts.to(device=self.device, dtype=self.dtype)
        
        # # Get latent size (dividing dimensions by VAE stride)
        # latent_height = height // 8
        # latent_width = width // 8
        
        # Generate initial noise
        logger.info(f"Generating latents: frames={nf}, height={lh}, width={lw}")

        # Diffusion process
        logger.info(f"Running {ns} steps")
        for t in tqdm(ts,desc="Generating"):
            with torch.no_grad():
                if gs>1.0:
                    # CFG: negative prompt
                    u_out,_=self.diffusion_model(zs,t,[emb["negative_prompt_embeds"][0]])
                    # Positive prompt
                    c_out,_=self.diffusion_model(zs,t,[emb["prompt_embeds"][0]])
                    # Combine
                    noise=u_out[0]+gs*(c_out[0]-u_out[0])
                else:
                    c_out,_=self.diffusion_model(zs,t,[emb["prompt_embeds"][0]])
                    noise=c_out[0]
            
            # Update latents
            z=self.scheduler.step(noise,t,zs[0])["prev_sample"]
            zs=[z]

        # Decode
        logger.info("Decoding")
        with torch.no_grad():
            v=self.vae.decode(zs)[0]
        v=(v/2+0.5).clamp(0,1)

        # Format output
        if ot=="latent":o=zs
        elif ot=="pt":o=v
        # Convert to numpy array
        elif ot=="np":o=v.cpu().permute(1,2,3,0).numpy()
        # Convert to PIL
        elif ot=="pil":
            vn=(v*255).round().clamp(0,255).cpu().permute(1,2,3,0).numpy().astype(np.uint8)
            o=[Image.fromarray(f) for f in vn]
        else:raise ValueError(f"Unsupported output type: {ot}")
        
        dt=time.time()-t0; fps=nf/dt
        logger.info(f"Done in {dt:.2f}s ({fps:.2f} fps)")
        return {"video":o,"latents":zs,"generation_time":dt,"fps":fps}
        
        # Clamp and normalize
        #video = (video / 2 + 0.5).clamp(0, 1)

        # # Process output based on requested type
        # if output_type == "latent":
        #     output = latents
        # elif output_type == "pt":
        #     output = video
        # elif output_type == "np":
        #     # Convert to numpy array
        #     video_np = video.cpu().permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
        #     output = video_np
        # elif output_type == "pil":
        #     # Convert to PIL images
        #     video_np = (video * 255).round().clamp(0, 255).cpu().permute(1, 2, 3, 0).numpy().astype(np.uint8)
        #     # Convert to PIL
        #     from PIL import Image
        #     frames = [Image.fromarray(frame) for frame in video_np]
        #     output = frames
        # else:
        #     raise ValueError(f"Unsupported output type: {output_type}")
    
    def save_output(self, output: Any, path: str, **kwargs):
        """
        Save pipeline output to file.
        
        Args:
            output: Pipeline output to save
            path: Output file path
            **kwargs: Additional save parameters
                - fps: Frames per second (default: 8)
        """
        # Get output directory
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Get FPS
        fps = kwargs.get("fps", self.generation_config.get("fps", 8))
        
        # Save based on output type
        if isinstance(output, torch.Tensor):
            # Convert to numpy first
            if output.ndim == 5:  # [B, C, T, H, W]
                video_np = output[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
                video_np = (video_np * 255).round().clip(0, 255).astype(np.uint8)
            elif output.ndim == 4:  # [C, T, H, W]
                video_np = output.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
                video_np = (video_np * 255).round().clip(0, 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported tensor shape: {output.shape}")
                
            # Save using imageio
            import imageio
            imageio.mimsave(path, video_np, fps=fps)
            
        elif isinstance(output, np.ndarray):
            # Convert if needed
            if output.dtype != np.uint8:
                output = (output * 255).round().clip(0, 255).astype(np.uint8)
                
            # Save using imageio
            import imageio
            imageio.mimsave(path, output, fps=fps)
            
        elif isinstance(output, list) and all(hasattr(frame, "save") for frame in output):
            # List of PIL images
            # Save using imageio
            import imageio
            imageio.mimsave(path, output, fps=fps)
            
        else:
            raise ValueError(f"Unsupported output type for saving: {type(output)}")
            
        logger.info(f"Saved output to {path}")
    
    def to(self, device: Optional[torch.device] = None, 
           dtype: Optional[torch.dtype] = None) -> "WanVideoPipeline":
        """
        Move pipeline to specified device and dtype.
        
        Args:
            device: Target device
            dtype: Target dtype
            
        Returns:
            Self for chaining
        """
        super().to(device, dtype)
        
        # Update components
        if device is not None:
            self.text_encoder.to(device=device)
            self.diffusion_model.to(device=device)
            # VAE might stay on its own device if specified in config
            if not self.config.get("vae", {}).get("device"):
                self.vae.to(device=device)
                
        if dtype is not None:
            self.text_encoder.to(dtype=dtype)
            self.diffusion_model.to(dtype=dtype)
            # VAE often stays in fp32 for stability
            if not self.config.get("vae", {}).get("dtype"):
                self.vae.to(dtype=dtype)
            
        return self