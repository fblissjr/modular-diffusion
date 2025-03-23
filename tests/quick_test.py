# tests/quick_test.py
import torch,logging,os,sys,argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipelines.wanvideo.pipeline import WanVideoPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

# Memory tracking helper
def log_mem():
    if torch.cuda.is_available():
        a=torch.cuda.memory_allocated()/1024**3
        r=torch.cuda.memory_reserved()/1024**3
        logger.info(f"GPU memory: {a:.2f}GB allocated, {r:.2f}GB reserved")

def main():
    # Parse arguments
    p=argparse.ArgumentParser(description="Test WanVideo pipeline")
    p.add_argument("--t5-cpu",action="store_true",help="Keep T5 on CPU")
    p.add_argument("--t5-on-demand",action="store_true",help="Load T5 only when needed")
    p.add_argument("--dummy-encoder",action="store_true",help="Use dummy encoder for debugging")
    p.add_argument("--steps",type=int,default=10,help="Inference steps")
    p.add_argument("--output",type=str,default="test_output.mp4",help="Output file")
    p.add_argument("--vae-cpu",action="store_true",help="Keep VAE on CPU")
    p.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu",help="Device (cuda[:n],cpu)")
    p.add_argument("--dtype",type=str,default="bf16",choices=["fp32","fp16","bf16","fp8"],help="Precision")
    p.add_argument("--t5-dtype",type=str,help="T5 precision (default: same as --dtype)")
    p.add_argument("--dit-dtype",type=str,help="DiT precision (default: same as --dtype)")
    p.add_argument("--vae-dtype",type=str,default="fp32",help="VAE precision")

    args=p.parse_args()
    
    try:
        logger.info("Testing WanVideo pipeline...")
        
        # Set up configuration
        cfg={
            "model_path":"./models",  # Base path
            "device":args.device,
            "dtype":args.dtype,
            "text_encoder":{
                "device":"cpu" if args.t5_cpu else args.device,
                "dtype":args.t5_dtype or args.dtype,
                "type":"T5TextEncoder",
                "model_path":"./models/text_encoder/umt5-xxl/umt5-xxl-enc-bf16.safetensors",
                "tokenizer":"google/umt5-xxl",
                "cpu_offload":args.t5_cpu,
                "load_on_demand":args.t5_on_demand
            },
            "diffusion_model":{
                "device":args.device,
                "dtype":args.dit_dtype or args.dtype,
                "type":"WanDiT",
                "model_path":"./models/dit/Wan2.1-T2V-1.3B"
            },
            "vae":{
                "type":"WanVAEAdapter",
                "model_path":"./models/vae/WanVideo/Wan2_1_VAE_bf16.safetensors",
                # fyi - don not set device to None - either specify "cpu" or omit
                "device":"cpu" if args.vae_cpu else args.device,
                "dtype":args.vae_dtype,
            },
            "device":"cuda" if torch.cuda.is_available() else "cpu",
            "dtype":"bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32",
            # Minimal generation config for test
            "height":320,
            "width":576,
            "num_frames":16,
            "num_inference_steps":args.steps,
            "t5_cpu":args.t5_cpu  # Add top-level flag for pipeline
        }
        
        # Log initial memory
        log_mem()
        
        # Create pipeline
        logger.info("Initializing pipeline...")
        pl=WanVideoPipeline(cfg)
        
        # Replace with dummy encoder for fast testing if needed
        if args.dummy_encoder:
            from src.models.text_encoders.base import TextEncoder
            class DummyEncoder(TextEncoder):
                def encode(self,prompt,negative_prompt=None):
                    e=torch.randn(1,512,4096,device=self.device,dtype=self.dtype)
                    return {"prompt_embeds":[e],"negative_prompt_embeds":[e]} if negative_prompt else {"prompt_embeds":[e]}
                def to(self,device=None,dtype=None):return self
            pl.text_encoder=DummyEncoder({"device":pl.device,"dtype":pl.dtype})
            logger.info("Using dummy encoder for testing")
        
        logger.info("Pipeline initialized!")
        
        # Log memory after loading
        log_mem()
        
        # Run inference
        logger.info("Running inference...")
        out=pl(
            prompt="A beautiful sunset over the ocean",
            negative_prompt="worst quality, blurry"
        )
        logger.info("Inference complete!")
        
        # Log final memory
        log_mem()
        
        # Save output
        logger.info(f"Saving to {args.output}...")
        pl.save_output(out["video"],args.output,fps=8)
        logger.info(f"Done! Output saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    return True

if __name__=="__main__":
    sys.exit(0 if main() else 1)