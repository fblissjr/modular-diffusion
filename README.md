# Modular Diffusion (probably need a better name)

a modular approach to diffusion transformer model inference, designed for tinkerers who want control, visibility, and flexibility.

## Vision & Purpose

modern AI systems are increasingly becoming compound systems built from multiple specialized models working together. most inference engines take a monolithic approach, where it tries to run everything. LLMs, for example, are optimized depending on your hardware and use case - and with LLMs becoming a common part of diffusion models (whether it's for prompt expansion or something else entirely) and modern LLMs having a vision component, it really starts to make more sense to run those where they work best for your use case.

- run the T5 text encoder on a Mac with `mlx-lm`
- run the diffusion model on your super fast CUDA box
- process parts of the end-to-end pipeline in whatever way you want, so long as it fits the inputs and outputs that the models and system expects
- swap components with better implementations as they emerge
- optimize each piece individually for your specific hardware

While I'm starting this project with wanvideo, the principles apply broadly to all compound AI systems combining LLMs, diffusion models, VAEs, and other components. It's likely it will take other twists and turns, but ideally remain constant to being a great way to learn and explore in a way that lets you mix and match components and see what happens

## Diffusion and LLMs are coming together into broader systems, unified

in current SOTA video models, it's not just a single model, but a compound systemn of them working together:

- a diffusion model transformer model (DiT) to turn noise into magic pixie dust
- UMT5 (multilingual T5) text encoder for text prompts, conditioning, embeddings, and cross-attention
- VAE (autoencoder) for latent space encoding/decoding and turning things we can't think about in our heads into pixels
- CLIPVision model for encoding images needed for image-to-video (working in conjunction with the T5)
- ..and often additional LLMs for prompt expansion and targeting your input prompts toward the training dataset

## planned future

- get things to actually **work** and run, which will probably take some time short of a 1:1 port
- ability to run LLMs and multimodal / vision LLMs using `vLLM`, `sglang`, `transformers`, `mlx-lm`, and that's probably it (from me, anyway)
- modular (it's in the name!) component architecture
- guides and notes as I build this (and learning about the complexity of diffusion along the way), as well as explaining design decisions and trade-offs - because they exist everywhere
- an easier way to run those random (what if I did this?) experiments without worrying about what other code is impacting your results

## Acknowledgments

- [Kijai's](https://github.com/kijai) [WanVideoWrapper for ComfyUI](kijai/ComfyUI) 🙏
- [WanVideo](https://github.com/Wan-Video/Wan2.1)
