# Modular Diffusion (probably need a better name)

a modular approach to diffusion transformer model inference, designed for tinkerers who want control, visibility, and flexibility.

## immediate next steps

- end-to-end inference in a workable state with `./tests/quick_test.py`
- immediately followed by a re-org the code base to be more... modular. i see what i'd do differently now, and now's the time to do it. i've removed `diffusers` completely now and went pure torch because it gives more flexibility downstream.
- focus on getting wanvideo right, but not so focused that i lose track of the real end goals here.
- write tests. lots of them.

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

## project structure

```
modular-diffusion/
├── src/
│   ├── core/                      # Foundation layer
│   │   ├── component.py           # Base component interface
│   │   ├── config.py              # Configuration system
│   │   ├── dtype.py               # Dtype management
│   │   ├── registry.py            # Component registry
│   │   ├── factory.py             # Component factory
│   │   └── telemetry.py           # Logging and tracing (OpenTelemetry)
│   │
│   ├── models/                    # Model implementations
│   │   ├── text_encoders/         # Text encoder implementations
│   │   │   ├── base.py            # Base interface
│   │   │   ├── t5.py              # T5 implementation
│   │   │   └── remote.py          # Remote encoder client
│   │   │
│   │   ├── diffusion/             # Diffusion models
│   │   │   ├── base.py            # Base interface
│   │   │   └── wandit.py          # WanVideo implementation
│   │   │
│   │   └── vae/                   # VAE implementations
│   │       ├── base.py            # Base interface
│   │       └── wanvae.py          # WanVideo VAE
│   │
│   ├── schedulers/                # Pure torch schedulers
│   │   ├── base.py                # Base scheduler interface
│   │   ├── flow_unipc.py          # UniPC implementation
│   │   ├── flow_dpm.py            # DPM++ implementation
│   │   └── euler.py               # Euler implementation
│   │
│   ├── pipelines/                 # Pipeline implementations
│   │   ├── base.py                # Base pipeline interface
│   │   └── wanvideo/              # WanVideo pipeline
│   │       ├── pipeline.py        # Main implementation
│   │       ├── context.py         # Context strategies
│   │       └── cache.py           # Caching implementations
│   │
│   ├── extensions/                # Optional extensions
│   │   ├── memory/                # Memory optimizations
│   │   │   ├── block_swap.py      # Transformer block swapping
│   │   │   └── vae_tiling.py      # VAE tiling optimizations
│   │   │
│   │   ├── optimization/          # Performance optimizations
│   │   │   ├── flash_attn.py      # Flash Attention integration
│   │   │   ├── compile.py         # torch.compile utilities
│   │   │   └── sage_attn.py       # SageAttention integration
│   │   │
│   │   ├── caching/               # Advanced caching strategies
│   │   │   ├── teacache.py        # TeaCache implementation
│   │   │   ├── diffusion_cache.py # Diffusion cache from WanVideo
│   │   │   └── feature_cache.py   # VAE feature caching
│   │   │
│   │   ├── quantization/          # Quantization methods
│   │   │   ├── fp8.py             # FP8 quantization
│   │   │   └── int8.py            # INT8 quantization
│   │   │
│   │   └── experiments/           # Experimental features
│   │       ├── plugin.py          # Plugin system
│   │       └── attention_mod.py   # Attention modification tools
│   │
│   └── utils/                     # General utilities
│       ├── io.py                  # I/O utilities
│       ├── storage.py             # Storage (sqlite, etc.)
│       └── telemetry.py           # Telemetry utilities
│
├── tests/                         # Test framework
│   ├── unit/                      # Unit tests
│   │   ├── models/                # Model tests
│   │   ├── schedulers/            # Scheduler tests
│   │   └── utils/                 # Utility tests
│   │
│   ├── integration/               # Integration tests
│   │   └── pipelines/             # Pipeline tests
│   │
│   └── benchmark/                 # Performance benchmarks
│
├── cli/                           # Command-line interface
├── examples/                      # Example scripts
│   ├── basic.py                   # Basic usage
│   ├── advanced.py                # Advanced customization
│   └── remote_encoder.py          # Remote text encoding
│
└── configs/                       # Configuration templates
    ├── base.json                  # Base configuration
    ├── low_memory.json            # Low memory configuration
    └── high_quality.json          # High quality configuration
```

## Acknowledgments

- [Kijai's](https://github.com/kijai) [WanVideoWrapper for ComfyUI](kijai/ComfyUI) 🙏
- [WanVideo](https://github.com/Wan-Video/Wan2.1)
