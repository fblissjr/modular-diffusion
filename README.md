# modular-diffusion
a tinkerer's attempt at an inference engine for diffusion, focusing on core end-to-end inference pipelines for video models, leveraging well-supported external libraries as much as possible for modularity

## Project Structure

```
modular_diffusion/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── t5_encoder.py          # T5 encoder with multiple backends
│   ├── diffusion_model.py     # DiT model with memory management
│   ├── vae.py                 # VAE with tiling support
├── pipelines/
│   ├── __init__.py
│   ├── wanvideo_pipeline.py   # Main WanVideo pipeline
├── schedulers/
│   ├── __init__.py
│   ├── flow_schedulers.py     # Flow matching schedulers
├── utils/
│   ├── __init__.py
│   ├── logging.py             # Simple structured logging
│   ├── memory.py              # Memory tracking utilities
│   ├── context.py             # Context window utilities
├── cli.py                     # Command-line interface
```
