# Pipeline code

This directory contains the paper-owned dataset, editing, evaluation, and Safe Attention code.

The diffusion entry points expect a compatible Stable Diffusion or InstructPix2Pix checkout at `stable_diffusion/`, along with the research configuration files and checkpoints. The base diffusion repository is intentionally not vendored here.

`safe_attention.py` contains the project-specific private-caption filtering and public-caption cross-attention layers. The surrounding editor scripts preserve the original research workflow and are released in rough form for later cleanup.

The dataset and metrics helpers can be used independently with local image and metadata paths.
