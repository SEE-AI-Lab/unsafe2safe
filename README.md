# Unsafe2Safe: Controllable Image Anonymization for Downstream Utility

Official repository for **Unsafe2Safe: Controllable Image Anonymization for Downstream Utility**.

[![Paper](https://img.shields.io/badge/arXiv-2603.28605-b31b1b.svg)](https://arxiv.org/abs/2603.28605)
[![Project Page](https://img.shields.io/badge/Project%20Page-online-0a7ea4.svg)](https://see-ai-lab.github.io/unsafe2safe/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d.svg)](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe)
![Demo](https://img.shields.io/badge/Demo-coming%20soon-lightgrey.svg)
![Model Weights](https://img.shields.io/badge/Model%20Weights-coming%20soon-lightgrey.svg)

> [!IMPORTANT]
> Unsafe2Safe was accepted at CVPR 2026 (Highlight).
>
> Public resources currently available: the [arXiv paper](https://arxiv.org/abs/2603.28605), the [project page](https://see-ai-lab.github.io/unsafe2safe/), the [Hugging Face dataset](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe), and the initial Stage 1 captioning code in this repository.

## Overview
Unsafe2Safe is a two-stage pipeline for privacy-preserving image anonymization with downstream utility:

1. **Stage 1: Privacy-aware text prior generation**
- Detect privacy risk
- Produce private and public captions
- Generate structured edit instructions

2. **Stage 2: Safe image generation**
- Edit unsafe images using diffusion-based editors
- Preserve non-sensitive semantics and scene structure
- Improve privacy while maintaining downstream utility

## What Has Been Released
- The paper is available on [arXiv](https://arxiv.org/abs/2603.28605).
- The project website is live at [see-ai-lab.github.io/unsafe2safe](https://see-ai-lab.github.io/unsafe2safe/).
- The dataset is public on [Hugging Face](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe).
- This repository includes the initial Stage 1 vision-language captioning and evaluation utilities used in the project.

## Current Repository Contents
```text
README.md
vlm_captioning/
  run_stage1.py
  internvl_common.py
  qwen_common.py
  configs/
    stage1.yaml
    eval.yaml
```

The current public code release focuses on the Stage 1 pipeline for privacy-aware caption generation, privacy flagging, edit-instruction generation, and pairwise anonymization evaluation.

## Evaluation Protocol
We report four metric groups in the paper:
- **Quality**: realism and semantic alignment
- **Cheating**: unintended copying from source images
- **Privacy**: leakage reduction and demographic anonymization behavior
- **Utility**: downstream task performance after training on anonymized data

Detailed metric definitions and results are available in the [paper](https://arxiv.org/abs/2603.28605) and on the [project page](https://see-ai-lab.github.io/unsafe2safe/).

## Quick Start
This initial release is not yet a fully packaged end-to-end training repository. The published code currently centers on the Stage 1 runner in `vlm_captioning/`.

Before running:
- Set up a Python environment with the dependencies required for InternVL or Qwen inference together with `pandas`, `PyYAML`, and `tqdm`.
- Update the dataset, cache, and output paths in `vlm_captioning/configs/stage1.yaml` and `vlm_captioning/configs/eval.yaml` to match your local environment.

Example Stage 1 run:
```bash
python vlm_captioning/run_stage1.py \
  --config vlm_captioning/configs/stage1.yaml \
  --purpose generate_captions \
  --dataset mscoco
```

Example pairwise evaluation run:
```bash
python vlm_captioning/run_stage1.py \
  --config vlm_captioning/configs/eval.yaml \
  --purpose compare_anonymization \
  --dataset mscoco
```

The YAML configs define additional purpose profiles such as privacy flag generation and edit-instruction generation.

## Dataset
The released dataset is hosted on [Hugging Face](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe). Please refer to the dataset card for the public data description, access details, and updates.

## Coming Soon
- Pretrained checkpoints and model weights
- Stage 2 diffusion editor training and inference code
- A cleaner end-to-end reproduction guide
- Additional documentation and public demo materials

## Citation
```bibtex
@misc{dinh2026unsafe2safe,
  title={Unsafe2Safe: Controllable Image Anonymization for Downstream Utility},
  author={Mih Dinh and SouYoung Jin},
  year={2026},
  eprint={2603.28605},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2603.28605},
  url={https://arxiv.org/abs/2603.28605}
}
```

## Acknowledgements
Unsafe2Safe builds on recent advances in vision-language modeling and diffusion-based image editing. Please see the [paper](https://arxiv.org/abs/2603.28605) and [project page](https://see-ai-lab.github.io/unsafe2safe/) for additional context.
