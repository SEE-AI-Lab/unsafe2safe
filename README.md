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
> Public resources currently available: the [arXiv paper](https://arxiv.org/abs/2603.28605), the [project page](https://see-ai-lab.github.io/unsafe2safe/), the [Hugging Face dataset](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe), and the rough research implementation in this repository.

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
- This repository includes Stage 1 captioning and evaluation utilities, dataset preparation scripts, diffusion editor entry points, evaluation code, and the Safe Attention layers used in the project.

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
pipeline/
  dataset_creation/
  metrics/
  edit_cli.py
  edit_privacy.py
  edit_df.py
  edit_unsafe2safe_df.py
  safe_attention.py
  unsafe2safe_dataset.py
```

The public code is a rough release of the paper implementation. Stage 1 is the most documented part. The `pipeline/` directory contains the existing dataset, editing, evaluation, and Safe Attention code; its diffusion entry points still expect the research checkpoints and configuration files used by the authors.

The prompt templates used by the released configs are included in `prompts/`. This includes the Stage 1 generation prompts and the paper-aligned evaluation templates for custom privacy flagging, text extraction, demographic analysis, and pairwise anonymization scoring. Experimental notebooks, generated datasets, and large external dependencies remain outside this public tree for now.

## Evaluation Protocol
We report four metric groups in the paper:
- **Quality**: realism and semantic alignment
- **Cheating**: unintended copying from source images
- **Privacy**: leakage reduction and demographic anonymization behavior
- **Utility**: downstream task performance after training on anonymized data

Detailed metric definitions and results are available in the [paper](https://arxiv.org/abs/2603.28605) and on the [project page](https://see-ai-lab.github.io/unsafe2safe/).

## Quick Start
This is a rough research release rather than a fully packaged end-to-end training repository. The published code currently centers on the documented Stage 1 runner, while the Stage 2 scripts expose the original training and inference paths for later cleanup.

Before running:
- Set up a Python environment and install the Stage 1 dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Install a PyTorch build appropriate for your CPU or CUDA platform if the default pip resolution is not suitable.
- Place dataset images under `data/` and provide metadata under `metadata/`. The default COCO captioning profile expects `metadata/mscoco.csv` with a `file` column whose values are relative to `data/mscoco`.

Expected layout for the default captioning run:
```text
data/mscoco/<images...>
metadata/mscoco.csv
prompts/intern_image_captioning.txt
outputs/mscoco/generate_captions/<generated JSON files>
```

The configs use relative paths and can be adapted for other datasets by changing the dataset profile or passing `--dataset`.

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

Structured responses can be parsed without model-specific dependencies:
```python
from vlm_captioning.output_parser import parse_structured_output

parsed = parse_structured_output(model_response)
public_caption = parsed.get("PUBLIC_CAPTION", "")
```

To assemble generated JSON files for later training or evaluation, use the notebook-derived collector:
```bash
python -m vlm_captioning.collect_captions \
  --captions-dir outputs/mscoco/generate_captions \
  --metadata metadata/mscoco.csv \
  --output metadata/mscoco_with_captions.csv
```

Use a `.jsonl` output for streaming records, or add `--parse-structured` to include privacy flags and caption sections as separate fields.

## Dataset
The released dataset is hosted on [Hugging Face](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe). Please refer to the dataset card for the public data description, access details, and updates.

## Follow-up cleanup
- Add a concise Stage 2 reproduction guide with checkpoint and configuration details.
- Separate optional demo and download dependencies from the core installation list.
- Remove duplicated helpers and debug output from the batch editor scripts.
- Add the remaining external-repository integrations through their own forks when ready.

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
