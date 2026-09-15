# Unsafe2Safe

Unsafe2Safe creates privacy-preserving image datasets while keeping the visual content useful for downstream tasks.

This repository contains the Stage 1 captioning tools, dataset preparation scripts, project-specific editing code, evaluation helpers, and Safe Attention layers.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build that matches the target CPU or CUDA platform when the default pip resolution is not suitable.

## Repository layout

```text
prompts/                 Prompt templates.
vlm_captioning/         Stage 1 captioning and parsing tools.
pipeline/                Dataset, editing, training, and evaluation code.
```

The code is released in rough research form. Stage 1 is the most documented part. The project-specific Stage 2 files preserve the original workflow and still require compatible model checkpoints and configuration files.

## Stage 1

Place images under `data/` and metadata under `metadata/`. The default configuration expects a CSV with a `file` column whose values are relative to the image directory.

Example layout:

```text
data/mscoco/<images...>
metadata/mscoco.csv
prompts/intern_image_captioning.txt
outputs/mscoco/generate_captions/
```

Generate captions:

```bash
python vlm_captioning/run_stage1.py --config vlm_captioning/configs/stage1.yaml --purpose generate_captions --dataset mscoco
```

Run pairwise anonymization evaluation:

```bash
python vlm_captioning/run_stage1.py --config vlm_captioning/configs/eval.yaml --purpose compare_anonymization --dataset mscoco
```

Parse a structured response:

```python
from vlm_captioning.output_parser import parse_structured_output

parsed = parse_structured_output(model_response)
```

Assemble generated caption files:

```bash
python -m vlm_captioning.collect_captions --captions-dir outputs/mscoco/generate_captions --metadata metadata/mscoco.csv --output metadata/mscoco_with_captions.csv
```

Add `--parse-structured` to include privacy flags and caption sections as separate columns.

## Pipeline code

The `pipeline/` directory includes dataset creation, edit dataset loaders, batch editors, the Unsafe2Safe training wrapper, Safe Attention, and CLIP or face-similarity evaluation helpers.

The diffusion code expects a compatible Stable Diffusion or InstructPix2Pix checkout at `stable_diffusion/`, plus the research configuration files and checkpoints. The unchanged base diffusion repository is not vendored here.

## Dataset and links

The public dataset is available on [Hugging Face](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe). Project links: [project page](https://see-ai-lab.github.io/unsafe2safe/) and [paper](https://arxiv.org/abs/2603.28605).

Keep downloaded datasets, checkpoints, generated images, and experiment outputs outside version control.

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
