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

Evaluate Stage 1 flags against VISPR annotation JSON files:

```bash
python -m vlm_captioning.evaluate_flags metadata/vispr_predictions.csv data/vispr/annotations
```

The prediction CSV must contain `file` and `PRIVACY_FLAG` columns. Relative image paths are matched to annotation JSON files under the annotation directory.

## Pipeline code

The `pipeline/` directory includes dataset creation, edit dataset loaders, batch editors, the Unsafe2Safe training wrapper, Safe Attention, and CLIP or face-similarity evaluation helpers.

The diffusion code expects a vanilla InstructPix2Pix checkout outside this repository, plus the compatible configuration files and checkpoints. The unchanged base diffusion repository is not vendored here.

Run batch inference with `./pipeline/scripts/run_unsafe2safe.sh INPUT_CSV OUTPUT_DIR CHECKPOINT IMAGE_ROOT`.

Start Stage 2 training from a compatible external diffusion checkout:

```bash
./pipeline/scripts/train_unsafe2safe.sh /path/to/instruct-pix2pix pipeline/configs/train_unsafe2safe.yaml LOG_DIR GPU_IDS
```

The training wrapper imports a vanilla external InstructPix2Pix checkout and patches its UNet in memory; it does not require modifying that checkout. See [`pipeline/README.md`](pipeline/README.md) and [`pipeline/configs/train_unsafe2safe.yaml`](pipeline/configs/train_unsafe2safe.yaml) for the canonical command and data contract.

For the OminiControl experiment, install a separate OminiControl checkout and
run the Unsafe2Safe adapter in [`pipeline/ominicontrol/`](pipeline/ominicontrol/README.md).
The upstream OminiControl source is not copied into this repository.

Launch the optional prompt demo locally with `datasets`, `gradio`, and `openai` installed:

```bash
python pipeline/prompt_app.py --openai-api-key "$OPENAI_API_KEY" --openai-model MODEL_NAME
```

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
