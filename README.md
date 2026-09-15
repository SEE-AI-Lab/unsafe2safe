# Unsafe2Safe

[![Paper](https://img.shields.io/badge/arXiv-2603.28605-b31b1b.svg)](https://arxiv.org/abs/2603.28605)
[![Project Page](https://img.shields.io/badge/Project%20Page-online-0a7ea4.svg)](https://see-ai-lab.github.io/unsafe2safe/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d.svg)](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe)

Unsafe2Safe creates privacy-preserving image edits while preserving the useful visual content of the source image.

This repository contains the project-owned captioning, dataset preparation, editing, training adapters, evaluation helpers, and prompt assets used by the project. Large third-party model repositories, checkpoints, datasets, and generated outputs stay outside the repository.

> [!NOTE]
> This is a practical research release. The public files cover the project-owned pipeline pieces; model-heavy stages still require external checkouts, local datasets, and checkpoints.

## Pipeline at a glance

```mermaid
flowchart LR
    A[Images and metadata] --> B[Stage 1: privacy review]
    B --> C[Private/public captions and edit instructions]
    C --> D[Edited pairs and CLIP filtering]
    D --> E[Stage 2: diffusion editing]
    E --> F[Quality, privacy, and utility evaluation]
```

Stage 2 can use the released project adapters for InstructPix2Pix, OminiControl, and FlowEdit while their upstream model repositories remain external.

## Installation

Create an environment and install the shared Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Install a PyTorch build that matches the target CPU or CUDA platform when the default pip resolution is not suitable. Stage 1 also requires the model dependencies for the selected backend. The diffusion entry points require a compatible external diffusion checkout.

## Project layout

```text
prompts/                  Captioning, privacy, and edit-instruction prompts.
src/unsafe2safe/stage1/         Stage 1 generation, parsing, and flag evaluation.
src/unsafe2safe/stage2/         Stage 2 editor, data loader, Safe Attention, and external boundary.
src/unsafe2safe/data_prep/      Metadata and image-pair preparation utilities.
src/unsafe2safe/evaluation/     CLIP, image, privacy, caption, and utility scores.
src/unsafe2safe/adapters/       Optional FlowEdit, OminiControl, LAVIS, and VQA adapters.
src/unsafe2safe/scripts/        Download, training, and batch-inference launchers.
src/unsafe2safe/legacy/         Historical editors and optional prompt demo; not the core method.
```

The code is released in practical research form. Paths, checkpoints, and model choices are explicit where possible, but the model-heavy stages still require compatible external installations and local data.

## Stage 1: captioning and privacy instructions

The default configuration uses an InternVL backend for image captioning and privacy flags, and a Qwen text backend for edit instructions and caption combination.

Expected local layout:

```text
data/mscoco/                  Source images.
metadata/mscoco.csv           CSV containing a file column.
outputs/mscoco/               Generated JSON files.
```

Generate privacy-aware captions:

```bash
python src/unsafe2safe/stage1/run_stage1.py \
  --config src/unsafe2safe/stage1/configs/stage1.yaml \
  --purpose generate_captions \
  --dataset mscoco
```

Compare original and anonymized images:

```bash
python src/unsafe2safe/stage1/run_stage1.py \
  --config src/unsafe2safe/stage1/configs/eval.yaml \
  --purpose compare_anonymization \
  --dataset mscoco
```

Collect generated captions into one CSV:

```bash
python -m unsafe2safe.stage1.collect_captions \
  --captions-dir outputs/mscoco/generate_captions \
  --metadata metadata/mscoco.csv \
  --output metadata/mscoco_with_captions.csv \
  --parse-structured
```

The `generate_edit_instructions` profile reads the collected CSV and maps its
`PUBLIC_CAPTION` field to the `{public_caption}` prompt argument. Collect its
outputs into a second manifest for the `combine_caption_and_edit` profile:

```bash
python -m unsafe2safe.stage1.collect_captions \
  --captions-dir outputs/mscoco/generate_edit_instructions \
  --metadata metadata/mscoco_with_captions.csv \
  --output metadata/mscoco_with_edit_instructions.csv \
  --output-column EDIT_INSTRUCTION
```

Evaluate privacy flags against VISPR annotations:

```bash
python -m unsafe2safe.stage1.evaluate_flags metadata/vispr_predictions.csv data/vispr/annotations
```

The flag-evaluation CSV must contain `file` and `PRIVACY_FLAG` columns. Structured model responses can also be parsed directly:

```python
from unsafe2safe.stage1.output_parser import parse_structured_output

parsed = parse_structured_output(model_response)
```

## Dataset preparation

The dataset helpers in `src/unsafe2safe/data_prep/` prepare image and text inputs for the captioning and editing stages. To keep edited pairs semantically aligned, filter CLIP scores with the normalized threshold used by the project:

```bash
python src/unsafe2safe/data_prep/filter_dataset.py \
  scores.csv filtered_scores.csv \
  --threshold 0.7
```

The input score CSV should contain `clip_orig` and `clip_edit` columns. Rows are kept when `clip_edit / clip_orig` is greater than the threshold.

## Editing and training

The original diffusion implementation is not vendored. For the InstructPix2Pix path, use a clean external checkout and keep its checkpoints outside this repository. The project adapter imports the external code and applies the project-specific model changes in memory.

Train with the example configuration:

```bash
./src/unsafe2safe/scripts/train_unsafe2safe.sh \
  /path/to/instruct-pix2pix \
  src/unsafe2safe/stage2/configs/train_unsafe2safe.yaml \
  /path/to/logs \
  0,1,2,3
```

Run batch editing from the repository root:

```bash
./src/unsafe2safe/scripts/run_unsafe2safe.sh \
  INPUT_CSV OUTPUT_DIR CHECKPOINT IMAGE_ROOT
```

The batch editor expects the external diffusion checkout at `stable_diffusion/` for its legacy runtime path, together with a compatible config and checkpoint. See `src/unsafe2safe/README.md` for the external checkout details and data columns used by the training loader.

The project also contains an Unsafe2Safe-specific OminiControl adapter in [`src/unsafe2safe/adapters/ominicontrol/`](src/unsafe2safe/adapters/ominicontrol/README.md). OminiControl and FLUX remain external dependencies; their upstream source is not copied or modified here.

The project also contains a minimal FlowEdit adapter in
[`src/unsafe2safe/adapters/flowedit/`](src/unsafe2safe/adapters/flowedit/README.md). FlowEdit remains an
external MIT-licensed dependency.

See [`THIRD_PARTY.md`](THIRD_PARTY.md) for upstream revisions, installation
boundaries, attribution, and license status.

## Evaluation

The reusable metric helpers in `src/unsafe2safe/evaluation/` cover the project’s current public evaluation surface:

- CLIP and directional CLIP similarity.
- SSIM and LPIPS image similarity.
- Nearest-counterpart FaceSim.
- Token-set TextSim and normalized Race Entropy.
- VLM anonymization score collection.
- BLEU-4 and CIDEr captioning scores.
- Downstream top-1 classification accuracy.

Keep downloaded datasets, checkpoints, generated images, and experiment outputs outside version control. The repository ignores common `outputs/` and `runs/` directories.

## Downstream VQA

The Qwen3-VL OK-VQA training and prediction scripts are under [`src/unsafe2safe/adapters/vqa/`](src/unsafe2safe/adapters/vqa/README.md). They accept dataset roots, safe/private manifests, adapter paths, and output paths as command-line arguments.

## Links

- [Dataset on Hugging Face](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe)
- [Project page](https://see-ai-lab.github.io/unsafe2safe/)
- [Paper](https://arxiv.org/abs/2603.28605)

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
