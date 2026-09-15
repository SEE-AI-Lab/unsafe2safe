# Unsafe2Safe

[![Paper](https://img.shields.io/badge/arXiv-2603.28605-b31b1b.svg)](https://arxiv.org/abs/2603.28605)
[![Project Page](https://img.shields.io/badge/Project%20Page-online-0a7ea4.svg)](https://see-ai-lab.github.io/unsafe2safe/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d.svg)](https://huggingface.co/datasets/minhdinh2/Unsafe2Safe)

Unsafe2Safe creates privacy-preserving image edits while preserving the useful visual content of the source image.

This repository contains the project-specific code for both paper stages,
prompt assets, evaluation helpers, and adapters for the downstream experiments.
Model weights and external trainers are installed separately.

> [!NOTE]
> The commands below use example paths. Replace them with the locations of your
> local datasets, checkpoints, and external model repositories.

## Contents

- [Installation](#installation)
- [Project layout](#project-layout)
- [Stage 1: captioning and privacy instructions](#stage-1-captioning-and-privacy-instructions)
- [CLIP filtering](#clip-filtering)
- [Stage 2: SafeAttention editing](#stage-2-safeattention-editing)
- [Other editing adapters](#other-editing-adapters)
- [Evaluation](#evaluation)
- [Downstream experiments](#downstream-experiments)
- [External dependencies](#external-dependencies)
- [Links](#links)
- [Citation](#citation)

The repository follows the paper's two-stage structure: Stage 1 creates the
privacy-safe text conditions, and Stage 2 trains the SafeAttention editor.

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
prompts/                    Captioning, privacy, and edit-instruction prompts.
pipeline/stage1/            Stage 1 generation, parsing, and flag evaluation.
pipeline/stage2/            Stage 2 editor, data loader, SafeAttention, and external boundary.
pipeline/filter_dataset.py  CLIP filtering for generated image pairs.
pipeline/evaluation/        CLIP, image, privacy, caption, and utility scores.
pipeline/adapters/          Optional FlowEdit, OminiControl, Face Anon Simple, LAVIS, and VQA adapters.
pipeline/scripts/           Training and adapter launchers.
```

The commands below assume they are run from the repository root. Change the
data, metadata, checkpoint, and output paths to match your local setup.

Paper-aligned handoff configs are kept beside the code they configure:

- `pipeline/stage1/config.yaml`: InternVL/Qwen captioning, flagging, and comparison jobs.
- `pipeline/stage2/configs/`: SafeAttention/InstructPix2Pix training.
- `pipeline/adapters/*/config.example.yaml`: FreePrompt, FlowEdit, OminiControl,
  ImageMAE, BLIP-2, and Qwen3-VL recipes.
- `pipeline/adapters/baselines.example.yaml` and
  `pipeline/evaluation/config.example.yaml`:
  external baselines and the evaluation checklist.

## Stage 1: captioning and privacy instructions

The default configuration uses an InternVL backend for image captioning and privacy flags, and a Qwen text backend for edit instructions and caption combination. Raw/anonymized image comparison is an evaluation step described below.

Expected local layout:

```text
data/mscoco/                  Source images.
metadata/mscoco.csv           CSV containing a file column.
outputs/mscoco/               Generated JSON files.
```

Generate privacy-aware captions:

```bash
python pipeline/stage1/run_stage1.py \
  --config pipeline/stage1/config.yaml \
  --purpose generate_captions \
  --dataset mscoco
```

Collect generated captions into one CSV:

```bash
python -m pipeline.stage1.collect_captions \
  --captions-dir outputs/mscoco/generate_captions \
  --metadata metadata/mscoco.csv \
  --output metadata/mscoco_with_captions.csv \
  --parse-structured
```

The `generate_edit_instructions` profile reads the collected CSV and maps its
`PUBLIC_CAPTION` field to the `{public_caption}` prompt argument. Collect its
outputs into a second manifest for the `combine_caption_and_edit` profile:

```bash
python pipeline/stage1/run_stage1.py \
  --config pipeline/stage1/config.yaml \
  --purpose generate_edit_instructions \
  --dataset mscoco
```

```bash
python -m pipeline.stage1.collect_captions \
  --captions-dir outputs/mscoco/generate_edit_instructions \
  --metadata metadata/mscoco_with_captions.csv \
  --output metadata/mscoco_with_edit_instructions.csv \
  --output-column EDIT_INSTRUCTION
```

Optionally combine the public caption and edit instruction into one caption for
workflows that use a single text condition:

```bash
python pipeline/stage1/run_stage1.py \
  --config pipeline/stage1/config.yaml \
  --purpose combine_caption_and_edit \
  --dataset mscoco
```

Generate privacy flags for VISPR images:

```bash
python pipeline/stage1/run_stage1.py \
  --config pipeline/stage1/config.yaml \
  --purpose generate_flags \
  --dataset vispr
```

Evaluate privacy flags against VISPR annotations:

```bash
python -m pipeline.stage1.evaluate_flags metadata/vispr_predictions.csv data/vispr/annotations
```

The flag-evaluation CSV must contain `file` and `PRIVACY_FLAG` columns. Structured model responses can also be parsed directly:

```python
from pipeline.stage1.output_parser import parse_structured_output

parsed = parse_structured_output(model_response)
```

## CLIP filtering

The helper in `pipeline/filter_dataset.py` filters generated image pairs with the normalized CLIP threshold used by the project:

```bash
python pipeline/filter_dataset.py \
  scores.csv filtered_scores.csv \
  --threshold 0.7
```

The input score CSV should contain `clip_orig` and `clip_edit` columns. Rows are kept when `clip_edit / clip_orig` is greater than the threshold.

## Stage 2: SafeAttention editing

Stage 2 is the paper's SafeAttention editor. It takes the unsafe image, a
privacy-safe caption, and an edit instruction, then learns to produce the safe
image. The project-specific attention code is in `pipeline/stage2/`. The
InstructPix2Pix trainer and model files are installed separately; setup is
described in the [pipeline README](pipeline/README.md#instructpix2pix-integration).

The implementation has three parts:

1. `stage2/data.py` loads paired images and the two text conditions.
2. `stage2/model.py` sends the public caption and edit instruction to the UNet.
3. `stage2/attention.py` fuses their attention maps and applies the fused map
   to the public-caption values.

Train with the example configuration:

```bash
./pipeline/scripts/train_unsafe2safe.sh \
  /path/to/instruct-pix2pix \
  pipeline/stage2/configs/train_unsafe2safe.yaml \
  /path/to/logs \
  0,1,2,3
```

Before training, set `ckpt_path`, `path`, `target_path`, and `csv_path` in
[`pipeline/stage2/configs/train_unsafe2safe.yaml`](pipeline/stage2/configs/train_unsafe2safe.yaml)
to your local files.

The training manifest needs `file`, `public_caption`, and `edit_instruction`
columns. In the config, `path` is the unsafe image root and `target_path` is
the safe image root.

## Other editing adapters

The optional [OminiControl adapter](pipeline/adapters/ominicontrol/README.md)
provides the paper's FLUX-based alternative using the project’s unsafe/safe
dataset mapping. The [FlowEdit adapter](pipeline/adapters/flowedit/README.md)
provides the paper's SD3 condition mapping. [Face Anon Simple](pipeline/adapters/face_anon_simple/README.md)
is an optional external baseline.

The [baseline config](pipeline/adapters/baselines.example.yaml) records the
external FreePrompt and DeepPrivacy2 handoffs.

## Evaluation

Evaluation is split between the Stage 1 outputs, generated image pairs, and
downstream task results. Use the same relative `file` names in the original
and edited roots.

There is no single evaluator command because each metric uses a different
model or input format. The public helpers are:

| Result | Helper | Input |
| --- | --- | --- |
| VLM anonymization score | `compare_anonymized.py`, `vlm_score.py` | raw/anonymized image pairs |
| CLIP similarity | `ClipSimilarity` | image and caption batches |
| Directional CLIP | `compute_directional_clip_score` | original/edited images and captions |
| SSIM and LPIPS | `compute_ssim`, `compute_lpips` | one original/edited path pair |
| FaceSim | `nearest_face_similarity` | two image roots and file names |
| TextSim and Race Entropy | `token_set_similarity`, `normalized_race_entropy` | text or predicted race labels |
| Captioning | `compute_caption_scores` | predictions and references |
| Classification | `top1_accuracy` | model and labeled dataloader |

Compare raw and anonymized images with the InternVL judge:

```bash
python -m pipeline.evaluation.compare_anonymized \
  --input-csv metadata/mscoco_pairs.csv \
  --raw-root data/mscoco \
  --anonymized-root data/mscoco_anonymized \
  --output-dir outputs/mscoco/compare_anonymization \
  --prompt prompts/intern_image_flagging-compare.txt
```

The command writes one `_caption.json` file per pair. Collect the scores from
those files:

```bash
python pipeline/evaluation/vlm_score.py \
  outputs/mscoco/compare_anonymization \
  outputs/mscoco/vlm_scores.json
```

The other metrics are small Python functions. For example, score one matching
original/edited pair with SSIM and LPIPS:

```python
from pathlib import Path

from pipeline.evaluation.image_similarity import compute_lpips, compute_ssim

file = "val2014/COCO_val2014_000000000042.jpg"
original = Path("/path/to/coco") / file
edited = Path("/path/to/unsafe2safe-images") / file
print({"ssim": compute_ssim(original, edited), "lpips": compute_lpips(original, edited)})
```

Apply the same calls to every `file` in your evaluation manifest to build a
CSV or JSON table. The CLIP, directional CLIP, FaceSim, privacy, and caption
helpers are used the same way; their function names are listed in
`pipeline/evaluation/`. The [evaluation config](pipeline/evaluation/config.example.yaml)
lists the expected roots and model inputs. The [pipeline README](pipeline/README.md)
contains the full BLIP-2/LAVIS preparation and training commands.

## Downstream experiments

The repository includes project-specific data adapters and handoff configs for
the downstream experiments in the paper:

- [ImageMAE](pipeline/adapters/image_mae/config.example.yaml) classification
  data and paper settings.
- [BLIP-2/LAVIS](pipeline/adapters/lavis/config.example.yaml) captioning data
  routing and launcher settings.
- [Qwen3-VL OK-VQA](pipeline/adapters/vqa/config.example.yaml) training and
  evaluation settings.

The BLIP-2/LAVIS and Qwen3-VL adapters have detailed command examples in
their READMEs. ImageMAE is a dataset adapter plus a handoff config for the
external trainer.

## External dependencies

See [`THIRD_PARTY.md`](THIRD_PARTY.md) for the external repositories used by
the optional workflows. They are not included in this repository.

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
