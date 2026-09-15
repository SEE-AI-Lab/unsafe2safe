# Pipeline

This directory contains the project-owned dataset, editing, training adapters, evaluation, and prompt-demo code. External model repositories and checkpoints are not copied into this repository.

## InstructPix2Pix integration

The InstructPix2Pix path expects a clean external checkout. The current local reproduction used [`timothybrooks/instruct-pix2pix`](https://github.com/timothybrooks/instruct-pix2pix) at commit `0dffd1e`.

```bash
git clone https://github.com/timothybrooks/instruct-pix2pix.git /path/to/instruct-pix2pix
git -C /path/to/instruct-pix2pix checkout 0dffd1e
```

`instruct_pix2pix.py` defines the external import boundary. `unsafe2safe_model.py` and `safe_attention.py` provide the project-specific model integration without modifying the external checkout.

The training path is intentionally three small pieces:

1. `unsafe2safe_dataset.py` returns an unsafe image, its public target, a public caption, and an edit instruction.
2. `unsafe2safe_model.py` encodes the two texts and passes them to the UNet as `(public, edit)`.
3. `safe_attention.py` keeps the upstream edit attention and adds the learned public-caption branch.

Only the UNet adapter is project-specific; the VAE, CLIP encoder, trainer, and checkpoint still come from the external InstructPix2Pix checkout.

Train with the example configuration:

```bash
./pipeline/scripts/train_unsafe2safe.sh \
  /path/to/instruct-pix2pix \
  pipeline/configs/train_unsafe2safe.yaml \
  /path/to/logs \
  0,1,2,3
```

The training CSV should contain `file`, `c1`, and `caption`. The loader treats `c1` as the public semantic caption and `caption` as the edit instruction. Older exports may call `c1` `priv_caption`.

Run the legacy batch editor from the repository root:

```bash
./pipeline/scripts/run_unsafe2safe.sh INPUT_CSV OUTPUT_DIR CHECKPOINT IMAGE_ROOT
```

That editor expects the external diffusion checkout at `stable_diffusion/`, a compatible config, and a checkpoint. Keep all three outside version control when possible.

## Dataset filtering

Filter edited pairs by normalized CLIP similarity:

```bash
python pipeline/dataset_creation/filter_dataset.py \
  scores.csv filtered_scores.csv \
  --threshold 0.7
```

The input CSV must contain `clip_orig` and `clip_edit`. A row is kept when `clip_edit / clip_orig` is greater than the threshold.

## Optional ImageMAE dataset

`image_mae_dataset.py` is the project-specific downstream classification
dataset, separate from `unsafe2safe_dataset.py` used by the diffusion editor.
It reads `file`, `class`, `split`, and optional `PRIVACY_FLAG` columns, selects
the original or edited image root, applies ImageNet preprocessing, and returns
`(image, class_id)` samples. The ImageMAE model and trainer remain in the
separately installed upstream checkout. See
[`configs/image_mae_example.yaml`](configs/image_mae_example.yaml).

## OminiControl adapter

`ominicontrol/` contains only the Unsafe2Safe-specific dataset adapter and launch wrappers. Install OminiControl separately, set `OMINICONTROL_ROOT`, and follow [`ominicontrol/README.md`](ominicontrol/README.md). The upstream OminiControl and FLUX source remain external.

## FlowEdit adapter

`flowedit/` contains the Unsafe2Safe CSV-to-caption mapping and portable batch
inference wrapper for an external FlowEdit checkout. It follows the paper's
SD3 configuration and imports the upstream sampler at runtime. See
[`flowedit/README.md`](flowedit/README.md) for the pinned revision, data
schema, and reproduction command.

## Evaluation helpers

The reusable modules under `metrics/` provide:

- CLIP and directional CLIP similarity.
- SSIM and LPIPS image similarity.
- Nearest-counterpart FaceSim.
- Token-set TextSim and normalized Race Entropy.
- VLM anonymization score collection.
- BLEU-4 and CIDEr captioning scores.
- Downstream top-1 classification accuracy.

For example, collect VLM scores from generated caption JSON files:

```bash
python pipeline/metrics/vlm_score.py outputs/scores outputs/vlm_scores.json
```

## BLIP-2 captioning evaluation

The historical BLIP-2 experiment used Salesforce LAVIS for COCO captioning.
LAVIS remains an external dependency; this repository contains only the
Unsafe2Safe image-selection adapter.  Install the pinned LAVIS checkout and
its dependencies in a separate environment because LAVIS pins an older
Transformers release than the Stage 1 environment:

```bash
git clone https://github.com/salesforce/LAVIS.git /path/to/LAVIS
git -C /path/to/LAVIS checkout baad2d7c8df599d8d9b081ba2e946626eaa2dc34
pip install -e /path/to/LAVIS
```

Start from LAVIS's
`lavis/projects/blip2/train/caption_coco_ft.yaml`.  The input annotations
must be LAVIS unified JSON lists with at least `image`, `caption`, and
`image_id` fields.  A safe manifest contains a `file` column naming the
source-relative paths that have anonymized copies.  A private manifest uses
the same column and marks images that must not silently fall back to their
original pixels.

Prepare annotations for a local run:

```bash
python -m pipeline.blip2_captioning \
  --train-annotations /path/to/coco_train.json \
  --val-annotations /path/to/coco_val.json \
  --test-annotations /path/to/coco_test.json \
  --original-root /path/to/coco \
  --safe-root /path/to/unsafe2safe-coco \
  --safe-manifest /path/to/safe_images.csv \
  --private-manifest /path/to/private_images.csv \
  --output-dir /tmp/unsafe2safe-blip2-annotations \
  --check-files
```

The adapter selects a safe image when listed in the safe manifest, retains an
original image otherwise, and drops private images with no safe counterpart.
Omit both manifests for an original-image baseline.  It writes absolute paths
into the generated local annotations so vanilla LAVIS can read mixed original
and safe roots without a patched dataset class.

Train through the portable wrapper (the historical run used four processes):

```bash
NPROC_PER_NODE=4 ./pipeline/scripts/train_blip2_captioning.sh \
  /path/to/LAVIS \
  /path/to/LAVIS/lavis/projects/blip2/train/caption_coco_ft.yaml \
  /tmp/unsafe2safe-blip2-annotations/train.json \
  /tmp/unsafe2safe-blip2-annotations/val.json \
  /tmp/unsafe2safe-blip2-annotations/test.json \
  /path/to/coco
```

The example config uses LAVIS's BLIP-2 captioning recipe; it controls the
model, optimizer, resolution, and checkpoint output.  The paper evaluates
generated captions with BLEU-4 and CIDEr; the reusable helper is
`pipeline/metrics/caption_scores.py`.

The prompt demo is optional and requires its own `datasets`, `gradio`, and `openai` installation:

```bash
python pipeline/prompt_app.py --openai-api-key "$OPENAI_API_KEY" --openai-model MODEL_NAME
```
