# Unsafe2Safe implementation

This directory contains the project-owned dataset, editing, training adapters, and evaluation helpers. External model repositories and checkpoints are not copied into this repository.

## InstructPix2Pix integration

The InstructPix2Pix path expects a clean external checkout. The current local reproduction used [`timothybrooks/instruct-pix2pix`](https://github.com/timothybrooks/instruct-pix2pix) at commit `0dffd1e`.

```bash
git clone https://github.com/timothybrooks/instruct-pix2pix.git /path/to/instruct-pix2pix
git -C /path/to/instruct-pix2pix checkout 0dffd1e
```

`stage2/external.py` defines the external import boundary. `stage2/model.py` and `stage2/attention.py` provide the project-specific model integration without modifying the external checkout.

The training path is intentionally three small pieces:

1. `stage2/data.py` returns an unsafe image, its public target, a public caption, and an edit instruction.
2. `stage2/model.py` encodes the two texts and passes them to the UNet as `(public, edit)`.
3. `stage2/attention.py` keeps the upstream edit attention and adds public features
   through a learned per-query gate.

The attention module is intentionally a small reference implementation. Its
gate is a compact approximation of the full attention-map fuser described in
the paper appendix; it should not be treated as an exact reproduction of that
fuser.

Only the UNet adapter is project-specific; the VAE, CLIP encoder, trainer, and checkpoint still come from the external InstructPix2Pix checkout.

Train with the example configuration:

```bash
./pipeline/scripts/train_unsafe2safe.sh \
  /path/to/instruct-pix2pix \
  pipeline/stage2/configs/train_unsafe2safe.yaml \
  /path/to/logs \
  0,1,2,3
```

Relative paths in the example config resolve from the repository root. Put
local data, metadata, checkpoints, and logs in those locations or replace
them with absolute paths in a copy of the config.

The training CSV columns are configured explicitly in
`stage2/configs/train_unsafe2safe.yaml`: `file_column` identifies the paired image
path, `public_caption_column` identifies the privacy-safe caption, and
`edit_caption_column` identifies the edit instruction. Use whatever column
names your manifest already has.

## Dataset filtering

Filter edited pairs by normalized CLIP similarity:

```bash
python pipeline/data_prep/filter_dataset.py \
  scores.csv filtered_scores.csv \
  --threshold 0.7
```

The input CSV must contain `clip_orig` and `clip_edit`. A row is kept when `clip_edit / clip_orig` is greater than the threshold.

## Optional ImageMAE dataset

`adapters/image_mae/dataset.py` is the project-specific downstream classification
dataset, separate from `stage2/data.py` used by the diffusion editor.
It reads `file`, `class`, `split`, and optional `PRIVACY_FLAG` columns, selects
the original or edited image root, applies ImageNet preprocessing, and returns
`(image, class_id)` samples. The ImageMAE model and trainer remain in the
separately installed upstream checkout. See
[`configs/image_mae_example.yaml`](configs/image_mae_example.yaml).
When creating separate train and validation datasets, pass the same
`class_to_idx` mapping to both instances so class IDs remain stable.

## OminiControl adapter

`adapters/ominicontrol/` contains only the Unsafe2Safe-specific dataset adapter and launch wrappers. Install OminiControl separately, set `OMINICONTROL_ROOT`, and follow [`adapters/ominicontrol/README.md`](adapters/ominicontrol/README.md). The upstream OminiControl and FLUX source remain external.

## FlowEdit adapter

`adapters/flowedit/` contains the Unsafe2Safe CSV-to-caption mapping and portable batch
inference wrapper for an external FlowEdit checkout. It follows the paper's
SD3 configuration and imports the upstream sampler at runtime. See
[`adapters/flowedit/README.md`](adapters/flowedit/README.md) for the pinned revision, data
schema, and reproduction command.

## Evaluation helpers

The reusable modules under `evaluation/` provide:

- CLIP and directional CLIP similarity.
- SSIM and LPIPS image similarity.
- Nearest-counterpart FaceSim.
- Token-set TextSim and normalized Race Entropy.
- VLM anonymization score collection.
- BLEU-4 and CIDEr captioning scores.
- Downstream top-1 classification accuracy.

For example, collect VLM scores from generated caption JSON files:

```bash
python pipeline/evaluation/vlm_score.py outputs/scores outputs/vlm_scores.json
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
python -m pipeline.adapters.lavis.blip2_captioning \
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
`pipeline/evaluation/caption_scores.py`.
