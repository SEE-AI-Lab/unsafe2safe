# Unsafe2Safe adapter for Face Anon Simple

This directory contains the project-owned batch adapter for [Face Anonymization
Made Simple](https://github.com/hanweikung/face_anon_simple). The upstream
ReferenceNet/Diffusers implementation, face extractor, checkpoints, and
sample data remain in a separately installed checkout.

## Install the upstream dependency

The audited upstream revision is
`c36f276352873827e9d559ee8d130b7563491171`:

```bash
git clone https://github.com/hanweikung/face_anon_simple.git /path/to/face_anon_simple
git -C /path/to/face_anon_simple checkout c36f276352873827e9d559ee8d130b7563491171
conda env create -f /path/to/face_anon_simple/environment.yml
conda activate face-anon-simple
python -m pip install -e /absolute/path/to/unsafe2safe
export FACE_ANON_SIMPLE_ROOT=/path/to/face_anon_simple
```

The upstream environment pins Python 3.8, Diffusers 0.25.1, PyTorch 2.1,
Transformers 4.46.1, `face-alignment` 1.4.1, and the other model/runtime
dependencies. Use the upstream environment as the compatibility baseline.
Model downloads use the normal Hugging Face cache configuration; credentials
are not passed through this repository.

## Data contract and generation

The adapter recursively reads images below `--input-root` and writes each
result below `--output-root` with the same relative path. Existing outputs are
skipped unless `--overwrite` is supplied. Images with no detected faces are
copied unchanged, matching the project-specific behavior in the historical
Unsafe2Safe runner.

```bash
python -m pipeline.adapters.face_anon_simple.generate_unsafe2safe \
  --input-root /path/to/coco/train2014 \
  --output-root /path/to/outputs/MSCOCO_FaceAnon \
  --cache-dir /path/to/huggingface-cache \
  --steps 25 \
  --guidance-scale 4.0 \
  --anonymization-degree 1.25
```

The defaults use `hkung/face-anon-simple`,
`stabilityai/stable-diffusion-2-1`, and
`openai/clip-vit-large-patch14`, at 512x512 face crops. Set `--device cpu`
only for small smoke checks; practical inference requires a CUDA-capable
machine. Use `--limit` to validate the wiring on a small subset.

## Provenance and license boundary

The public adapter owns only model loading configuration, portable image
discovery/output mapping, and the no-face guard. `src/diffusers/`, the
ReferenceNet models, `utils/extractor.py`, `utils/merger.py`, upstream demo and
training code, checkpoints, and datasets are intentionally not copied here.

Face Anon Simple is released under the upstream [AGPL-3.0
license](https://github.com/hanweikung/face_anon_simple/blob/main/LICENSE);
retain and comply with that license when obtaining or modifying the external
checkout. The Unsafe2Safe project has no top-level license decision yet.
