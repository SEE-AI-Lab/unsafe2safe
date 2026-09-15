# Pipeline code

This directory contains the paper-owned dataset, editing, evaluation, and Safe Attention code.

The diffusion entry points expect a compatible Stable Diffusion or InstructPix2Pix checkout at `stable_diffusion/`, along with the research configuration files and checkpoints. The base diffusion repository is intentionally not vendored here.

`safe_attention.py` contains the project-specific private-caption filtering and public-caption cross-attention layers. The surrounding editor scripts preserve the original research workflow and are released in rough form for later cleanup.

`unsafe2safe_model.py` provides the project-specific training wrapper on top of the external diffusion base.

The dataset and metrics helpers can be used independently with local image and metadata paths.

To keep edited training pairs with enough semantic overlap, first compute the original and edited CLIP scores, then run:

```bash
python pipeline/dataset_creation/filter_dataset.py scores.csv filtered_scores.csv --threshold 0.7
```

The default threshold follows the paper's MS-COCO filtering step.

Collect VLM anonymization scores from generated caption JSON files:

```bash
python pipeline/metrics/vlm_score.py outputs/scores outputs/vlm_scores.json
```

`pipeline/metrics/face_similarity.py` also exposes `nearest_face_similarity`, which follows the paper's nearest-counterpart FaceSim definition.
